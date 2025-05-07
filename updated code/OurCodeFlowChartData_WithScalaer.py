import math
import logging
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
# Import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

plt.ion()

# --- Dataset (No changes needed here) ---
class TimeSeriesDataset(Dataset):
    def __init__(self, series, input_len, pred_len):
        self.series = series
        self.input_len = input_len
        self.pred_len = pred_len
        # Ensure series is numpy array for consistent processing later
        if isinstance(self.series, torch.Tensor):
            self.series = self.series.cpu().numpy()
        self.total_samples = len(self.series) - self.input_len - self.pred_len + 1
        logger.info(
            f"TimeSeriesDataset init: series_shape={series.shape}, "
            f"input_len={input_len}, pred_len={pred_len}, total_samples={self.total_samples}"
        )

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        x = self.series[idx : idx + self.input_len]
        y = self.series[idx + self.input_len : idx + self.input_len + self.pred_len]
        return (
            torch.tensor(x, dtype=torch.float32, device=device),
            torch.tensor(y, dtype=torch.float32, device=device)
        )

# --- Transformer Components (No changes needed here) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500): # Increased max_len just in case
        super().__init__()
        pe = torch.zeros(max_len, d_model) # Removed device here, will be moved later
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(1) # Shape: (max_len, 1, d_model)
        # Register buffer AFTER potential device move in main model
        self.register_buffer('pe', pe, persistent=False) # persistent=False if not saving with state_dict

    def forward(self, x):
        # x shape: (sequence_length, batch_size, d_model)
        # self.pe is (max_len, 1, d_model). Device handled by main model's .to(device)
        return x + self.pe[:x.size(0)].to(x.device) # Ensure PE is on same device as x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        # Defer .to(device) until main model call
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(self.d_k)

    def forward(self, q, k, v):
        B = q.size(1) # Batch size
        # Apply linear layers
        Q = self.Wq(q); K = self.Wk(k); V = self.Wv(v)
        # Reshape for multi-head: (seq_len, B, d_model) -> (seq_len, B, num_heads, d_k) -> (B, num_heads, seq_len, d_k)
        Qh = Q.view(q.size(0), B, self.num_heads, self.d_k).permute(1, 2, 0, 3)
        Kh = K.view(k.size(0), B, self.num_heads, self.d_k).permute(1, 2, 0, 3)
        Vh = V.view(v.size(0), B, self.num_heads, self.d_k).permute(1, 2, 0, 3)

        # Scaled dot-product attention: (B, num_heads, seq_len_q, d_k) * (B, num_heads, d_k, seq_len_k) -> (B, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(Qh, Kh.transpose(-2,-1)) / self.scale
        attn = torch.softmax(scores, dim=-1)

        # Apply attention to V: (B, num_heads, seq_len_q, seq_len_k) * (B, num_heads, seq_len_v, d_k) -> (B, num_heads, seq_len_q, d_k) [Note: seq_len_k == seq_len_v]
        ctx = torch.matmul(attn, Vh)

        # Concatenate heads: (B, num_heads, seq_len_q, d_k) -> (B, seq_len_q, num_heads, d_k) -> (B, seq_len_q, d_model) -> (seq_len_q, B, d_model)
        ctx2 = ctx.permute(0, 2, 1, 3).contiguous().view(B, q.size(0), self.num_heads * self.d_k)
        ctx2 = ctx2.permute(1, 0, 2) # Back to (seq_len, B, d_model)

        # Final linear layer
        return self.Wo(ctx2)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedfwd=256, dropout=0.1):
        super().__init__()
        # Defer .to(device)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, dim_feedfwd),
            nn.ReLU(),
            nn.Linear(dim_feedfwd, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention sublayer
        attn_output = self.attn(x, x, x)
        x = x + self.drop(attn_output) # Residual connection
        x = self.norm1(x) # Layer norm

        # Feed-forward sublayer
        ff_output = self.ff(x)
        x = x + self.drop(ff_output) # Residual connection
        x = self.norm2(x) # Layer norm
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, num_heads, num_layers, pred_len, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        # Defer .to(device) until model instantiation
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model) # PositionalEncoding handles its buffer
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(d_model, input_size)

    def forward(self, src):
        # src shape: (input_len, batch_size, input_size)
        x = self.input_proj(src)  # (input_len, batch_size, d_model)
        x = self.pos_enc(x)       # Add positional encoding
        for layer in self.layers:
            x = layer(x)          # (input_len, batch_size, d_model)

        # Select last 'pred_len' outputs and project
        output_embeddings = x[-self.pred_len:] # (pred_len, batch_size, d_model)
        predictions = self.output_proj(output_embeddings) # (pred_len, batch_size, input_size)
        return predictions

# --- Training (No changes needed here) ---
def train(model, loader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(1, epochs+1):
        epoch_loss = 0.0
        for xb, yb in loader:
            # xb shape: (batch_size, input_len, num_features)
            # yb shape: (batch_size, pred_len, num_features)
            x = xb.permute(1, 0, 2).to(device) # Reshape and move to device
            y = yb.permute(1, 0, 2).to(device) # Reshape and move to device

            optimizer.zero_grad()
            preds = model(x) # preds shape: (pred_len, batch_size, num_features)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        logger.info(f"Epoch {epoch}: avg loss {epoch_loss/len(loader):.6f}")

# --- Evaluation (Modified to handle scaling) ---
def evaluate(model, loader, criterion, scaler): # Pass scaler here
    model.eval()
    preds_scaled_all, actual_scaled_all = [], []
    with torch.no_grad():
        for xb, yb in loader:
            # xb, yb are already scaled from the Dataset
            x = xb.permute(1, 0, 2).to(device) # Reshape and move to device
            y_scaled = yb.permute(1, 0, 2).cpu().numpy() # Keep on CPU for numpy conversion

            preds_scaled = model(x).cpu().numpy() # Get scaled preds from model

            preds_scaled_all.append(preds_scaled)
            actual_scaled_all.append(y_scaled)

    # Concatenate batches: result shape (pred_len, total_samples_in_loader, num_features)
    preds_scaled_all = np.concatenate(preds_scaled_all, axis=1)
    actual_scaled_all = np.concatenate(actual_scaled_all, axis=1)

    # Reshape for scaler: (pred_len * total_samples_in_loader, num_features)
    num_features = preds_scaled_all.shape[2]
    preds_scaled_flat = preds_scaled_all.transpose(1, 0, 2).reshape(-1, num_features)
    actual_scaled_flat = actual_scaled_all.transpose(1, 0, 2).reshape(-1, num_features)

    # ---> Inverse Transform <---
    preds_original_scale = scaler.inverse_transform(preds_scaled_flat)
    actual_original_scale = scaler.inverse_transform(actual_scaled_flat)

    # ---> Calculate Metrics on Original Scale <---
    mse = mean_squared_error(actual_original_scale, preds_original_scale)
    mae = mean_absolute_error(actual_original_scale, preds_original_scale)
    # MAPE can be problematic if actual values are zero or close to zero.
    # Filter out zeros or use a stabilized version if needed.
    non_zero_mask = actual_original_scale != 0
    if np.any(non_zero_mask):
         mape = mean_absolute_percentage_error(actual_original_scale[non_zero_mask], preds_original_scale[non_zero_mask])
    else:
         mape = np.nan # Or handle as appropriate
    r2 = r2_score(actual_original_scale, preds_original_scale)

    logger.info(f"Eval (Original Scale) MSE: {mse:.6f}, MAE: {mae:.6f}, MAPE: {mape:.6f}, R2: {r2:.6f}")
    print(f"Test Metrics (Original Scale) -> MSE: {mse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2%}, R2: {r2:.4f}")

    # Return predictions and actuals in original scale
    return preds_original_scale.reshape(preds_scaled_all.shape[1], preds_scaled_all.shape[0], num_features).transpose(1,0,2), \
           actual_original_scale.reshape(actual_scaled_all.shape[1], actual_scaled_all.shape[0], num_features).transpose(1,0,2)


# ------------------------ Main Function (Modified for Scaling) -------------------------------
def main():
    """
    Pipeline steps:
    1. Load and clean the CSV time series data.
    2. Visualize original and resampled daily mean plots.
    3. Split the dataset into training and testing.
    4. **Initialize and fit StandardScaler on training data.**
    5. **Scale training and testing data.**
    6. Create DataLoaders with scaled data.
    7. Create Transformer model with attention blocks.
    8. Train the model on scaled training data.
    9. Evaluate the model on scaled test data (inverse transforming results).
    10. Forecast the final window (scaling input, inverse transforming output).
    11. Plot predicted vs actual (original scale).
    12. Print final accuracy metrics (original scale).
    """

    global device, logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
    logger = logging.getLogger(__name__)

    # --- Device Setup ---
    # Use DirectML if available, otherwise CUDA or CPU
    try:
        import torch_directml
        if torch_directml.is_available():
            device = torch_directml.device()
            logger.info(f"Using DirectML device: {device}")
        elif torch.cuda.is_available():
             device = torch.device("cuda")
             logger.info(f"Using CUDA device: {device}")
        else:
            device = torch.device("cpu")
            logger.info(f"Using CPU device: {device}")
    except ImportError:
        if torch.cuda.is_available():
             device = torch.device("cuda")
             logger.info("torch_directml not found. Using CUDA device.")
        else:
            device = torch.device("cpu")
            logger.info("torch_directml not found. Using CPU device.")


    # --- Data Loading and Preprocessing ---
    csv_path = 'data/train.csv' # Make sure this path is correct
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        logger.error(f"Error: CSV file not found at {csv_path}")
        return # Exit if file not found

    if '5 Minutes' in df.columns:
        df.rename(columns={'5 Minutes': 'DateTime'}, inplace=True)

    try:
        df['DateTime'] = pd.to_datetime(df['DateTime'], dayfirst=True, errors='coerce') # Use errors='coerce'
        df.dropna(subset=['DateTime'], inplace=True) # Drop rows where date parsing failed
        df.set_index('DateTime', inplace=True)
    except KeyError:
         logger.error("Error: 'DateTime' column not found or could not be parsed.")
         return
    except Exception as e:
        logger.error(f"Error setting DateTime index: {e}")
        return

    # Drop columns robustly
    cols_to_drop = ['datetime', 'Unnamed: 0', 'day_of_week', '# Lane Points', '% Observed']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

    # Ensure all remaining columns are numeric, coercing errors
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True) # Drop rows with non-numeric values after coercion

    if df.empty:
        logger.error("DataFrame is empty after preprocessing. Check data quality and column names.")
        return

    logger.info(f"Loaded and cleaned data shape: {df.shape}")
    logger.info(f"Data columns: {df.columns.tolist()}")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")


    # --- Visualization (Optional) ---
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df.iloc[:, 0]) # Plot first column
    plt.title(f'First Feature: {df.columns[0]}')
    plt.xlabel('DateTime')
    plt.ylabel(df.columns[0])
    plt.tight_layout()
    plt.show()

    # Resample and plot daily means (handle potential empty resample)
    try:
        daily_mean = df.resample('D').mean()
        if not daily_mean.empty:
            plt.figure(figsize=(12, 6))
            for col in daily_mean.columns:
                plt.plot(daily_mean.index, daily_mean[col], label=col)
            plt.title('Daily Mean of Features')
            plt.xlabel('Date')
            plt.ylabel('Mean Value')
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            logger.warning("Could not generate daily mean plot (possibly too little data).")
    except Exception as e:
         logger.warning(f"Could not resample or plot daily mean: {e}")

    # --- Configurable Parameters ---
    input_len = 60         # Length of input sequence window
    pred_len = 15          # Length of prediction sequence window
    train_ratio = 0.8      # Proportion of data for training
    batch_size = 32        # Batch size for training/evaluation
    d_model = 128           # Transformer embedding dimension (divisible by num_heads)
    num_heads = 8          # Number of attention heads
    num_layers = 2         # Number of Transformer blocks
    learning_rate = 1e-4   # Learning rate (may need tuning)
    epochs = 50            # Number of training epochs (may need more)

    # --- Data Splitting and Scaling ---
    data_np = df.values.astype(np.float32)
    total_len = len(data_np)
    if total_len < input_len + pred_len:
         logger.error(f"Error: Not enough data ({total_len}) for input_len ({input_len}) + pred_len ({pred_len}).")
         return

    split_idx = int(total_len * train_ratio)

    # Ensure test set has enough data for at least one sample
    if total_len - split_idx < input_len + pred_len:
        # Adjust split_idx to leave enough for one test sample
        split_idx = total_len - (input_len + pred_len)
        logger.warning(f"Adjusted split_idx to {split_idx} to ensure valid test set.")
        if split_idx <= input_len: # Still not enough for training after adjustment
             logger.error("Error: Not enough data to create both training and testing sets with specified window sizes.")
             return


    train_data_unscaled = data_np[:split_idx]
    # Overlap test data start with train data end by input_len for correct windowing
    test_data_unscaled = data_np[split_idx - input_len:]

    logger.info(f"Unscaled train data shape: {train_data_unscaled.shape}")
    logger.info(f"Unscaled test data shape: {test_data_unscaled.shape}")

    # ---> Initialize and Fit Scaler <---
    scaler = StandardScaler()
    scaler.fit(train_data_unscaled) # Fit ONLY on training data

    # ---> Scale Data <---
    train_series_scaled = scaler.transform(train_data_unscaled)
    test_series_scaled = scaler.transform(test_data_unscaled)

    logger.info(f"Scaled train series shape: {train_series_scaled.shape}")
    logger.info(f"Scaled test series shape: {test_series_scaled.shape}")

    # --- Create Datasets and DataLoaders ---
    train_ds = TimeSeriesDataset(train_series_scaled, input_len, pred_len)
    test_ds = TimeSeriesDataset(test_series_scaled, input_len, pred_len)

    # Check if datasets are empty
    if len(train_ds) == 0 or len(test_ds) == 0:
        logger.error(f"Error: TimeSeriesDataset creation resulted in zero samples. "
                     f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}. "
                     f"Check split_idx and window sizes relative to data length.")
        return

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False) # No shuffle for test

    # --- Model, Criterion, Optimizer ---
    input_size = df.shape[1] # Number of features
    model = TransformerModel(input_size, d_model, num_heads, num_layers, pred_len).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # --- Training ---
    logger.info("Starting training...")
    train(model, train_loader, optimizer, criterion, epochs=epochs)
    logger.info("Training finished.")

    # --- Evaluation ---
    logger.info("Starting evaluation...")
    # Evaluate returns preds/actuals in original scale now
    preds_eval_orig, actual_eval_orig = evaluate(model, test_loader, criterion, scaler)
    logger.info("Evaluation finished.")


    # --- Future Forecasting (Final Window) ---
    logger.info("Forecasting final window...")
    # 1. Get the last input window from the UNSCALED test data
    last_window_unscaled = test_data_unscaled[-input_len:] # Shape: (input_len, num_features)

    # 2. Scale this window using the SAME fitted scaler
    last_window_scaled = scaler.transform(last_window_unscaled)

    # 3. Convert to tensor, add batch dimension, permute, and move to device
    win_t = torch.tensor(last_window_scaled, dtype=torch.float32).unsqueeze(1).to(device)

    # 4. Get scaled predictions from the model
    model.eval()
    with torch.no_grad():
        preds_future_scaled = model(win_t).squeeze(1).cpu().numpy() # (pred_len, num_features)

    # 5. Inverse transform the predictions back to the original scale
    preds_future_original = scaler.inverse_transform(preds_future_scaled)
    num_steps_forecasted = len(preds_future_original) # Actual number of steps model predicted

    # 6. Get the corresponding actual future values (unscaled) and dates
    # --- Refined index calculation and Logging ---
    try:
        # The index in the original df corresponding to the END of the last training input window
        # This is split_idx - 1.
        # The index corresponding to the START of the first test input window is split_idx - input_len
        # The number of test samples generated is len(test_ds)
        # The start index of the LAST test input window is (split_idx - input_len) + len(test_ds) - 1
        # The start index of the TARGET (y) for the LAST test input window is
        # (split_idx - input_len) + len(test_ds) - 1 + input_len = split_idx + len(test_ds) - 1
        # This is the index in the original 'df' where the future actuals should start.
        actual_future_start_idx = split_idx + len(test_ds) - 1

        # Ensure the start index is within bounds
        if actual_future_start_idx < 0 or actual_future_start_idx >= len(df):
            raise IndexError(f"Calculated actual_future_start_idx ({actual_future_start_idx}) is out of bounds for DataFrame length ({len(df)}).")

        # Calculate end index, ensuring it doesn't exceed dataframe length
        actual_future_end_idx = min(actual_future_start_idx + num_steps_forecasted, len(df))

        logger.info(f"Attempting to get actual future data:")
        logger.info(f"  split_idx: {split_idx}")
        logger.info(f"  len(test_ds): {len(test_ds)}")
        logger.info(f"  Calculated actual_future_start_idx: {actual_future_start_idx}")
        logger.info(f"  Calculated actual_future_end_idx: {actual_future_end_idx}")

        # Slice the original dataframe to get actual values and dates
        actual_future_original = df.iloc[actual_future_start_idx:actual_future_end_idx].values
        future_dates = df.index[actual_future_start_idx:actual_future_end_idx]

        logger.info(f"  Shape of sliced actual_future_original: {actual_future_original.shape}")
        logger.info(f"  Number of future_dates obtained: {len(future_dates)}")
        if len(future_dates) > 0:
            logger.info(f"  First 5 future_dates: \n{future_dates[:5]}")
            logger.info(f"  Last 5 future_dates: \n{future_dates[-5:]}")
            # Log the actual values for the specific feature being plotted
            feature_index_to_plot = 0 # Assuming first feature
            if actual_future_original.shape[1] > feature_index_to_plot:
                 logger.info(f"  First 5 actual values (feature {feature_index_to_plot}): {actual_future_original[:5, feature_index_to_plot]}")
                 logger.info(f"  NaN check for actual feature slice: {np.isnan(actual_future_original[:, feature_index_to_plot]).all()}")
            else:
                 logger.warning(f"Feature index {feature_index_to_plot} is out of bounds for actual_future_original columns ({actual_future_original.shape[1]})")

        # Handle case where actual data is shorter than forecast
        if len(actual_future_original) < num_steps_forecasted:
            logger.warning(f"Actual future data available ({len(actual_future_original)} steps) is shorter than forecast ({num_steps_forecasted} steps). Plotting and metrics will use available actuals.")
            # We will compare only the available steps later in plotting/metrics

    except IndexError as e:
         logger.error(f"Error calculating indices or slicing for actual future data: {e}")
         logger.warning("Cannot get actual future values for comparison. Plot will only show forecast.")
         actual_future_original = np.full_like(preds_future_original, np.nan) # Placeholder with NaNs
         # Try to estimate future dates if actuals failed (less reliable)
         try:
             last_known_date = df.index[-1]
             # Try to infer frequency, default to 5 minutes if needed
             freq = pd.infer_freq(df.index[-10:]) or pd.Timedelta(minutes=5)
             future_dates = pd.date_range(start=last_known_date + freq, periods=num_steps_forecasted, freq=freq)
             logger.info(f"Estimated future dates based on last known date and frequency '{freq}'.")
         except Exception as date_err:
             logger.error(f"Could not estimate future dates: {date_err}")
             future_dates = pd.Index([]) # Empty index

    except Exception as e:
        logger.error(f"An unexpected error occurred retrieving actual future data: {e}")
        actual_future_original = np.full_like(preds_future_original, np.nan)
        future_dates = pd.Index([])
    # --- End of Refined index calculation and Logging ---


    # --- Plot Future Forecast (Single Feature - Resembling Figure_1.png) ---
    feature_index_to_plot = 0 # Plot the first feature (column 0)
    # Check if feature index is valid before proceeding
    if feature_index_to_plot >= df.shape[1]:
         logger.error(f"Feature index {feature_index_to_plot} is out of bounds for DataFrame columns ({df.shape[1]}). Cannot plot.")
         return # Or handle differently

    feature_name = df.columns[feature_index_to_plot]
    # Use the length of forecast for title, but plot length depends on available actuals/dates
    plot_title_steps = len(preds_future_original)

    logger.info(f"Plotting forecast for feature: '{feature_name}'")
    logger.info(f"Number of forecast steps to plot: {len(preds_future_original)}")
    logger.info(f"Number of actual steps to plot (max): {len(actual_future_original)}")
    logger.info(f"Number of dates available for x-axis: {len(future_dates)}")


    fig, ax = plt.subplots(figsize=(15, 4)) # Adjust figsize for a wider plot

    # Determine the number of points to actually plot based on shortest length
    # (forecast length vs number of dates vs number of actuals)
    plot_len = min(len(preds_future_original), len(future_dates), len(actual_future_original))

    # Plot Actual Data (if available and not all NaN)
    actual_data_to_plot = actual_future_original[:plot_len, feature_index_to_plot]
    if plot_len > 0 and not np.isnan(actual_data_to_plot).all():
        ax.plot(future_dates[:plot_len],
                actual_data_to_plot,
                label='Actual',
                linestyle='-',
                linewidth=2)
        logger.info(f"Plotting {plot_len} actual points.")
    elif not np.isnan(actual_future_original).all():
         # Log why actuals are not plotted if they exist but lengths mismatch
         logger.warning(f"Actual data exists but not plotting due to length mismatch or insufficient dates (plot_len={plot_len}).")
    else:
         logger.info("Actual data is all NaN or unavailable, not plotting.")


    # Plot Forecasted Data (use plot_len to align with dates/actuals)
    if plot_len > 0 :
         ax.plot(future_dates[:plot_len],
                 preds_future_original[:plot_len, feature_index_to_plot],
                 label='Forecast',
                 linestyle='-',
                 linewidth=1.5)
         logger.info(f"Plotting {plot_len} forecast points.")
    else:
         logger.warning("Not plotting forecast due to insufficient dates or zero plot length.")

    # --- Formatting ---
    ax.set_title(f'{feature_name} --- next {plot_title_steps} steps (showing {plot_len} available points)') # Adjusted title
    ax.set_xlabel('Date')
    ax.set_ylabel(feature_name)
    if plot_len > 0: # Only show legend if something was plotted
         ax.legend(loc='upper left')
    ax.tick_params(axis='x', rotation=30)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Optional: Explicitly format dates if auto-formatting fails
    # import matplotlib.dates as mdates
    # if plot_len > 0:
    #    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M')) # Example format

    plt.tight_layout()
    plt.show(block=True)


    # --- Final Metrics for Future Forecast Window (Original Scale) ---
    # Calculate metrics only on the overlapping 'plot_len' points
    if plot_len > 0 and not np.isnan(actual_future_original[:plot_len, :]).all():
        actual_compare = actual_future_original[:plot_len]
        preds_compare = preds_future_original[:plot_len]

        mse_f = mean_squared_error(actual_compare, preds_compare)
        mae_f = mean_absolute_error(actual_compare, preds_compare)
        non_zero_mask_f = actual_compare != 0
        mape_f = np.nan
        if np.any(non_zero_mask_f):
             # Calculate MAPE element-wise and then average, avoiding division by zero
             abs_pct_error = np.abs((actual_compare[non_zero_mask_f] - preds_compare[non_zero_mask_f]) / actual_compare[non_zero_mask_f])
             mape_f = np.mean(abs_pct_error) * 100 # As percentage

        r2_f = r2_score(actual_compare, preds_compare)
        print(f"\nFuture Forecast Metrics (Final Window - Original Scale):")
        print(f"Compared {plot_len} steps.")
        print(f"MSE: {mse_f:.4f}, MAE: {mae_f:.4f}, MAPE: {mape_f:.2f}%, R2: {r2_f:.4f}")
    else:
        print("\nFuture Forecast Metrics (Final Window - Original Scale):")
        print(f"Could not calculate metrics. Comparison points available: {plot_len}. Actuals available and not all NaN: {not np.isnan(actual_future_original[:plot_len, :]).all() if plot_len > 0 else False}")


if __name__ == '__main__':
    main()