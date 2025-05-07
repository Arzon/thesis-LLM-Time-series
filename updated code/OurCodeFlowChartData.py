import math
import logging
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
plt.ion() 

# --- Dataset ---------------------------------------------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, series, input_len, pred_len):
        self.series = series
        self.input_len = input_len
        self.pred_len = pred_len
        self.total_samples = len(self.series) - self.input_len - self.pred_len + 1
        logger.info(
            f"TimeSeriesDataset init: series_shape={series.shape}, "
            f"input_len={input_len}, pred_len={pred_len}, total_samples={self.total_samples}"
        )

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        x = self.series[idx : idx + self.input_len] # [10 : 40]
        y = self.series[idx + self.input_len : idx + self.input_len + self.pred_len] # [40 : 60]
        return (
            torch.tensor(x, dtype=torch.float32, device=device),
            torch.tensor(y, dtype=torch.float32, device=device)
        )

# --- Transformer Components ------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100): # max_len should be >= longest sequence
        super().__init__()
        # Create a zero matrix for positional encodings: (max_len, d_model)
        pe = torch.zeros(max_len, d_model, device=device)
				
        # Create position indices: tensor([[0.], [1.], [2.], ..., [max_len-1]])
        pos = torch.arange(0, max_len, device=device).unsqueeze(1).float()
		
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device).float() * -(math.log(10000.0) / d_model)
        )
				
        # Calculate sine for even indices (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(pos * div_term)
        # Calculate cosine for odd indices (1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(pos * div_term)

        # Add a batch dimension: shape (max_len, 1, d_model)
        pe = pe.unsqueeze(1)

        # Register 'pe' as a buffer. Buffers are part of the model's state
        # but are not updated by the optimizer during training.
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (sequence_length, batch_size, d_model)
        # Add positional encoding to the input embeddings x.
        return x + self.pe[:x.size(0)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.Wq = nn.Linear(d_model, d_model).to(device)
        self.Wk = nn.Linear(d_model, d_model).to(device)
        self.Wv = nn.Linear(d_model, d_model).to(device)
        self.Wo = nn.Linear(d_model, d_model).to(device)

        self.scale = math.sqrt(self.d_k)

    def forward(self, q, k, v): 
        B = q.size(1)
        Q = self.Wq(q); K = self.Wk(k); V = self.Wv(v)
        Qh = Q.view(-1, B, self.num_heads, self.d_k).transpose(1,2) # 30, B, 8, 8
        Kh = K.view(-1, B, self.num_heads, self.d_k).transpose(1,2)
        Vh = V.view(-1, B, self.num_heads, self.d_k).transpose(1,2)
        scores = torch.matmul(Qh, Kh.transpose(-2,-1)) / self.scale # 30, 8, B, 1
        attn = torch.softmax(scores, dim=-1)
        ctx  = torch.matmul(attn, Vh) # 30, 8, B, 8
        ctx2 = ctx.transpose(1,2).contiguous().view(-1, B, self.num_heads*self.d_k) # 30, B, 64
        return self.Wo(ctx2)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedfwd=256, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model).to(device)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, dim_feedfwd),
            nn.ReLU(),
            nn.Linear(dim_feedfwd, d_model)
        ).to(device)
        self.norm2 = nn.LayerNorm(d_model).to(device)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        a  = self.attn(x, x, x) # (30,B,64)
        x2 = self.norm1(x + self.drop(a)) # 30, B, 64
        f  = self.ff(x2) # 30, B, 64
        return self.norm2(x2 + self.drop(f)) # 30, B, 64

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, num_heads, num_layers, pred_len, dropout=0.1):
        super().__init__()
        # --- 1
        # Projects the input features (input_size) at each time step
        # into a higher-dimensional space (d_model). This creates the initial
        # "embedding" for each time step.
        self.input_proj = nn.Linear(input_size, d_model).to(device)

        # --- 2. Positional Encoding ---
        # Adds information about the position of each time step in the sequence
        self.pos_enc = PositionalEncoding(d_model)

        # --- 3. Transformer Blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.pred_len = pred_len # Store prediction length

        # --- 4
        self.output_proj = nn.Linear(d_model, input_size).to(device)

    def forward(self, src):
        # src shape: (input_len, batch_size, input_size)

        x = self.input_proj(src)
        # Example: If input_len=30, batch=32, input_size=1, d_model=64
        # src shape: (30, 32, 1) -> x shape: (30, 32, 64)

        # Result 'x' shape remains: (input_len, batch_size, d_model)
        x = self.pos_enc(x)

        # The shape remains: (input_len, batch_size, d_model)
        for layer in self.layers:
            x = layer(x) # 30, B, 64

        # output_embeddings shape: (pred_len, batch_size, d_model) # 20 (last 20 of 30), 1 , 64
        output_embeddings = x[-self.pred_len:]

        # Result 'predictions' shape: (pred_len, batch_size, input_size) (20, 1 , 64)
        predictions = self.output_proj(output_embeddings)
        return predictions

# --- Training & Evaluation ------------------------------------------------
def train(model, loader, optimizer, criterion, epochs=100):
    model.train() # 1. Set model to training mode
    for epoch in range(1, epochs+1): # 2. Loop over epochs
        epoch_loss = 0.0 # 3. Initialize loss for this epoch
        for xb, yb in loader: # 4. Loop over batches in the DataLoader
            # xb: Batch of input sequences, shape (batch_size, input_len, num_features)
            # yb: Batch of target sequences, shape (batch_size, pred_len, num_features)
						
			#The Transformer expects sequence length first.
            x = xb.permute(1, 0, 2) # 5. Reshape input for Transformer: (input_len, batch_size, num_features)
            y = yb.permute(1, 0, 2) # 6. Reshape target for comparison: (pred_len, batch_size, num_features)

            optimizer.zero_grad() # 7. Clear previous gradients

            preds = model(x) # 8. Forward pass: Get model predictions
            # preds shape: (pred_len, batch_size, num_features)

            loss = criterion(preds, y) # 9. Calculate loss (error)

            loss.backward() # 10. Backward pass: Calculate gradients

            optimizer.step() # 11. Update model weights

            epoch_loss += loss.item() # 12. Accumulate batch loss

        logger.info(f"Epoch {epoch}: avg loss {epoch_loss/len(loader):.6f}")

def evaluate(model, loader, criterion):
    model.eval()
    preds_all, actual_all = [], []
    with torch.no_grad():
        for xb, yb in loader:
            x = xb.permute(1, 0, 2)
            y = yb.permute(1, 0, 2)
            preds = model(x).cpu().numpy()
            preds_all.append(preds)
            actual_all.append(y.cpu().numpy())
    preds_all = np.concatenate(preds_all, axis=1).reshape(-1, preds_all[0].shape[2])
    actual_all = np.concatenate(actual_all, axis=1).reshape(-1, actual_all[0].shape[2])

    return preds_all, actual_all

# ------------------------ Main Function -------------------------------
def main():

    global device, logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
    logger = logging.getLogger(__name__)

    import torch_directml
    # Picking up AMD card via the DirectML backend:
    device = torch_directml.device()
    logger.info(f"Using device: {device}")

    csv_path = 'data/train.csv'
    df = pd.read_csv(csv_path)

    # Pre process the data   
    # ------------------------ #
    if '5 Minutes' in df.columns:
        df.rename(columns={'5 Minutes': 'DateTime'}, inplace=True)
    df['DateTime'] = pd.to_datetime(df['DateTime'], dayfirst=True, infer_datetime_format=True)
    df.set_index('DateTime', inplace=True)
    df = df.drop(['datetime', 'Unnamed: 0', 'day_of_week', '# Lane Points', '% Observed'], axis=1, errors='ignore')
    logger.info(f"Loaded and cleaned data: {df.shape}")
    # ------------------------- #
    # Pre processed data in df where you will have only 'DateTime' as index and all other features

    plt.figure(figsize=(12,6))
    plt.plot(df.index, df.iloc[:,0])
    plt.title(df.columns[0])
    plt.xlabel('DateTime')
    plt.ylabel(df.columns[0])
    plt.tight_layout()
    plt.show(block = True)

    # Check later on which column we would like to resample it.
    # Data preprocessing and this viualization can be done through config file.

    daily_mean = df.resample('D').mean()
    daily_mean = daily_mean.dropna()
    plt.figure(figsize=(12,6))
    for col in daily_mean.columns:
        plt.plot(daily_mean.index, daily_mean[col], label=col)
    plt.title('Daily Mean of ' + ', '.join(daily_mean.columns))
    plt.xlabel('Date')
    plt.ylabel('Mean Value')
    plt.legend()
    plt.tight_layout()
    plt.show(block = True)

    # What would be the input_len, pred_len and train_ratio, batch_size (depends on the GPU resource), d_model should be in config file.
    # Also num_heads, num_layers ?
    # Rule of thumb: d_model should be divisible by num_heads.
    # input_len = 2 × pred_len

    # Scale the data
    scaler = StandardScaler()  # Added scaler instantiation
    data_np = scaler.fit_transform(df.values.astype(np.float32))  # Scale features

    input_len, pred_len, train_ratio = 60, 15, 0.8
    total_len = len(data_np)
    split_idx = int(total_len * train_ratio)


    train_series = data_np[:split_idx]
    test_series = data_np[split_idx - input_len:]
    logger.info(f"Shape of np array of train: {train_series.shape}")
    logger.info(f"Shape of np array of test: {test_series.shape}")

    train_ds = TimeSeriesDataset(train_series, input_len, pred_len)
    test_ds = TimeSeriesDataset(test_series, input_len, pred_len)
    
    #logger.info(f"Shape of time series dataset of train set: {train_ds.shape}")
    #logger.info(f"Shape of time series dataset of train set: {test_ds.shape}")
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)
    
    #Look how data loader converted the data series
    #should be (32, 30, 1) (batch_size, input_len, feature)

    input_size = df.shape[1]
    model = TransformerModel(input_size, d_model=128, num_heads=8, num_layers=2, pred_len=pred_len).to(device)
    
    #This function measures the average squared difference between the model's predictions and the actual target values. 
    #The goal of training is to minimize this error.
    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 1.
    train(model, train_loader, optimizer, criterion)

    # 2. 
    # preds_all, actual_all = evaluate(model, test_loader, criterion)

   
    window = test_series[:input_len]
    win_t = torch.tensor(window, dtype=torch.float32, device=device).unsqueeze(1)
    model.eval()
    with torch.no_grad():
        preds_future = model(win_t).squeeze(1).cpu().numpy() # shape : (30,1,1)

    # Inverse scale the predictions
    preds_future = scaler.inverse_transform(preds_future)  # Added inverse transform

    actual_future = df.iloc[split_idx + input_len : split_idx + input_len + pred_len].values
    future_dates = df.index[split_idx + input_len : split_idx + input_len + pred_len]

    plt.figure(figsize=(12, 2 * input_size))
    for i, col in enumerate(df.columns):
        ax = plt.subplot(input_size, 1, i + 1)
        ax.plot(future_dates, actual_future[:, i], label='Actual Future')
        ax.plot(future_dates, preds_future[:, i], label='Forecasted Future')
        ax.set_title(f'Future Forecast (next {pred_len} steps): {col}')
        if i == input_size - 1:
            ax.set_xlabel('Date')
        ax.legend(loc='upper left')
    plt.tight_layout()
    plt.show(block=True)

    # 7. Compute and print metrics for future forecast window
    mse_f = mean_squared_error(actual_future, preds_future)
    mae_f = mean_absolute_error(actual_future, preds_future)
    mape_f = mean_absolute_percentage_error(actual_future, preds_future)
    r2_f = r2_score(actual_future, preds_future)
    print(f"\nFuture Forecast Metrics (Final Window):")
    print(f"MSE: {mse_f:.4f}, MAE: {mae_f:.4f}, MAPE: {mape_f:.2%}, R2: {r2_f:.4f}")



if __name__ == '__main__':
    main()