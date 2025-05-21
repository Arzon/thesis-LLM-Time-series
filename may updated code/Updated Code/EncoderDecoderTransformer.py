import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)

#  User‐configuration
CSV_PATH    = "data/EncoderDecoder.csv"
TIME_COL    = "datetime"                             
TARGET_COLS = ["Lane 1 Flow (Veh/5 Minutes)"]
INPUT_LEN   = 100
PRED_LEN    = 20
BATCH_SIZE  = 32
TRAIN_RATIO = 0.8 

class TimeSeriesDataset(Dataset):
    def __init__(self, series: np.ndarray, input_len: int, pred_len: int):
        # series: shape (T, F)
        self.series = series
        self.input_len = input_len
        self.pred_len = pred_len
        self.total_samples = len(series) - input_len - pred_len + 1
        logger.info(f"[Dataset] series={series.shape}, input={input_len}, pred={pred_len}, samples={self.total_samples}")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        x = self.series[idx : idx + self.input_len]                         # (input_len, F)
        y = self.series[idx + self.input_len : idx + self.input_len + self.pred_len]  # (pred_len, F)
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (seq_len, batch, d_model)
        return x + self.pe[: x.size(0)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, q, k, v, mask=None):
        # q: (len_q, batch, d_model)
        # k,v: (len_k, batch, d_model)
        len_q, B, _ = q.size()
        len_k, _, _ = k.size()

        Q = self.Wq(q)  # (len_q, B, D)
        K = self.Wk(k)  # (len_k, B, D)
        V = self.Wv(v)  # (len_k, B, D)

        # reshape & permute to (B, H, seq_len, d_k)
        Q = Q.view(len_q, B, self.num_heads, self.d_k).permute(1, 2, 0, 3)  # (B, H, len_q, d_k) (32, 8, 100/20, 32)
        K = K.view(len_k, B, self.num_heads, self.d_k).permute(1, 2, 0, 3)  # (B, H, len_k, d_k)
        V = V.view(len_k, B, self.num_heads, self.d_k).permute(1, 2, 0, 3)  # (B, H, len_k, d_k)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, len_q, len_k) (32, 8, 100/20, 100)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        ctx = torch.matmul(attn, V)  # (B, H, len_q, d_k) (32, 8, 100/20, 32)

        ctx = ctx.permute(2,0,1,3).contiguous()  # (len_q, B, H, d_k) (100, 32, 8, 32)
        ctx = ctx.view(len_q, B, self.num_heads * self.d_k)  # (len_q, B, D)
        return self.Wo(ctx)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attended = self.norm1(x)
        attended = self.attn(attended, attended, attended) # (100, 32, 256)
        x = x + self.dropout(attended)
        
        ff_out = self.norm2(x)
        ff_out = self.ff(ff_out) # (100, 32, 256)
        x = x + self.dropout(ff_out)
        
        return x

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        q = k = v = self.norm1(tgt) # (20, 32, 256)
        tgt2 = self.self_attn(q, k, v, tgt_mask) 
        tgt = tgt + self.dropout(tgt2) # (20, 32, 256)
        
        q = self.norm2(tgt)
        tgt2 = self.cross_attn(q, memory, memory, memory_mask) # q (20, 32, 256), memory (100, 32, 256)
        tgt = tgt + self.dropout(tgt2)
        
        # Feed-forward
        tgt2 = self.ff(self.norm3(tgt))
        tgt = tgt + self.dropout(tgt2)
        
        return tgt # (20, 32, 256)

class FullEncodeDecoderTransformar(nn.Module):
    def __init__(
        self,
        input_size,
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=1024,
        dropout=0.2,
        pred_len=20
    ):
        super().__init__()
        # Simple input projection
        self.input_proj = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model)
        
        # Encoder stack
        self.enc_layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder stack
        self.dec_layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, input_size) 
        
        self.pred_len = pred_len
        self.d_model = d_model
        self.input_size = input_size
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def create_causal_mask(self, seq_len):
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return ~mask  # Flip to 0 for masked positions

    def forward(self, src, src_mask=None):
        B, S, F = src.size()
        device = src.device
        
        x = self.input_proj(src).permute(1, 0, 2)  # (S, B, d_model) (100, 32, 256)
        
        # Add positional encoding
        x = self.pos_enc(x) # (100, 32, 256)
        
        for enc in self.enc_layers:
            x = enc(x) # (100, 32, 256)
        
        memory = x  # (S, B, d_model) (S=100, B=32, d_model=256)
        
        last_val = x[-1:].repeat(self.pred_len, 1, 1)  # (pred_len, B, d_model) (20 , 32, 256)
        tgt = self.pos_enc(last_val)  # (20, 32, 256)
        
        for dec in self.dec_layers:
            tgt_mask = self.create_causal_mask(tgt.size(0)).to(device) # (20, 20)
            tgt = dec(tgt, memory, tgt_mask) # tgt (20, 32, 256) , memory (100, 32, 256), tgt_mask (20, 20)
        
        # Project back to input dimension
        out = self.output_proj(tgt).permute(1, 0, 2)  # (B, pred_len, F)
        
        return out


class TemporalLoss(nn.Module):
    """Loss that emphasizes direction and pattern prediction"""
    def __init__(self, pred_len):
        super().__init__()
        self.pred_len = pred_len
        self.mse = nn.MSELoss(reduction='none')
        
        # Increasing weights for future time steps
        self.register_buffer(
            "time_weights", 
            torch.linspace(1.0, 2.0, pred_len).view(1, pred_len, 1)
        )
        
    def forward(self, preds, targets):
        """
        preds, targets: (B, pred_len, F)
        """
        # Calculate standard MSE
        mse_loss = self.mse(preds, targets)  # (B, pred_len, F)
        
        # Weight the loss to emphasize later time steps
        weighted_mse = mse_loss * self.time_weights
        
        # Add direction penalty
        pred_diff = preds[:, 1:] - preds[:, :-1]  # (B, pred_len-1, F)
        target_diff = targets[:, 1:] - targets[:, :-1]
        
        # Penalize wrong direction predictions
        direction_penalty = (torch.sign(pred_diff) != torch.sign(target_diff)).float() * 0.5
        direction_penalty = F.pad(direction_penalty, (0, 0, 0, 1))  # Pad to match original shape
        
        # Combine losses
        final_loss = weighted_mse + direction_penalty * weighted_mse
        
        return final_loss.mean()


def train_model(model, train_loader, test_loader, num_epochs=50, patience=7):
    logger.info("Starting model training...")
    
    # Loss functions
    mse_criterion = nn.MSELoss()
    temporal_criterion = TemporalLoss(model.pred_len).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1e-3,
        weight_decay=1e-4
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(1, num_epochs+1):
        # Training
        model.train()
        train_loss = 0.0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            preds = model(xb)
            
            # Calculate loss
            alpha = max(0.3, 0.7 - 0.01 * epoch)
            loss = alpha * mse_criterion(preds, yb) + (1-alpha) * temporal_criterion(preds, yb)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                val_loss += mse_criterion(preds, yb).item()
        
        val_loss /= len(test_loader)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        logger.info(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            logger.info(f"New best model saved (val_loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter} epochs")
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info("Loaded best model from training")
    
    # Plot loss curves
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return model, train_losses, val_losses

def evaluate(model, loader, criterion):
    model.eval()
    all_p, all_y = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            p = model(xb)
            all_p.append(p.cpu().numpy())
            all_y.append(yb.cpu().numpy())
    
    preds = np.concatenate(all_p, axis=0)
    actual = np.concatenate(all_y, axis=0)
    
    # Calculate metrics
    flat_p = preds.reshape(-1)
    flat_y = actual.reshape(-1)
    
    return (
        mean_squared_error(flat_y, flat_p),
        mean_absolute_error(flat_y, flat_p),
        mean_absolute_percentage_error(flat_y, flat_p),
        r2_score(flat_y, flat_p),
        preds,
        actual
    )

def forecast_autoregressive(model, history_np, input_len, pred_len, horizon, device):
    model.eval()
    preds = []
    window = history_np[-input_len:].copy() 

    total_predicted = 0
    with torch.no_grad():
        while total_predicted < horizon:
            # Shape input for model: (batch=1, input_len, F)
            x = torch.tensor(window, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Get prediction
            y_hat = model(x)  # (batch=1, pred_len, F)
            y_hat = y_hat.cpu().numpy()[0]  # (pred_len, F)
            
            # Take as many steps as needed
            n = min(pred_len, horizon - total_predicted)
            preds.append(y_hat[:n])
            
            # Update window for next iteration
            window = np.vstack([window[n:], y_hat[:n]])
            total_predicted += n

    return np.vstack(preds)  # (horizon, F)


def main():
    global device, logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("FullEncodeDecoderTransformar")

    # Device setup
    try:
        import torch_directml
        device = torch_directml.device()
    except ImportError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    df = pd.read_csv(CSV_PATH)
    if TIME_COL != "DateTime":
        df.rename(columns={TIME_COL: "DateTime"}, inplace=True)
    print("Columns after rename:", df.columns)
    df["DateTime"] = pd.to_datetime(df["DateTime"], dayfirst=True)
    df.set_index("DateTime", inplace=True)
    df = df[TARGET_COLS]
    logger.info(f"Loaded data with columns {TARGET_COLS}; df.shape={df.shape}")

    # Add time features
    df_enhanced = df.copy()
    df_enhanced['hour'] = df.index.hour / 24.0
    df_enhanced['day_of_week'] = df.index.dayofweek / 7.0
    df_enhanced['day_of_month'] = df.index.day / 31.0
    df_enhanced['month'] = df.index.month / 12.0
    
    # Plot raw data
    plt.figure(figsize=(12, 5))
    for c in TARGET_COLS:
        plt.plot(df.index, df[c], label=c)
    plt.title("Raw Traffic Flow Data")
    plt.xlabel("DateTime")
    plt.ylabel("Traffic Flow")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Split data
    split_idx = int(len(df) * TRAIN_RATIO)
    train_df = df_enhanced.iloc[:split_idx]
    test_df = df_enhanced.iloc[split_idx - INPUT_LEN:]
    
    # Scale data
    scaler = StandardScaler()
    scaler.fit(train_df.values)

    logger.info(scaler.scale_[0])
    
    train_np = scaler.transform(train_df.values)
    test_np = scaler.transform(test_df.values)
    
    # Create datasets
    train_ds = TimeSeriesDataset(train_np, INPUT_LEN, PRED_LEN)
    test_ds = TimeSeriesDataset(test_np, INPUT_LEN, PRED_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    # Create model
    input_features = train_np.shape[1]  # Number of features
    logger.info(f"Input features: {input_features}")
    
    model = FullEncodeDecoderTransformar(
        input_size=input_features,    
        d_model=256,          
        nhead=8,                     
        num_encoder_layers=3,         
        num_decoder_layers=3,        
        dim_feedforward=1024,         
        dropout=0.2,      
        pred_len=PRED_LEN
    ).to(device)
    
    # Log model size
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {total_params:,} parameters")
    
    # Train model
    model, train_losses, val_losses = train_model(
        model, train_loader, test_loader, num_epochs=50
    )
    
    # Evaluate
    mse, mae, mape, r2, preds_all, actual_all = evaluate(model, test_loader, nn.MSELoss())
    
    # Print metrics
    print(f"\nTEST METRICS:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2%}")
    print(f"R²: {r2:.4f}")
    
    # Plot sample prediction
    ix = 0  # First batch sample
    y_true = actual_all[ix]
    y_pred = preds_all[ix]
    
    # Inverse transform
    y_true_i = scaler.inverse_transform(y_true)
    y_pred_i = scaler.inverse_transform(y_pred)
    
    # Plot dates
    validation_dates = df.index[split_idx:split_idx + PRED_LEN]
    
    plt.figure(figsize=(10, 6))
    for f in range(len(TARGET_COLS)):
        plt.plot(validation_dates, y_true_i[:, f], 'b-', label=f"Actual {TARGET_COLS[f]}")
        plt.plot(validation_dates, y_pred_i[:, f], 'r--', label=f"Forecast {TARGET_COLS[f]}")
    
    plt.title("Forecast vs Actual (Validation Sample)")
    plt.xlabel("DateTime")
    plt.ylabel("Traffic Flow")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Generate forecasts
    logger.info("Generating long-term forecasts...")
    horizon = 150
    
    # Context window
    context_window = train_np[-INPUT_LEN:].copy()
    
    logger.info("  - Basic autoregressive forecast...")
    preds_auto = forecast_autoregressive(
        model, context_window, INPUT_LEN, PRED_LEN, horizon, device
    )
    
    # Inverse transform
    preds_auto_i = scaler.inverse_transform(preds_auto)
    
    actual_forecast_period = df.iloc[split_idx:split_idx + horizon].values
    forecast_dates = df.index[split_idx:split_idx + horizon]
    
    plt.figure(figsize=(14, 8))
    plt.plot(forecast_dates, actual_forecast_period, 'k-', label="Actual Data", linewidth=2)
    plt.plot(forecast_dates, preds_auto_i[:, 0], 'b--', label="Basic Autoregressive", alpha=0.7)
    
    plt.title(f"Long-term Traffic Flow Forecast ({horizon} steps)")
    plt.xlabel("DateTime")
    plt.ylabel("Traffic Flow")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Calculate metrics
    auto_mse = mean_squared_error(actual_forecast_period, preds_auto_i)
    
    print("\nLONG-TERM FORECAST METRICS:")
    print(f"Basic Autoregressive → MSE: {auto_mse:.4f}")
    
    logger.info("Forecasting complete.")

if __name__ == "__main__":
    main()
