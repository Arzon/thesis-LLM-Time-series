import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import torch_directml

# --- Dataset ---------------------------------------------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, series, input_len, pred_len):
        self.series = series
        self.input_len = input_len
        self.pred_len = pred_len
        self.total_samples = len(series) - input_len - pred_len + 1

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        x = self.series[idx : idx + self.input_len]  # (input_len, 1)
        y = self.series[idx + self.input_len : idx + self.input_len + self.pred_len]  # (pred_len, 1)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# --- Positional Encoding ---------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# --- Encoder-Only Transformer ---------------------------------------------
class TransformerTimeSeries(nn.Module):
    def __init__(
        self,
        input_size,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        dropout,
        pred_len,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,  # Post-Norm
        )
        self.transformer_enc = TransformerEncoder(encoder_layer, num_layers)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        self.output_proj = nn.Linear(d_model, input_size)
        self.pred_len = pred_len

    def forward(self, src):
        # src shape: (batch, seq_len, input_size)
        x = self.input_proj(src)             # (batch, seq_len, d_model)
        x = self.pos_enc(x)                  # add positional encoding
        x = self.transformer_enc(x)         # (batch, seq_len, d_model)
        out = x[:, -self.pred_len :, :]      # (batch, pred_len, d_model)
        out = self.output_proj(out)          # (batch, pred_len, input_size)
        return out

# --- Training & Evaluation -----------------------------------------------
def train(model, loader, optimizer, criterion, device, epochs=50):
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)  # (batch, pred_len, 1)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")


def evaluate(model, loader, criterion, device):
    model.eval()
    preds_all, actual_all = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            preds_all.append(preds.cpu().numpy())
            actual_all.append(yb.cpu().numpy())
    preds_all = np.concatenate(preds_all, axis=0)   # (num_samples, pred_len, 1)
    actual_all = np.concatenate(actual_all, axis=0)
    mse  = mean_squared_error(actual_all.reshape(-1), preds_all.reshape(-1))
    mae  = mean_absolute_error(actual_all.reshape(-1), preds_all.reshape(-1))
    mape = mean_absolute_percentage_error(actual_all.reshape(-1), preds_all.reshape(-1))
    r2   = r2_score(actual_all.reshape(-1), preds_all.reshape(-1))
    return mse, mae, mape, r2, preds_all, actual_all

# --- Main ---------------------------------------------------------------
def main():
    # Device setup
    device = torch_directml.device()
    print("Using device:", device)

    # Load & preprocess data
    df = pd.read_csv('data/train_processed.csv', index_col='datetime', parse_dates=['datetime'])
    series = df['Lane 1 Flow (Veh/5 Minutes)'].values.astype(np.float32).reshape(-1, 1)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(series)

    # Train/test split
    input_len, pred_len, train_ratio = 60, 15, 0.8
    split = int(len(data_scaled) * train_ratio)
    train_series = data_scaled[:split]
    test_series  = data_scaled[split - input_len :]

    train_ds = TimeSeriesDataset(train_series, input_len, pred_len)
    test_ds  = TimeSeriesDataset(test_series,  input_len, pred_len)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=32)

    # Model instantiation
    model = TransformerTimeSeries(
        input_size=1,
        d_model=128,
        nhead=8,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        pred_len=pred_len,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Training
    train(model, train_loader, optimizer, criterion, device, epochs=50)

    # Evaluation
    mse, mae, mape, r2, preds_all, actual_all = evaluate(model, test_loader, criterion, device)
    print(f"Test MSE: {mse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2%}, R2: {r2:.4f}")

    # Plot one forecast window
    future_pred   = preds_all[-1].reshape(-1, 1)
    future_actual = actual_all[-1].reshape(-1, 1)
    # future_dates  = df.index[split : split + pred_len]
    future_dates  = df.index[-pred_len:] 

    plt.figure(figsize=(10,4))
    plt.plot(future_dates, scaler.inverse_transform(future_actual), label='Actual')
    plt.plot(future_dates, scaler.inverse_transform(future_pred),   label='Forecast')
    plt.title('Future Forecast vs Actual')
    plt.xlabel('DateTime')
    plt.ylabel('Lane 1 Flow (Veh/5 Minutes)')
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)

if __name__ == '__main__':
    main()
