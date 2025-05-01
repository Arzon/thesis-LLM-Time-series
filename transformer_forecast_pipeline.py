# -*- coding: utf-8 -*-
"""OurCodewithOurData: Transformer pipeline with preprocessing, plots, and accuracy metrics"""

import math
import logging
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

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
        x = self.series[idx : idx + self.input_len]
        y = self.series[idx + self.input_len : idx + self.input_len + self.pred_len]
        return (
            torch.tensor(x, dtype=torch.float32, device=device),
            torch.tensor(y, dtype=torch.float32, device=device)
        )

# --- Transformer Components ------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=40000):
        super().__init__()
        pe = torch.zeros(max_len, d_model, device=device)
        pos = torch.arange(0, max_len, device=device).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
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
        Qh = Q.view(-1, B, self.num_heads, self.d_k).transpose(1,2)
        Kh = K.view(-1, B, self.num_heads, self.d_k).transpose(1,2)
        Vh = V.view(-1, B, self.num_heads, self.d_k).transpose(1,2)
        scores = torch.matmul(Qh, Kh.transpose(-2,-1)) / self.scale
        attn = torch.softmax(scores, dim=-1)
        ctx  = torch.matmul(attn, Vh)
        ctx2 = ctx.transpose(1,2).contiguous().view(-1, B, self.num_heads*self.d_k)
        return self.Wo(ctx2)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedfwd=4096, dropout=0.1):
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
        a  = self.attn(x, x, x)
        x2 = self.norm1(x + self.drop(a))
        f  = self.ff(x2)
        return self.norm2(x2 + self.drop(f))

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, num_heads, num_layers, pred_len, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model).to(device)
        self.pos_enc    = PositionalEncoding(d_model)
        self.layers     = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.pred_len   = pred_len
        self.output_proj= nn.Linear(d_model, input_size).to(device)

    def forward(self, src):
        x = self.input_proj(src)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x)
        return self.output_proj(x[-self.pred_len:])

# --- Training & Evaluation ------------------------------------------------
def train(model, loader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(1, epochs+1):
        epoch_loss = 0.0
        for xb, yb in loader:
            x = xb.permute(1, 0, 2)
            y = yb.permute(1, 0, 2)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
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
    # mse = mean_squared_error(actual_all, preds_all)
    # mae = mean_absolute_error(actual_all, preds_all)
    # mape = mean_absolute_percentage_error(actual_all, preds_all)
    # r2 = r2_score(actual_all, preds_all)
    # logger.info(f"Eval MSE: {mse:.6f}, MAE: {mae:.6f}, MAPE: {mape:.6f}, R2: {r2:.6f}")
    # print(f"Test Metrics -> MSE: {mse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2%}, R2: {r2:.4f}")
    return preds_all, actual_all

# ------------------------ Main Function -------------------------------
def main():
    """
    Pipeline steps:
    1. Load and clean the CSV time series data.
    2. Visualize original and resampled daily mean plots.
    3. Split the dataset into training and testing.
    4. Create Transformer model with attention blocks.
    5. Train the model on training data.
    6. Evaluate the model on test data.
    7. Forecast the final window and plot predicted vs actual.
    8. Print final accuracy metrics.
    """

    global device, logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
    logger = logging.getLogger(__name__)
    fh = logging.FileHandler('transformer_tutorial.log')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    csv_path = '/data/train_preprocessed_flow.csv'
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
    plt.show()

    # Check later on which column we would like to resample it.
    # Data preprocessing and this viualization can be done through config file.

    daily_mean = df.resample('D').mean()
    plt.figure(figsize=(12,6))
    for col in daily_mean.columns:
        plt.plot(daily_mean.index, daily_mean[col], label=col)
    plt.title('Daily Mean of ' + ', '.join(daily_mean.columns))
    plt.xlabel('Date')
    plt.ylabel('Mean Value')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # What would be the input_len, pred_len and train_ratio, batch_size (depends on the GPU resource), d_model should be in config file.
    # Also num_heads, num_layers ?
    # Rule of thumb: d_model should be divisible by num_heads.
    # input_len = 2 × pred_len

    input_len, pred_len, train_ratio = 30, 20, 0.8
    total_len = len(df)
    split_idx = int(total_len * train_ratio)
    data_np = df.values.astype(np.float32)

    train_series = data_np[:split_idx]
    test_series = data_np[split_idx - input_len:]

    train_ds = TimeSeriesDataset(train_series, input_len, pred_len)
    test_ds = TimeSeriesDataset(test_series, input_len, pred_len)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)

    input_size = df.shape[1]
    model = TransformerModel(input_size, d_model=64, num_heads=8, num_layers=2, pred_len=pred_len).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 1.
    train(model, train_loader, optimizer, criterion)

    # 2. 
    preds_all, actual_all = evaluate(model, test_loader, criterion)

    #
    start_idx = split_idx + input_len
    end_idx = start_idx + len(preds_all)
    dates_all = df.index[start_idx:end_idx]

    plt.figure(figsize=(12, 2 * input_size))
    for i, col in enumerate(df.columns):
        ax = plt.subplot(input_size, 1, i + 1)
        ax.plot(dates_all, actual_all[:, i], label='Actual')
        ax.plot(dates_all, preds_all[:, i], label='Predicted')
        ax.set_title(f'Full Test Set Prediction: {col}')
        if i == input_size - 1:
            ax.set_xlabel('Date')
        ax.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    
    mse_all = mean_squared_error(actual_all, preds_all)
    mae_all = mean_absolute_error(actual_all, preds_all)
    mape_all = mean_absolute_percentage_error(actual_all, preds_all)
    r2_all = r2_score(actual_all, preds_all)
    print(f"\nFull Test Set Metrics:")
    print(f"MSE: {mse_all:.4f}, MAE: {mae_all:.4f}, MAPE: {mape_all:.2%}, R2: {r2_all:.4f}")

   
    window = test_series[-input_len:]
    win_t = torch.tensor(window, dtype=torch.float32, device=device).unsqueeze(1)
    model.eval()
    with torch.no_grad():
        preds_future = model(win_t).squeeze(1).cpu().numpy()

    
    actual_future = df.iloc[split_idx:split_idx+pred_len].values
    future_dates = df.index[split_idx:split_idx+pred_len]

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
    plt.show()

    # 7. Compute and print metrics for future forecast window
    mse_f = mean_squared_error(actual_future, preds_future)
    mae_f = mean_absolute_error(actual_future, preds_future)
    mape_f = mean_absolute_percentage_error(actual_future, preds_future)
    r2_f = r2_score(actual_future, preds_future)
    print(f"\nFuture Forecast Metrics (Final Window):")
    print(f"MSE: {mse_f:.4f}, MAE: {mae_f:.4f}, MAPE: {mape_f:.2%}, R2: {r2_f:.4f}")



if __name__ == '__main__':
    main()
