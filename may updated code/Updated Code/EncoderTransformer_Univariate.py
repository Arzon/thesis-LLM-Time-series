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

    # def forward(self, q, k, v): 
    #     B = q.size(1)
    #     Q = self.Wq(q); K = self.Wk(k); V = self.Wv(v)
    #     Qh = Q.view(-1, B, self.num_heads, self.d_k).transpose(1,2) # 30, B, 8, 8
    #     Kh = K.view(-1, B, self.num_heads, self.d_k).transpose(1,2)
    #     Vh = V.view(-1, B, self.num_heads, self.d_k).transpose(1,2)
    #     scores = torch.matmul(Qh, Kh.transpose(-2,-1)) / self.scale # 30, 8, B, 1
    #     attn = torch.softmax(scores, dim=-1)
    #     ctx  = torch.matmul(attn, Vh) # 30, 8, B, 8
    #     ctx2 = ctx.transpose(1,2).contiguous().view(-1, B, self.num_heads*self.d_k) # 30, B, 64
    #     return self.Wo(ctx2)

    def forward(self, q, k, v):
        # q, k, v: (seq_len, batch, d_model)
        S, B, _ = q.size()
        # 1) Linear projections
        Q = self.Wq(q)   # (S, B, D)
        K = self.Wk(k)
        V = self.Wv(v)

        # 2) reshape+permute to (batch, heads, seq_len, d_k)
        Q = Q.view(S, B, self.num_heads, self.d_k).permute(1,2,0,3)
        K = K.view(S, B, self.num_heads, self.d_k).permute(1,2,0,3)
        V = V.view(S, B, self.num_heads, self.d_k).permute(1,2,0,3)

        # 3) scaled dot-product over seq_len
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, S, S)
        attn   = torch.softmax(scores, dim=-1)
        ctx    = torch.matmul(attn, V)                              # (B, H, S, d_k)

        # 4) back to (seq_len, batch, d_model)
        ctx = ctx.permute(2,0,1,3).contiguous().view(S, B, self.num_heads*self.d_k)
        return self.Wo(ctx)

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
 
    def forward(self, x): # Only post norm
        a  = self.attn(x, x, x) # (30,B,64)
        x2 = self.norm1(x + self.drop(a)) # 30, B, 64
        f  = self.ff(x2) # 30, B, 64
        return self.norm2(x2 + self.drop(f)) # 30, B, 64

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, num_heads, num_layers, pred_len, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model).to(device)

        self.pos_enc = PositionalEncoding(d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.pred_len = pred_len # Store prediction length

        self.output_proj = nn.Linear(d_model, input_size).to(device)

    def forward(self, src):
        # src shape: (input_len, batch_size, input_size)

        x = self.input_proj(src)

        x = self.pos_enc(x)

        for layer in self.layers:
            x = layer(x) # 30, B, 64

        output_embeddings = x[-self.pred_len:]

        predictions = self.output_proj(output_embeddings)
        return predictions

def train(model, loader, optimizer, criterion, epochs=50):
    model.train() # 1. Set model to training mode
    for epoch in range(1, epochs+1): # 2. Loop over epochs
        epoch_loss = 0.0 # 3. Initialize loss for this epoch
        for xb, yb in loader: # 4. Loop over batches in the DataLoader

            x = xb.permute(1, 0, 2) 
            y = yb.permute(1, 0, 2) 

            optimizer.zero_grad() 

            preds = model(x)


            loss = criterion(preds, y)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        logger.info(f"Epoch {epoch}: avg loss {epoch_loss/len(loader):.6f}")

# def evaluate(model, loader, criterion):
#     model.eval()
#     preds_all, actual_all = [], []
#     with torch.no_grad():
#         for xb, yb in loader:
#             x = xb.permute(1, 0, 2)
#             y = yb.permute(1, 0, 2)
#             preds = model(x).cpu().numpy()
#             preds_all.append(preds)
#             actual_all.append(y.cpu().numpy())
#     preds_all = np.concatenate(preds_all, axis=1).reshape(-1, preds_all[0].shape[2])
#     actual_all = np.concatenate(actual_all, axis=1).reshape(-1, actual_all[0].shape[2])

#     return preds_all, actual_all

def evaluate(model, loader, criterion):
    model.eval()
    preds_list, actual_list = [], []
    with torch.no_grad():
        for xb, yb in loader:
            # 1) Move to device
            xb, yb = xb.to(device), yb.to(device)

            x = xb.permute(1, 0, 2)  # (input_len, batch, 1)
            y = yb.permute(1, 0, 2)  # (pred_len,  batch, 1)

            # 3) Forward
            preds = model(x)        # -> (pred_len, batch, 1)

            preds = preds.permute(1, 0, 2)  # (batch, pred_len, 1)
            actual = y.permute(1, 0, 2)     # (batch, pred_len, 1)

            preds_list.append(preds.cpu().numpy())
            actual_list.append(actual.cpu().numpy())

    preds_all  = np.concatenate(preds_list,  axis=0)  # (n_windows, pred_len, 1)
    actual_all = np.concatenate(actual_list, axis=0)  # (n_windows, pred_len, 1)

    flat_pred   = preds_all .reshape(-1)
    flat_actual = actual_all.reshape(-1)
    mse  = mean_squared_error(flat_actual, flat_pred)
    mae  = mean_absolute_error(flat_actual, flat_pred)
    mape = mean_absolute_percentage_error(flat_actual, flat_pred)
    r2   = r2_score(flat_actual, flat_pred)

    preds_all  = preds_all .squeeze(-1)
    actual_all = actual_all.squeeze(-1)

    return mse, mae, mape, r2, preds_all, actual_all



def main():
    print(">>> main() start")
    global device, logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
    logger = logging.getLogger(__name__)

    import torch_directml
    # Picking up AMD card via the DirectML backend:
    device = torch_directml.device()
    logger.info(f"Using device: {device}")

    CSV_PATH   = "data/Encoder_univariate.csv"
    TIME_COL   = "datetime"                             
    TARGET_COLS = ["Lane 1 Flow (Veh/5 Minutes)"]

    df = pd.read_csv(CSV_PATH)
    if  TIME_COL not in df.columns:
        if TIME_COL != "datetime":
            df.rename(columns={TIME_COL: "datetime"}, inplace=True)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
    

    if isinstance(TARGET_COLS, str):
        TARGET_COLS = [TARGET_COLS]
    df = df[TARGET_COLS]
    logger.info(f"Kept features: {TARGET_COLS}, new df.shape = {df.shape}")


    plt.figure(figsize=(12,6))
    plt.plot(df.index, df.iloc[:,0])
    plt.title(df.columns[0])
    plt.xlabel('DateTime')
    plt.ylabel(df.columns[0])
    plt.tight_layout()
    plt.show(block = True)

    # Check later on which column we would like to resample it.
    # Data preprocessing and this viualization can be done through config file.

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
    
    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    train(model, train_loader, optimizer, criterion)

    # preds_all, actual_all = evaluate(model, test_loader, criterion)

    # preds_future = scaler.inverse_transform(preds_future)  # Added inverse transform

    # actual_future = df.iloc[split_idx + input_len : split_idx + input_len + pred_len].values
    # future_dates = df.index[split_idx + input_len : split_idx + input_len + pred_len]

    # plt.figure(figsize=(12, 2 * input_size))
    # for i, col in enumerate(df.columns):
    #     ax = plt.subplot(input_size, 1, i + 1)
    #     ax.plot(future_dates, actual_future[:, i], label='Actual Future')
    #     ax.plot(future_dates, preds_future[:, i], label='Forecasted Future')
    #     ax.set_title(f'Future Forecast (next {pred_len} steps): {col}')
    #     if i == input_size - 1:
    #         ax.set_xlabel('Date')
    #     ax.legend(loc='upper left')
    # plt.tight_layout()
    # plt.show(block=True)

    # mse_f = mean_squared_error(actual_future, preds_future)
    # mae_f = mean_absolute_error(actual_future, preds_future)
    # mape_f = mean_absolute_percentage_error(actual_future, preds_future)
    # r2_f = r2_score(actual_future, preds_future)
    # print(f"\nFuture Forecast Metrics (Final Window):")
    # print(f"MSE: {mse_f:.4f}, MAE: {mae_f:.4f}, MAPE: {mape_f:.2%}, R2: {r2_f:.4f}")
        # Evaluation on test set (no shuffle, first test window)
    
    mse, mae, mape, r2, preds_all, actual_all = evaluate(model, test_loader, criterion)
    print(f"Test MSE: {mse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2%}, R2: {r2:.4f}")

    ix = -1
    future_pred   = preds_all[ix].reshape(-1,1)   # => (pred_len,1)
    future_actual = actual_all[ix].reshape(-1,1)  # => (pred_len,1)

    future_dates  = df.index[-pred_len:] 

    plt.figure(figsize=(12,4))
    plt.plot(future_dates,
            scaler.inverse_transform(future_actual),
            label='Actual Future')
    plt.plot(future_dates,
            scaler.inverse_transform(future_pred),
            label='Forecasted Future')
    plt.title(f'Future Forecast (next {pred_len} steps): Lane 1 Flow (Veh/5 Minutes)')
    plt.xlabel('DateTime')
    plt.ylabel('Flow (Veh/5 Minutes)')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show(block=True)



if __name__ == '__main__':
    main()