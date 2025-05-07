# evaluate_flow_model.py

import sys, os, json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)

sys.path.append(os.path.abspath(r"D:\Thesis\flow-forecast"))

from flood_forecast.time_model import pytorch_model_dict

# ---------------- CONFIG & DATA PATHS ----------------
CONFIG_PATH  = "model_save/01_May_202507_55PM.json"
WEIGHTS_PATH = "model_save/01_May_202507_55PM_model.pth"
DATA_PATH    = "train_processed.csv"

# ---------------- LOAD & PATCH CONFIG ----------------
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# Extract relevant settings
model_name      = config["model_name"]
model_params    = config["model_params"]
data_params     = config["dataset_params"]
forecast_history= data_params["forecast_history"]
forecast_length = data_params["forecast_length"]
target_col      = data_params["target_col"][0]
test_start      = data_params["test_start"]

# ---------------- LOAD & SCALE DATA ----------------
df = pd.read_csv(DATA_PATH, parse_dates=["DateTime"], index_col="DateTime")
series = df[[target_col]].values.astype(float)

scaler = StandardScaler()
series_scaled = scaler.fit_transform(series)

device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_cls= pytorch_model_dict[model_name]
model    = model_cls(**model_params).to(device)

# 3) Load weights
state = torch.load(WEIGHTS_PATH, map_location=device)
model.load_state_dict(state)
model.eval()

# ---------------- PREPARE INPUT WINDOW ----------------
input_len = forecast_history
pred_len  = forecast_length

window   = series_scaled[test_start - input_len : test_start]
win_t    = torch.tensor(window, dtype=torch.float32, device=device).unsqueeze(1)
#   shape: (seq_len, batch=1, features=1)

# ---------------- RUN INFERENCE ----------------
with torch.no_grad():
    out_scaled = model(win_t)                       # (output_seq_len, 1, 1)
out = out_scaled.squeeze(1).cpu().numpy()           # (output_seq_len, 1)
preds = scaler.inverse_transform(out)               # back to original units

# ---------------- ACTUAL FUTURE SLICE ----------------
actual = series[test_start : test_start + pred_len] # (pred_len, 1)
dates  = df.index[test_start : test_start + pred_len]

# ---------------- PLOTTING ----------------
plt.figure(figsize=(12, 6))
plt.plot(dates, actual, label="Actual")
plt.plot(dates, preds,  label="Predicted")
plt.title("Flow-Forecast Model: Actual vs Predicted")
plt.xlabel("Date")
plt.ylabel(target_col)
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()

# ---------------- METRICS ----------------
mse  = mean_squared_error(actual, preds)
mae  = mean_absolute_error(actual, preds)
mape = mean_absolute_percentage_error(actual, preds)
r2   = r2_score(actual, preds)

print("\n Evaluation Metrics (Final Forecast Window):")
print(f"  MSE:  {mse:.4f}")
print(f"  MAE:  {mae:.4f}")
print(f"  MAPE: {mape:.2%}")
print(f"  R²:   {r2:.4f}")
