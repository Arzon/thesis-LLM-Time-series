import os
import wandb

wandb.login(key="249dabcebd94431945bc7c51921b5c64075fbbd1")

import pandas as pd
import datetime as datetime

df = pd.read_csv("data/train.csv")
# if '5 Minutes' in df.columns:
#     df.rename(columns={'5 Minutes': 'DateTime'}, inplace=True)
df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True, infer_datetime_format=True)
df.set_index('datetime', inplace=True)
df = df.drop(['5 Minutes', 'Unnamed: 0', 'day_of_week', '# Lane Points', '% Observed'], axis=1, errors='ignore')

df.to_csv("data/train_processed.csv", index=True)

def make_config_file(file_path, train_end, valid_end):
  run = wandb.init(project="library_demos")
  wandb_config = wandb.config
  C = wandb.config
  print(wandb_config["out_seq_length"])
  config_default={
    "model_name": "SimpleTransformer",
    "model_type": "PyTorch",
    "model_params": {
    "d_model": 64,
    "n_heads": 4,
    "dropout": 0.1,
    "forward_dim": 256,
    "seq_length": wandb_config["forecast_history"],
    "output_seq_len": wandb_config["out_seq_length"],
    "number_time_series": 1
    },
    "dataset_params":
    {  "class": "default",
       "training_path": file_path,
       "validation_path": file_path,
       "test_path": file_path,
       "batch_size":wandb_config["batch_size"],
       "forecast_history":wandb_config["forecast_history"],
       "forecast_length":wandb_config["out_seq_length"],
       "train_end": train_end,
       "valid_start":int(train_end+1),
       "valid_end": int(valid_end),
       "test_start":int(valid_end) + 1,
       "target_col": ["Lane 1 Flow (Veh/5 Minutes)"],
       "relevant_cols": ["Lane 1 Flow (Veh/5 Minutes)"],
       "scaler": "StandardScaler",
       "interpolate": False
    },
    "training_params":
    {
       "criterion":"MSE",
       "optimizer": "Adam",
       "optim_params":
       {

       },
       "lr": wandb_config["lr"],
       "epochs": 10,
       "batch_size":wandb_config["batch_size"]
    },
    "GCS": False,
    "sweep":True,
    "wandb":False,
    "forward_params":{},
   "metrics":["MSE"],
   "inference_params":
   {
          "datetime_start":"2016-02-24",
          "hours_to_forecast":150,
          "test_csv_path":file_path,
          "decoder_params":{
              "decoder_function": "greedy_decode",
            "unsqueeze_dim": 1
          },
          "dataset_params":{
             "file_path": file_path,
             "forecast_history":wandb_config["forecast_history"],
             "forecast_length":wandb_config["out_seq_length"],
             "relevant_cols": ["Lane 1 Flow (Veh/5 Minutes)"],
             "target_col": ["Lane 1 Flow (Veh/5 Minutes)"],
             "scaling": "StandardScaler",
             "interpolate_param": False
          }
      }
  }
  wandb.config.update(config_default)
  return config_default


sweep_config = {
  "name": "transformer-sweep",
  "method": "random",
  "parameters": {
        "batch_size": {
            "values": [32]
        },
        "lr":{
            "values":[0.001]
        },
        "forecast_history":{
            "values":[30]
        },
        "out_seq_length":{
            "values":[20]
        }
    }
}

import os, sys

# Insert the folder containing this script (and trainer.py) at the front of the module search path
here = os.path.abspath(os.path.dirname("flow-forecast/flood_forecast/trainer.py"))
sys.path.insert(0, here)

# Now this will load your local trainer.py
from trainer import train_function

# from flood_forecast.trainer import train_function
import wandb
def main():
    sweep_id = wandb.sweep(sweep_config)
    os.environ["SWEEP_ID"] = sweep_id
    wandb.agent(sweep_id, lambda: train_function("PyTorch", make_config_file("data/train_processed.csv", 4500, 6000)))

if __name__ == "__main__":
    main()