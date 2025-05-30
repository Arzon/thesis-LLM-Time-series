import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
from transformer_model_definition import SimpleTransformerForecast, TimeSeriesTransformer

import torch_directml
device = torch_directml.device()
print(f"DirectML available and Using device: {device}")


# 1. Load and preprocess data
DATA_PATH = 'data/weather_data_min.csv'
df = pd.read_csv(DATA_PATH, parse_dates=['DateTime'], dayfirst=True)
df.set_index('DateTime', inplace=True)

df = df.rename(columns={
    'T (degC)': 'temperature',
    'rh (%)': 'humidity',
    'rho (g/m**3)': 'air_density'
})


features = ['temperature', 'humidity', 'air_density']
orig_data = df[features].values.astype(np.float32)

# Save the last 10% for anomaly detection
split_point = int(0.9 * len(orig_data))
train_data = orig_data[:split_point]
test_data = orig_data[split_point:]

# Save test data and dates for anomaly detection
test_dates = df.index[split_point:]
np.save('transformer_test_data_for_anomaly.npy', test_data)
test_dates.to_series().to_csv('transformer_test_dates.csv')

print(f"Training data shape: {train_data.shape}")
print(f"Test data for anomaly detection shape: {test_data.shape}")

# Normalize data (fit only on training data)
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)

# Save the scaler for later use
with open('transformer_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

def split_sequences(seq, seq_length):
    X, y = [], []
    for i in range(len(seq) - seq_length):
        X.append(seq[i:i+seq_length])
        y.append(seq[i+1:i+seq_length+1])  # next-step targets
    return np.array(X), np.array(y)

SEQ_LEN = 30
X, y = split_sequences(train_scaled, SEQ_LEN)

# Train/val split from the 90% training data (80/10 split of original data)
n = len(X)
train_end = int(0.85 * n)  # Use 85% for training, 15% for validation
X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:], y[train_end:]

print(f"Training sequences: {X_train.shape[0]}")
print(f"Validation sequences: {X_val.shape[0]}")

# PyTorch Dataset setup
def to_tensor(arr): 
    return torch.from_numpy(arr).float()

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = to_tensor(X)
        self.y = to_tensor(y)
    def __len__(self): 
        return len(self.X)
    def __getitem__(self, idx): 
        return self.X[idx], self.y[idx]

train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=32)

# 2. Create Transformer model
print("Creating Transformer model...")

model = SimpleTransformerForecast(
    input_size=len(features),
    d_model=128,
    num_heads=8,
    num_layers=3,
    dim_feedforward=512,
    dropout=0.1,
    seq_len=SEQ_LEN
).to(device)

# model = TimeSeriesTransformer(
#     input_size=len(features),
#     output_size=len(features),
#     d_model=128,
#     num_heads=8,
#     num_encoder_layers=3,
#     num_decoder_layers=3,
#     dim_feedforward=512,
#     dropout=0.1,
#     seq_len=SEQ_LEN,
#     pred_len=SEQ_LEN
# ).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
criterion = nn.MSELoss()

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


EPOCHS = 50
best_val = float('inf')
patience, trials = 10, 0
history = {'train_loss': [], 'val_loss': []}

print("Starting Transformer training...")
try:
    for epoch in range(1, EPOCHS+1):
        # Training phase
        model.train()
        tloss = []
        
        try:
            for batch_idx, (xb, yb) in enumerate(train_loader):
                xb, yb = xb.to(device), yb.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                preds = model(xb)
                
                # Calculate loss
                loss = criterion(preds, yb)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update weights
                optimizer.step()
                
                tloss.append(loss.item())
                
                # Print progress for first epoch
                if epoch == 1 and batch_idx % 50 == 0:
                    print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
        
        except Exception as e:
            print(f"Error during training at epoch {epoch}: {e}")
            break
        
        avg_train_loss = np.mean(tloss)
        history['train_loss'].append(avg_train_loss)

        # Validation phase
        model.eval()
        vloss = []
        
        try:
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    vloss.append(loss.item())
        
        except Exception as e:
            print(f"Error during validation at epoch {epoch}: {e}")
            break
        
        avg_val_loss = np.mean(vloss)
        history['val_loss'].append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")
        
        # Early stopping check
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': best_val,
            }, 'best_transformer_model.pth')
            trials = 0
        else:
            trials += 1
            if trials >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

except Exception as e:
    print(f"Training failed with error: {e}")
    print("Attempting to save current model state...")
    torch.save(model.state_dict(), 'transformer_model_checkpoint.pth')

# Load best model if it exists
try:
    checkpoint = torch.load('best_transformer_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint.get('epoch', 'unknown')}")
except:
    print("Using final model state")

# Save training history
with open('transformer_training_history.pkl', 'wb') as f:
    pickle.dump(history, f)

# Save model configuration
model_config = {
    'input_size': len(features),
    'd_model': 128,
    'num_heads': 8,
    'num_layers': 3,
    'dim_feedforward': 512,
    'dropout': 0.1,
    'seq_len': SEQ_LEN,
    'features': features,
    'model_type': 'SimpleTransformerForecast'
}
with open('transformer_model_config.pkl', 'wb') as f:
    pickle.dump(model_config, f)

# Plot training curves
if len(history['train_loss']) > 0:
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Transformer Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    if len(history['train_loss']) > 10:
        # Show last 80% of training for better view of convergence
        start_idx = len(history['train_loss']) // 5
        plt.plot(range(start_idx, len(history['train_loss'])), 
                history['train_loss'][start_idx:], label='Train Loss', color='blue')
        plt.plot(range(start_idx, len(history['val_loss'])), 
                history['val_loss'][start_idx:], label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Transformer Training Convergence (Last 80%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
else:
    print("No training history to plot")

print("Transformer training completed!")
if len(history['val_loss']) > 0:
    print(f"Best validation loss: {best_val:.6f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.6f}")
print("Model and preprocessing objects saved.")
print("Ready for anomaly detection phase!")

# Quick model test
print("\nTesting Transformer model...")
try:
    model.eval()
    with torch.no_grad():
        test_input = torch.randn(1, SEQ_LEN, len(features)).to(device)
        test_output = model(test_input)
        print(f"Model test successful! Input shape: {test_input.shape}, Output shape: {test_output.shape}")
except Exception as e:
    print(f"Model test failed: {e}")

print("\nTransformer model training files saved:")
print("- best_transformer_model.pth")
print("- transformer_model_config.pkl")
print("- transformer_scaler.pkl")
print("- transformer_training_history.pkl")
print("- transformer_test_data_for_anomaly.npy")
print("- transformer_test_dates.csv")
