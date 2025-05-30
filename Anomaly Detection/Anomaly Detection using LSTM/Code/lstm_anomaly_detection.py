import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
import pickle
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    import torch_directml
    if torch_directml.is_available():
        device = torch.device('cpu')  # Use CPU for consistency with training
        print(f"Using device: {device}")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device}")
except ImportError:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

# Load saved model configuration and preprocessing objects
with open('model_config.pkl', 'rb') as f:
    model_config = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load test data
test_data = np.load('test_data_for_anomaly.npy')
test_dates = pd.read_csv('test_dates.csv', parse_dates=['DateTime'], index_col=0)

print(f"Test data shape: {test_data.shape}")
print(f"Date range: {test_dates.index[0]} to {test_dates.index[-1]}")


class LSTMForecast(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers,
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, input_dim)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize LSTM weights"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    
    def forward(self, x):
        # Initialize hidden state
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        # Forward pass through LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Apply linear layer to all time steps
        out = self.fc(out)
        
        return out

model_params = {k: v for k, v in model_config.items() if k in ['input_dim', 'hidden_dim', 'num_layers', 'dropout']}
model = LSTMForecast(**model_params).to(device)

# Load model weights
try:
    checkpoint = torch.load('best_model.pth', map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model from checkpoint")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model state dict")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Trying to load direct state dict...")
    model.load_state_dict(torch.load('best_model.pth', map_location=device))

model.eval()
print("Model loaded successfully!")


SEQ_LEN = model_config['seq_len']
features = model_config['features']

# Create DataFrame for results
results_df = pd.DataFrame(index=test_dates.index)
for i, feature in enumerate(features):
    results_df[f'actual_{feature}'] = test_data[:, i]

# Initialize anomaly column (0 = normal, 1 = anomaly)
results_df['anomaly_true'] = 0

# Function to inject anomalies
def inject_anomalies(data, indices, anomaly_type='spike', intensity=3.0):
    """
    Inject anomalies into the data
    
    Args:
        data: numpy array of data
        indices: list of indices where to inject anomalies
        anomaly_type: 'spike', 'drop', 'drift', 'noise'
        intensity: how strong the anomaly should be (in terms of std deviations)
    """
    modified_data = data.copy()
    
    for idx in indices:
        if idx < len(data):
            for feature_idx in range(data.shape[1]):
                original_val = data[idx, feature_idx]
                feature_std = np.std(data[:, feature_idx])
                feature_mean = np.mean(data[:, feature_idx])
                
                if anomaly_type == 'spike':
                    modified_data[idx, feature_idx] = original_val + (intensity * feature_std)
                elif anomaly_type == 'drop':
                    modified_data[idx, feature_idx] = original_val - (intensity * feature_std)
                elif anomaly_type == 'drift':
                    # Gradual drift over several points
                    for j in range(min(10, len(data) - idx)):
                        if idx + j < len(data):
                            modified_data[idx + j, feature_idx] += (j * intensity * feature_std / 10)
                elif anomaly_type == 'noise':
                    modified_data[idx, feature_idx] = np.random.normal(feature_mean, intensity * feature_std)
    
    return modified_data

# Inject anomalies at specific indices
print("\nInjecting anomalies...")
anomaly_indices = [50, 150, 250, 350, 450, 550, 650, 750, 850, 950] 
anomaly_types = ['spike', 'drop', 'spike', 'noise', 'drift', 'drop', 'spike', 'noise', 'drop', 'spike']


test_data_with_anomalies = test_data.copy()
for i, (idx, anom_type) in enumerate(zip(anomaly_indices, anomaly_types)):
    if idx < len(test_data):
        test_data_with_anomalies = inject_anomalies(test_data_with_anomalies, [idx], anom_type, intensity=2.5)
        results_df.iloc[idx, results_df.columns.get_loc('anomaly_true')] = 1
        
        if anom_type == 'drift':
            for j in range(1, min(10, len(test_data) - idx)):
                if idx + j < len(test_data):
                    results_df.iloc[idx + j, results_df.columns.get_loc('anomaly_true')] = 1

# Update DataFrame with modified data
for i, feature in enumerate(features):
    results_df[f'modified_{feature}'] = test_data_with_anomalies[:, i]

print(f"Injected {len(anomaly_indices)} anomalies at indices: {anomaly_indices}")
print(f"Total anomalous points: {results_df['anomaly_true'].sum()}")

# Normalize the modified test data
test_data_scaled = scaler.transform(test_data_with_anomalies)

# Step-by-step prediction
print("\nStarting step-by-step prediction...")
predictions = []
reconstruction_errors = []



for i in range(len(test_data_scaled) - SEQ_LEN):
    seq = test_data_scaled[i:i+SEQ_LEN]
    
    input_seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Predict next step
    with torch.no_grad():
        try:
            pred = model(input_seq)
            next_pred = pred[0, -1, :].cpu().numpy()  # Last timestep prediction
        except Exception as e:
            print(f"Prediction error at step {i}: {e}")
            next_pred = np.zeros(len(features))
    
    predictions.append(next_pred)
    
    # Calculate reconstruction error (anomaly score)
    if i + SEQ_LEN < len(test_data_scaled):
        actual_next = test_data_scaled[i + SEQ_LEN]
        error = np.mean((next_pred - actual_next) ** 2)  # MSE
        reconstruction_errors.append(error)
    else:
        reconstruction_errors.append(0)

print(f"Completed {len(predictions)} predictions")

# Convert predictions back to original scale
predictions = np.array(predictions)
if len(predictions) > 0:
    predictions_original = scaler.inverse_transform(predictions)
    print("Predictions converted to original scale")
else:
    predictions_original = np.array([])
    print("No predictions generated")


prediction_start_idx = SEQ_LEN
for i, feature in enumerate(features):
    pred_values = np.full(len(results_df), np.nan)
    if len(predictions_original) > 0:
        end_idx = min(prediction_start_idx + len(predictions_original), len(pred_values))
        pred_values[prediction_start_idx:end_idx] = predictions_original[:end_idx-prediction_start_idx, i]
    results_df[f'predicted_{feature}'] = pred_values

# Calculate anomaly scores and detection
reconstruction_errors = np.array(reconstruction_errors)
if len(reconstruction_errors) > 0:
    # Use multiple thresholds for comparison
    thresholds = {
        'percentile_90': np.percentile(reconstruction_errors, 90),
        'percentile_95': np.percentile(reconstruction_errors, 95),
        'percentile_99': np.percentile(reconstruction_errors, 99)
    }
    
    print(f"\nAnomaly detection thresholds:")
    for name, threshold in thresholds.items():
        print(f"  {name}: {threshold:.6f}")
    
    # Using 95th percentile as primary threshold
    anomaly_threshold = thresholds['percentile_95']
    anomaly_detected = (reconstruction_errors > anomaly_threshold).astype(int)
    
    # Add anomaly detection results to DataFrame
    anomaly_detected_full = np.zeros(len(results_df))
    error_start_idx = SEQ_LEN
    if len(anomaly_detected) > 0:
        end_idx = min(error_start_idx + len(anomaly_detected), len(anomaly_detected_full))
        anomaly_detected_full[error_start_idx:end_idx] = anomaly_detected[:end_idx-error_start_idx]
    
    results_df['anomaly_detected'] = anomaly_detected_full
    
    # Add reconstruction errors
    recon_errors_full = np.full(len(results_df), np.nan)
    if len(reconstruction_errors) > 0:
        end_idx = min(error_start_idx + len(reconstruction_errors), len(recon_errors_full))
        recon_errors_full[error_start_idx:end_idx] = reconstruction_errors[:end_idx-error_start_idx]
    results_df['reconstruction_error'] = recon_errors_full
    
    # Calculate evaluation metrics for anomaly detection
    valid_indices = ~np.isnan(results_df['reconstruction_error'].values)
    if valid_indices.sum() > 0:
        y_true = results_df['anomaly_true'].values[valid_indices]
        y_pred = results_df['anomaly_detected'].values[valid_indices]
        
        if len(y_true) > 0 and len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            cm = confusion_matrix(y_true, y_pred)
            
            print(f"\nAnomaly Detection Results:")
            print(f"Threshold: {anomaly_threshold:.6f}")
            print(f"Precision: {precision:.3f}")
            print(f"Recall: {recall:.3f}")
            print(f"F1-Score: {f1:.3f}")
        else:
            print("Insufficient variation in true/predicted labels for complete evaluation")
            print(f"True anomalies: {y_true.sum()}")
            print(f"Detected anomalies: {y_pred.sum()}")
    else:
        print("No valid reconstruction errors for evaluation")
else:
    print("No reconstruction errors calculated")

# Calculate prediction accuracy for non-anomalous points
normal_mask = (results_df['anomaly_true'] == 0) & (~np.isnan(results_df[f'predicted_{features[0]}']))
if normal_mask.sum() > 0:
    mae_scores = {}
    mse_scores = {}
    r2_scores = {}
    
    for feature in features:
        actual = results_df[normal_mask][f'modified_{feature}'].values
        predicted = results_df[normal_mask][f'predicted_{feature}'].values
        
        mae_scores[feature] = mean_absolute_error(actual, predicted)
        mse_scores[feature] = mean_squared_error(actual, predicted)
        r2_scores[feature] = r2_score(actual, predicted)
    
    print(f"\nPrediction Accuracy on Normal Points:")
    for feature in features:
        print(f"{feature} - MAE: {mae_scores[feature]:.3f}, MSE: {mse_scores[feature]:.3f}, R²: {r2_scores[feature]:.3f}")
else:
    print("No normal points with valid predictions for accuracy calculation")

# Save results
results_df.to_csv('anomaly_detection_results.csv')
print(f"\nResults saved to 'anomaly_detection_results.csv'")

# Create comprehensive plots
print("\nCreating visualization plots...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('LSTM Anomaly Detection Results', fontsize=16)

# Plot each feature
for i, feature in enumerate(features):
    ax = axes[i//2, i%2]
    
    # Plot actual vs predicted
    ax.plot(results_df.index, results_df[f'modified_{feature}'], 
            label='Actual (with anomalies)', color='blue', alpha=0.7, linewidth=1)
    ax.plot(results_df.index, results_df[f'predicted_{feature}'], 
            label='Predicted', color='red', linestyle='--', alpha=0.8, linewidth=1)
    
    # Highlight true anomalies
    true_anomalies = results_df[results_df['anomaly_true'] == 1]
    if len(true_anomalies) > 0:
        ax.scatter(true_anomalies.index, true_anomalies[f'modified_{feature}'], 
                  color='orange', s=60, label='True Anomalies', zorder=5, marker='o')
    
    # Highlight detected anomalies
    detected_anomalies = results_df[results_df['anomaly_detected'] == 1]
    if len(detected_anomalies) > 0:
        ax.scatter(detected_anomalies.index, detected_anomalies[f'modified_{feature}'], 
                  color='red', s=40, marker='x', label='Detected Anomalies', zorder=6)
    
    ax.set_title(f'{feature}')
    ax.set_xlabel('Date')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Summary statistics
print(f"\n=== FINAL SUMMARY ===")
print(f"Total data points: {len(results_df)}")
print(f"Prediction points: {(~np.isnan(results_df[f'predicted_{features[0]}'])).sum()}")
print(f"True anomalies: {results_df['anomaly_true'].sum()}")
print(f"Detected anomalies: {results_df['anomaly_detected'].sum()}")
if len(reconstruction_errors) > 0:
    print(f"Mean reconstruction error: {np.mean(reconstruction_errors):.6f}")
    print(f"Std reconstruction error: {np.std(reconstruction_errors):.6f}")
    print(f"Max reconstruction error: {np.max(reconstruction_errors):.6f}")

print("\nAnomaly detection completed successfully!")
