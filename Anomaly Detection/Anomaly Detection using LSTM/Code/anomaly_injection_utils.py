import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

class AnomalyInjector:
    """
    Utility class for injecting different types of anomalies into time series data
    """
    
    def __init__(self, data: np.ndarray):
        self.original_data = data.copy()
        self.modified_data = data.copy()
        self.anomaly_log = []
        
    def inject_spike(self, indices: List[int], intensity: float = 3.0, features: List[int] = None):
        """Inject spike anomalies (sudden high values)"""
        if features is None:
            features = list(range(self.original_data.shape[1]))
            
        for idx in indices:
            if 0 <= idx < len(self.original_data):
                for feature_idx in features:
                    feature_std = np.std(self.original_data[:, feature_idx])
                    self.modified_data[idx, feature_idx] += intensity * feature_std
                    
                self.anomaly_log.append({
                    'index': idx,
                    'type': 'spike',
                    'intensity': intensity,
                    'features': features
                })
    
    def inject_drop(self, indices: List[int], intensity: float = 3.0, features: List[int] = None):
        """Inject drop anomalies (sudden low values)"""
        if features is None:
            features = list(range(self.original_data.shape[1]))
            
        for idx in indices:
            if 0 <= idx < len(self.original_data):
                for feature_idx in features:
                    feature_std = np.std(self.original_data[:, feature_idx])
                    self.modified_data[idx, feature_idx] -= intensity * feature_std
                    
                self.anomaly_log.append({
                    'index': idx,
                    'type': 'drop',
                    'intensity': intensity,
                    'features': features
                })
    
    def inject_drift(self, start_idx: int, duration: int = 10, intensity: float = 2.0, features: List[int] = None):
        """Inject gradual drift anomalies"""
        if features is None:
            features = list(range(self.original_data.shape[1]))
            
        for i in range(duration):
            idx = start_idx + i
            if 0 <= idx < len(self.original_data):
                for feature_idx in features:
                    feature_std = np.std(self.original_data[:, feature_idx])
                    drift_amount = (i / duration) * intensity * feature_std
                    self.modified_data[idx, feature_idx] += drift_amount
        
        self.anomaly_log.append({
            'index': start_idx,
            'type': 'drift',
            'duration': duration,
            'intensity': intensity,
            'features': features
        })
    
    def inject_noise(self, indices: List[int], intensity: float = 2.0, features: List[int] = None):
        """Inject random noise anomalies"""
        if features is None:
            features = list(range(self.original_data.shape[1]))
            
        for idx in indices:
            if 0 <= idx < len(self.original_data):
                for feature_idx in features:
                    feature_mean = np.mean(self.original_data[:, feature_idx])
                    feature_std = np.std(self.original_data[:, feature_idx])
                    self.modified_data[idx, feature_idx] = np.random.normal(
                        feature_mean, intensity * feature_std
                    )
                    
                self.anomaly_log.append({
                    'index': idx,
                    'type': 'noise',
                    'intensity': intensity,
                    'features': features
                })
    
    def inject_seasonal_shift(self, start_idx: int, duration: int = 20, shift_amount: float = 1.5, features: List[int] = None):
        """Inject seasonal shift anomalies (phase shift in patterns)"""
        if features is None:
            features = list(range(self.original_data.shape[1]))
            
        for i in range(duration):
            idx = start_idx + i
            if 0 <= idx < len(self.original_data):
                for feature_idx in features:
                    feature_std = np.std(self.original_data[:, feature_idx])
                    # Create a sinusoidal shift
                    shift = shift_amount * feature_std * np.sin(2 * np.pi * i / duration)
                    self.modified_data[idx, feature_idx] += shift
        
        self.anomaly_log.append({
            'index': start_idx,
            'type': 'seasonal_shift',
            'duration': duration,
            'shift_amount': shift_amount,
            'features': features
        })
    
    def get_anomaly_mask(self) -> np.ndarray:
        """Get binary mask indicating which points are anomalous"""
        mask = np.zeros(len(self.original_data), dtype=int)
        
        for anomaly in self.anomaly_log:
            idx = anomaly['index']
            if anomaly['type'] == 'drift':
                duration = anomaly.get('duration', 10)
                for i in range(duration):
                    if 0 <= idx + i < len(mask):
                        mask[idx + i] = 1
            elif anomaly['type'] == 'seasonal_shift':
                duration = anomaly.get('duration', 20)
                for i in range(duration):
                    if 0 <= idx + i < len(mask):
                        mask[idx + i] = 1
            else:
                if 0 <= idx < len(mask):
                    mask[idx] = 1
        
        return mask
    
    def plot_anomalies(self, feature_names: List[str] = None, figsize: Tuple[int, int] = (15, 10)):
        """Plot original vs modified data with anomalies highlighted"""
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(self.original_data.shape[1])]
        
        n_features = self.original_data.shape[1]
        fig, axes = plt.subplots(n_features, 1, figsize=figsize, sharex=True)
        if n_features == 1:
            axes = [axes]
        
        anomaly_mask = self.get_anomaly_mask()
        
        for i, (ax, feature_name) in enumerate(zip(axes, feature_names)):
            # Plot original and modified data
            ax.plot(self.original_data[:, i], label='Original', alpha=0.7, color='blue')
            ax.plot(self.modified_data[:, i], label='With Anomalies', alpha=0.8, color='red')
            
            # Highlight anomalous points
            anomaly_indices = np.where(anomaly_mask == 1)[0]
            if len(anomaly_indices) > 0:
                ax.scatter(anomaly_indices, self.modified_data[anomaly_indices, i], 
                          color='orange', s=50, label='Anomalies', zorder=5)
            
            ax.set_title(f'{feature_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.xlabel('Time Step')
        plt.tight_layout()
        plt.show()
    
    def get_summary(self) -> Dict:
        """Get summary of injected anomalies"""
        anomaly_types = {}
        for anomaly in self.anomaly_log:
            atype = anomaly['type']
            if atype not in anomaly_types:
                anomaly_types[atype] = 0
            anomaly_types[atype] += 1
        
        return {
            'total_anomalies': len(self.anomaly_log),
            'anomaly_types': anomaly_types,
            'anomalous_points': self.get_anomaly_mask().sum(),
            'anomaly_percentage': (self.get_anomaly_mask().sum() / len(self.original_data)) * 100
        }
    
    def reset(self):
        """Reset to original data"""
        self.modified_data = self.original_data.copy()
        self.anomaly_log = []


class AnomalyDetectionEvaluator:
    """
    Utility class for evaluating anomaly detection performance
    """
    
    def __init__(self, y_true: np.ndarray, y_scores: np.ndarray, y_pred: np.ndarray = None):
        self.y_true = y_true
        self.y_scores = y_scores
        self.y_pred = y_pred
        
    def calculate_threshold_metrics(self, thresholds: np.ndarray = None) -> pd.DataFrame:
        """Calculate precision, recall, F1 for different thresholds"""
        if thresholds is None:
            thresholds = np.percentile(self.y_scores, np.arange(80, 100, 1))
        
        results = []
        for threshold in thresholds:
            y_pred = (self.y_scores > threshold).astype(int)
            
            # Handle edge cases
            if len(np.unique(y_pred)) == 1:
                if y_pred[0] == 1:  # All predicted as anomalies
                    precision = self.y_true.sum() / len(y_pred) if len(y_pred) > 0 else 0
                    recall = 1.0 if self.y_true.sum() > 0 else 0
                else:  # All predicted as normal
                    precision = 0
                    recall = 0
                f1 = 0
            else:
                try:
                    from sklearn.metrics import precision_score, recall_score, f1_score
                    precision = precision_score(self.y_true, y_pred, zero_division=0)
                    recall = recall_score(self.y_true, y_pred, zero_division=0)
                    f1 = f1_score(self.y_true, y_pred, zero_division=0)
                except:
                    precision = recall = f1 = 0
            
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'anomalies_detected': y_pred.sum()
            })
        
        return pd.DataFrame(results)
    
    def find_optimal_threshold(self, metric: str = 'f1_score') -> Tuple[float, Dict]:
        """Find optimal threshold based on specified metric"""
        threshold_results = self.calculate_threshold_metrics()
        
        if len(threshold_results) == 0:
            return 0.0, {}
        
        optimal_idx = threshold_results[metric].idxmax()
        optimal_row = threshold_results.iloc[optimal_idx]
        
        return optimal_row['threshold'], optimal_row.to_dict()
    
    def plot_threshold_analysis(self, figsize: Tuple[int, int] = (12, 8)):
        """Plot precision, recall, F1 vs threshold"""
        threshold_results = self.calculate_threshold_metrics()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Precision vs Threshold
        axes[0, 0].plot(threshold_results['threshold'], threshold_results['precision'], 'b-', label='Precision')
        axes[0, 0].set_xlabel('Threshold')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].set_title('Precision vs Threshold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Recall vs Threshold
        axes[0, 1].plot(threshold_results['threshold'], threshold_results['recall'], 'r-', label='Recall')
        axes[0, 1].set_xlabel('Threshold')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].set_title('Recall vs Threshold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1-Score vs Threshold
        axes[1, 0].plot(threshold_results['threshold'], threshold_results['f1_score'], 'g-', label='F1-Score')
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].set_title('F1-Score vs Threshold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Number of detected anomalies vs Threshold
        axes[1, 1].plot(threshold_results['threshold'], threshold_results['anomalies_detected'], 'purple', label='Detected Anomalies')
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel('Number of Detected Anomalies')
        axes[1, 1].set_title('Detected Anomalies vs Threshold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return threshold_results
    
    def plot_score_distribution(self, bins: int = 50, figsize: Tuple[int, int] = (12, 5)):
        """Plot distribution of anomaly scores for normal vs anomalous points"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        normal_scores = self.y_scores[self.y_true == 0]
        anomaly_scores = self.y_scores[self.y_true == 1]
        
        # Histogram
        axes[0].hist(normal_scores, bins=bins, alpha=0.7, label='Normal', color='blue', density=True)
        axes[0].hist(anomaly_scores, bins=bins, alpha=0.7, label='Anomaly', color='red', density=True)
        axes[0].set_xlabel('Anomaly Score')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Score Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot([normal_scores, anomaly_scores], labels=['Normal', 'Anomaly'])
        axes[1].set_ylabel('Anomaly Score')
        axes[1].set_title('Score Distribution (Box Plot)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print("Score Statistics:")
        print(f"Normal points - Mean: {normal_scores.mean():.6f}, Std: {normal_scores.std():.6f}")
        print(f"Anomaly points - Mean: {anomaly_scores.mean():.6f}, Std: {anomaly_scores.std():.6f}")
        
        if len(anomaly_scores) > 0 and len(normal_scores) > 0:
            from scipy import stats
            statistic, p_value = stats.ttest_ind(normal_scores, anomaly_scores)
            print(f"T-test p-value: {p_value:.6f} (significant difference: {p_value < 0.05})")


# Example usage and demonstration
def demonstrate_anomaly_injection():
    """Demonstrate how to use the anomaly injection utilities"""
    
    # Create sample time series data
    np.random.seed(42)
    n_points = 100
    n_features = 4
    
    # Generate synthetic time series with some patterns
    t = np.linspace(0, 4*np.pi, n_points)
    data = np.column_stack([
        10 + 5 * np.sin(t) + np.random.normal(0, 0.5, n_points),  # temp_min
        20 + 8 * np.sin(t + np.pi/4) + np.random.normal(0, 0.8, n_points),  # temp_max
        60 + 20 * np.cos(t) + np.random.normal(0, 2, n_points),  # humidity_min
        80 + 15 * np.cos(t + np.pi/2) + np.random.normal(0, 3, n_points)  # humidity_max
    ])
    
    feature_names = ['min_temp', 'max_temp', 'min_humidity', 'max_humidity']
    
    # Create anomaly injector
    injector = AnomalyInjector(data)
    
    # Inject different types of anomalies
    injector.inject_spike([15, 25], intensity=2.5, features=[0, 1])  # Temperature spikes
    injector.inject_drop([35, 45], intensity=3.0, features=[2, 3])   # Humidity drops
    injector.inject_drift(60, duration=15, intensity=2.0, features=[0])  # Temperature drift
    injector.inject_noise([75, 80, 85], intensity=2.5)  # Random noise
    injector.inject_seasonal_shift(20, duration=10, shift_amount=1.5, features=[2])  # Humidity shift
    
    # Plot results
    injector.plot_anomalies(feature_names)
    
    # Get summary
    summary = injector.get_summary()
    print("Anomaly Injection Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    return injector

if __name__ == "__main__":
    # Run demonstration
    injector = demonstrate_anomaly_injection()