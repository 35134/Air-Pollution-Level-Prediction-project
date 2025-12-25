"""
Model evaluation utilities for air quality prediction project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import json
from datetime import datetime

from .config import RESULTS_DIR, TARGET_VARIABLE

class ModelEvaluator:
    """Comprehensive model evaluation and visualization."""
    
    def __init__(self, model_name="Model"):
        self.model_name = model_name
        
    def evaluate_model(self, model, X_test, y_test, feature_names=None):
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            feature_names: Feature names for feature importance
            
        Returns:
            dict: Evaluation results
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred)
        
        # Feature importance if available
        if hasattr(model, 'feature_importances_'):
            importance_dict = dict(zip(feature_names, model.feature_importances_))
            metrics['feature_importance'] = importance_dict
        
        return {
            'predictions': y_pred,
            'actual': y_test,
            'metrics': metrics
        }
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive evaluation metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Additional metrics
        metrics['mean_actual'] = np.mean(y_true)
        metrics['mean_predicted'] = np.mean(y_pred)
        metrics['std_actual'] = np.std(y_true)
        metrics['std_predicted'] = np.std(y_pred)
        
        # Residual analysis
        residuals = y_true - y_pred
        metrics['mean_residual'] = np.mean(residuals)
        metrics['std_residual'] = np.std(residuals)
        
        # Percentage within different error thresholds
        abs_errors = np.abs(residuals)
        metrics['pct_within_5'] = np.mean(abs_errors <= 5) * 100
        metrics['pct_within_10'] = np.mean(abs_errors <= 10) * 100
        metrics['pct_within_15'] = np.mean(abs_errors <= 15) * 100
        metrics['pct_within_25'] = np.mean(abs_errors <= 25) * 100
        
        return metrics
    
    def plot_predictions_vs_actual(self, y_true, y_pred, sample_size=5000, save_path=None):
        """
        Create prediction vs actual scatter plot.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            sample_size: Number of points to sample for visualization
            save_path: Path to save the plot
            
        Returns:
            matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Sample data if too large
        if sample_size and len(y_true) > sample_size:
            indices = np.random.choice(len(y_true), sample_size, replace=False)
            y_true_sample = y_true[indices]
            y_pred_sample = y_pred[indices]
        else:
            y_true_sample = y_true
            y_pred_sample = y_pred
        
        # Create scatter plot
        ax.scatter(y_true_sample, y_pred_sample, alpha=0.6, s=1)
        
        # Add perfect prediction line
        min_val = min(y_true_sample.min(), y_pred_sample.min())
        max_val = max(y_true_sample.max(), y_pred_sample.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        # Labels and title
        ax.set_xlabel('Actual PM2.5 (μg/m³)')
        ax.set_ylabel('Predicted PM2.5 (μg/m³)')
        ax.set_title(f'{self.model_name}: Predictions vs Actual')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved predictions vs actual plot to {save_path}")
        
        return fig
    
    def plot_residuals(self, y_true, y_pred, save_path=None):
        """
        Create residual analysis plots.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            save_path: Path to save the plot
            
        Returns:
            matplotlib figure
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6, s=1)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted PM2.5')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals histogram
        axes[0, 1].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Residuals Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot of Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals vs Actual
        axes[1, 1].scatter(y_true, residuals, alpha=0.6, s=1)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Actual PM2.5')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals vs Actual')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved residuals plot to {save_path}")
        
        return fig
    
    def plot_feature_importance(self, feature_importance_dict, top_n=15, save_path=None):
        """
        Plot feature importance.
        
        Args:
            feature_importance_dict: Dictionary of feature names and importances
            top_n: Number of top features to show
            save_path: Path to save the plot
            
        Returns:
            matplotlib figure
        """
        # Sort features by importance
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        
        features = [item[0] for item in top_features]
        importances = [item[1] for item in top_features]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create horizontal bar plot
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importances, color='skyblue', edgecolor='navy')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('Importance')
        ax.set_title(f'{self.model_name}: Top {top_n} Feature Importances')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(importances):
            ax.text(v + 0.001, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved feature importance plot to {save_path}")
        
        return fig

def compare_models(results_dict, save_path=None):
    """
    Compare multiple models' performance.
    
    Args:
        results_dict: Dictionary with model names as keys and results as values
        save_path: Path to save comparison plot
        
    Returns:
        matplotlib figure
    """
    # Extract metrics for comparison
    model_names = list(results_dict.keys())
    metrics = ['rmse', 'mae', 'r2', 'pct_within_10', 'pct_within_25']
    
    # Create comparison table
    comparison_data = []
    for model_name in model_names:
        model_metrics = results_dict[model_name]['metrics']
        row = [
            model_metrics['rmse'],
            model_metrics['mae'],
            model_metrics['r2'],
            model_metrics['pct_within_10'],
            model_metrics['pct_within_25']
        ]
        comparison_data.append(row)
    
    # Create DataFrame
    df_comparison = pd.DataFrame(comparison_data, 
                                index=model_names, 
                                columns=['RMSE', 'MAE', 'R²', 'Within 10μg/m³ (%)', 'Within 25μg/m³ (%)'])
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # RMSE comparison
    rmse_values = [results_dict[name]['metrics']['rmse'] for name in model_names]
    axes[0].bar(model_names, rmse_values, color=['lightcoral', 'lightblue'])
    axes[0].set_ylabel('RMSE (μg/m³)')
    axes[0].set_title('Model Comparison: RMSE')
    axes[0].grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(rmse_values):
        axes[0].text(i, v + 0.5, f'{v:.2f}', ha='center', va='bottom')
    
    # R² comparison
    r2_values = [results_dict[name]['metrics']['r2'] for name in model_names]
    axes[1].bar(model_names, r2_values, color=['lightcoral', 'lightblue'])
    axes[1].set_ylabel('R² Score')
    axes[1].set_title('Model Comparison: R² Score')
    axes[1].grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(r2_values):
        axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved model comparison plot to {save_path}")
    
    return fig, df_comparison

def save_results(results_dict, filename='model_results.json'):
    """Save evaluation results to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for model_name, results in results_dict.items():
        json_results[model_name] = {
            'metrics': results['metrics'],
            'predictions': results['predictions'].tolist() if isinstance(results['predictions'], np.ndarray) else results['predictions'],
            'actual': results['actual'].tolist() if isinstance(results['actual'], np.ndarray) else results['actual']
        }
    
    filepath = RESULTS_DIR / filename
    with open(filepath, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"Results saved to {filepath}")
    return filepath
# Final Project

# Final Project Module

