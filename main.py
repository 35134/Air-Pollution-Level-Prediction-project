"""
Main execution script for Air Pollution Level Prediction Project
Polished version with clean, organized structure
"""

import sys
import os
sys.path.append('src')

from src.models import BaselineModel, OptimizedModel
import pandas as pd
from pathlib import Path

def main():
    """Main execution function for the air pollution prediction project."""
    
    print("=" * 60)
    print("AIR POLLUTION LEVEL PREDICTION PROJECT")
    print("=" * 60)
    print("\nStarting project execution...\n")
    
    try:
        # Step 1: Load and prepare data
        print("Step 1: Loading and preparing data...")
        
        # Load preprocessed data
        train_df = pd.read_csv('datasets/cleaned/train_cleaned.csv', index_col=0, parse_dates=True)
        val_df = pd.read_csv('datasets/cleaned/val_cleaned.csv', index_col=0, parse_dates=True)
        test_df = pd.read_csv('datasets/cleaned/test_cleaned.csv', index_col=0, parse_dates=True)
        
        print("Data loaded successfully!")
        print(f"   Training set: {train_df.shape}")
        print(f"   Validation set: {val_df.shape}")
        print(f"   Test set: {test_df.shape}")
        
        # Step 2: Train baseline model
        print("\nStep 2: Training baseline model...")
        
        # Initialize and train baseline model
        baseline_model = BaselineModel()
        
        # Prepare features for baseline
        X_train_base = baseline_model.prepare_features(train_df, feature_type='raw')
        X_val_base = baseline_model.prepare_features(val_df, feature_type='raw')
        X_test_base = baseline_model.prepare_features(test_df, feature_type='raw')
        
        # Create complete dataset by combining features and target
        train_data = pd.concat([X_train_base, train_df['PM2.5']], axis=1).dropna()
        val_data = pd.concat([X_val_base, val_df['PM2.5']], axis=1).dropna()
        test_data = pd.concat([X_test_base, test_df['PM2.5']], axis=1).dropna()
        
        # Separate features and target after dropping NaN
        X_train_clean = train_data.drop('PM2.5', axis=1)
        y_train_clean = train_data['PM2.5']
        X_val_clean = val_data.drop('PM2.5', axis=1)
        y_val_clean = val_data['PM2.5']
        X_test_clean = test_data.drop('PM2.5', axis=1)
        y_test_clean = test_data['PM2.5']
        
        # Train baseline model
        baseline_metrics = baseline_model.train(X_train_clean, y_train_clean, X_val_clean, y_val_clean)
        baseline_test_metrics = baseline_model.evaluate(X_test_clean, y_test_clean)
        
        print("Baseline model trained!")
        print(f"   Test RMSE: {baseline_test_metrics['rmse']:.2f} ug/m3")
        print(f"   Test R2: {baseline_test_metrics['r2']:.3f}")
        
        # Step 3: Use a simpler approach for optimized model
        print("\nStep 3: Training optimized model with consistent features...")
        
        # Load engineered data but ensure consistency
        train_eng = pd.read_csv('datasets/cleaned/train_engineered_optimized.csv', index_col=0, parse_dates=True)
        val_eng = pd.read_csv('datasets/cleaned/val_engineered_optimized.csv', index_col=0, parse_dates=True)
        test_eng = pd.read_csv('datasets/cleaned/test_engineered_optimized.csv', index_col=0, parse_dates=True)
        
        # Use only numeric features that exist in all datasets
        common_features = set(train_eng.columns) & set(val_eng.columns) & set(test_eng.columns)
        common_features = list(common_features - {'PM2.5', 'station', 'datetime'})
        
        # Filter to only numeric features for XGBoost
        numeric_features = []
        for feature in common_features:
            if train_eng[feature].dtype in ['float64', 'int64', 'bool']:
                numeric_features.append(feature)
        
        common_features = numeric_features
        print(f"Using {len(common_features)} common numeric features across all datasets")
        
        # Initialize and train optimized model
        optimized_model = OptimizedModel()
        
        # Prepare consistent features
        X_train_opt = train_eng[common_features].copy()
        X_val_opt = val_eng[common_features].copy()
        X_test_opt = test_eng[common_features].copy()
        
        # Fill numeric columns with median
        for col in common_features:
            median_val = X_train_opt[col].median()
            X_train_opt[col] = X_train_opt[col].fillna(median_val)
            X_val_opt[col] = X_val_opt[col].fillna(median_val)
            X_test_opt[col] = X_test_opt[col].fillna(median_val)
        
        # Create complete dataset by combining features and target
        train_opt_data = pd.concat([X_train_opt, train_eng['PM2.5']], axis=1).dropna()
        val_opt_data = pd.concat([X_val_opt, val_eng['PM2.5']], axis=1).dropna()
        test_opt_data = pd.concat([X_test_opt, test_eng['PM2.5']], axis=1).dropna()
        
        # Separate features and target after dropping NaN
        X_train_opt_clean = train_opt_data.drop('PM2.5', axis=1)
        y_train_opt_clean = train_opt_data['PM2.5']
        X_val_opt_clean = val_opt_data.drop('PM2.5', axis=1)
        y_val_opt_clean = val_opt_data['PM2.5']
        X_test_opt_clean = test_opt_data.drop('PM2.5', axis=1)
        y_test_opt_clean = test_opt_data['PM2.5']
        
        # Train optimized model
        optimized_metrics = optimized_model.train(X_train_opt_clean, y_train_opt_clean, 
                                                X_val_opt_clean, y_val_opt_clean, optimize=False)
        optimized_test_metrics = optimized_model.evaluate(X_test_opt_clean, y_test_opt_clean)
        
        print("Optimized model trained!")
        print(f"   Test RMSE: {optimized_test_metrics['rmse']:.2f} ug/m3")
        print(f"   Test R2: {optimized_test_metrics['r2']:.3f}")
        
        # Step 4: Compare models
        print("\nStep 4: Comparing models...")
        
        # Calculate improvement
        rmse_improvement = ((baseline_test_metrics['rmse'] - optimized_test_metrics['rmse']) / baseline_test_metrics['rmse']) * 100
        r2_improvement = ((optimized_test_metrics['r2'] - baseline_test_metrics['r2']) / baseline_test_metrics['r2']) * 100
        
        print("Performance Improvement:")
        print(f"   RMSE improvement: {rmse_improvement:.1f}%")
        print(f"   R2 improvement: {r2_improvement:.1f}%")
        
        # Step 5: Save results
        print("\nStep 5: Saving results...")
        
        # Save models
        baseline_model.save_model('results_models/baseline_model.pkl')
        optimized_model.save_model('results_models/optimized_model.pkl')
        
        # Save comparison
        comparison = {
            'baseline_rmse': baseline_test_metrics['rmse'],
            'optimized_rmse': optimized_test_metrics['rmse'],
            'improvement_pct': rmse_improvement
        }
        
        import json
        with open('results_models/model_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print("Results saved successfully!")
        
        print("\n" + "=" * 60)
        print("PROJECT EXECUTION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nKey Results:")
        print(f"   • Baseline Model RMSE: {baseline_test_metrics['rmse']:.2f} ug/m3")
        print(f"   • Optimized Model RMSE: {optimized_test_metrics['rmse']:.2f} ug/m3")
        print(f"   • Performance Improvement: {rmse_improvement:.1f}%")
        print("\nCheck the 'results_models/' directory for detailed outputs and visualizations.")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\nProject completed successfully! Check the results directory for outputs.")
    else:
        print("\nProject failed. Please check the error messages above.")
        sys.exit(1)


# Final Project - Air Pollution Prediction

