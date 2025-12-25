"""
Model definitions for the Air Pollution Prediction project.

This module contains:
1. Baseline model: Linear Regression
2. Optimized model: XGBoost with hyperparameter tuning
3. Model pipelines with preprocessing
4. Model persistence and loading
"""

import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

# Scikit-learn imports
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# XGBoost imports
import xgboost as xgb

# Optuna for hyperparameter optimization
import optuna

# Import configuration
from config import (
    TARGET_VARIABLE, RANDOM_STATE, N_JOBS, LINEAR_REGRESSION_PARAMS,
    XGBOOST_BASE_PARAMS, XGBOOST_SEARCH_SPACE, OPTUNA_N_TRIALS,
    OPTUNA_TIMEOUT, OPTUNA_N_JOBS, EARLY_STOPPING_ROUNDS,
    EARLY_STOPPING_MIN_DELTA, CV_N_SPLITS, SAVE_MODELS, RESULTS_DIR
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineModel:
    """
    Baseline Linear Regression model for PM2.5 prediction.
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.training_params = None
        self.metrics = {}
        self.is_fitted = False
        
    def prepare_features(self, df: pd.DataFrame, feature_type: str = 'raw') -> pd.DataFrame:
        """
        Prepare features for the baseline model.
        
        Args:
            df: Input DataFrame
            feature_type: 'raw' for baseline, 'engineered' for comparison
            
        Returns:
            Feature matrix
        """
        if feature_type == 'raw':
            # Use only raw meteorological and pollutant variables
            feature_cols = ['PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
            # Add temporal features
            if 'hour' in df.columns:
                feature_cols.extend(['hour', 'month', 'day_of_week'])
        else:
            # Use all columns except target and metadata
            exclude_cols = [TARGET_VARIABLE, 'station', 'datetime'] if 'datetime' in df.columns else [TARGET_VARIABLE, 'station']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Select available features
        available_features = [col for col in feature_cols if col in df.columns]
        
        if not available_features:
            raise ValueError("No valid features found for modeling")
        
        X = df[available_features].copy()
        
        # Handle any remaining missing values
        X = X.fillna(X.median())
        
        self.feature_names = available_features
        return X
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Train the baseline Linear Regression model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            
        Returns:
            Training metrics
        """
        logger.info("Training baseline Linear Regression model")
        
        # Create pipeline with scaling and linear regression
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression(**LINEAR_REGRESSION_PARAMS))
        ])
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Training metrics
        train_pred = self.model.predict(X_train)
        self.metrics['train_rmse'] = np.sqrt(mean_squared_error(y_train, train_pred))
        self.metrics['train_mae'] = mean_absolute_error(y_train, train_pred)
        self.metrics['train_r2'] = r2_score(y_train, train_pred)
        
        # Validation metrics if provided
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            self.metrics['val_rmse'] = np.sqrt(mean_squared_error(y_val, val_pred))
            self.metrics['val_mae'] = mean_absolute_error(y_val, val_pred)
            self.metrics['val_r2'] = r2_score(y_val, val_pred)
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=CV_N_SPLITS)
        cv_scores = cross_val_score(
            self.model, X_train, y_train, 
            cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=N_JOBS
        )
        self.metrics['cv_rmse_mean'] = -cv_scores.mean()
        self.metrics['cv_rmse_std'] = cv_scores.std()
        
        logger.info(f"Training completed. Train RMSE: {self.metrics['train_rmse']:.2f}")
        if 'val_rmse' in self.metrics:
            logger.info(f"Validation RMSE: {self.metrics['val_rmse']:.2f}")
        
        return self.metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.predict(X_test)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        # Custom metric: percentage within 10 μg/m³
        within_10 = np.abs(y_test - y_pred) <= 10
        metrics['pct_within_10'] = np.mean(within_10) * 100
        
        logger.info(f"Test RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.3f}")
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'training_params': self.training_params
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model from disk."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.metrics = model_data['metrics']
        self.training_params = model_data['training_params']
        self.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance (coefficients) from linear regression."""
        if not self.is_fitted:
            raise ValueError("Model must be trained to get feature importance")
        
        # Get coefficients from the linear regression model
        coefficients = self.model.named_steps['regressor'].coef_
        
        # Create series with feature names
        importance = pd.Series(np.abs(coefficients), index=self.feature_names)
        importance = importance.sort_values(ascending=False)
        
        return importance


class OptimizedModel:
    """
    Optimized XGBoost model with hyperparameter tuning using Optuna.
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.best_params = None
        self.training_params = None
        self.metrics = {}
        self.is_fitted = False
        self.study = None
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for the optimized model.
        
        Args:
            df: Input DataFrame with engineered features
            
        Returns:
            Feature matrix
        """
        # Exclude target and metadata columns
        exclude_cols = [TARGET_VARIABLE, 'station']
        if 'datetime' in df.columns:
            exclude_cols.append('datetime')
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        if not feature_cols:
            raise ValueError("No valid features found for modeling")
        
        X = df[feature_cols].copy()
        
        # Handle missing values separately for numeric and categorical columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns
        
        # Fill numeric columns with median
        if len(numeric_cols) > 0:
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        
        # Encode categorical columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'unknown')
                # Simple label encoding for categorical variables
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        self.feature_names = feature_cols
        return X
    
    def objective(self, trial, X_train: pd.DataFrame, y_train: pd.Series, 
                  X_val: pd.DataFrame, y_val: pd.Series) -> float:
        """
        Optuna objective function for hyperparameter optimization.
        """
        # Suggest hyperparameters
        params = XGBOOST_BASE_PARAMS.copy()
        
        for param, config in XGBOOST_SEARCH_SPACE.items():
            if config['type'] == 'int':
                params[param] = trial.suggest_int(param, config['low'], config['high'])
            elif config['type'] == 'float':
                if config.get('log', False):
                    params[param] = trial.suggest_float(param, config['low'], config['high'], log=True)
                else:
                    params[param] = trial.suggest_float(param, config['low'], config['high'])
        
        # Create XGBoost model
        model = xgb.XGBRegressor(**params)
        
        # Train model
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Validate
        val_pred = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        
        return val_rmse
    
    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                                X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Best parameters and study results
        """
        logger.info("Starting hyperparameter optimization with Optuna")
        
        # Create Optuna study
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
        )
        
        # Optimize
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_val, y_val),
            n_trials=OPTUNA_N_TRIALS,
            timeout=OPTUNA_TIMEOUT,
            n_jobs=OPTUNA_N_JOBS
        )
        
        self.study = study
        self.best_params = study.best_params
        
        logger.info(f"Optimization completed. Best RMSE: {study.best_value:.2f}")
        logger.info(f"Best parameters: {study.best_params}")
        
        return study.best_params
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame, y_val: pd.Series,
              optimize: bool = True) -> Dict[str, Any]:
        """
        Train the optimized XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            optimize: Whether to perform hyperparameter optimization
            
        Returns:
            Training metrics
        """
        logger.info("Training optimized XGBoost model")
        
        # Hyperparameter optimization if requested
        if optimize:
            best_params = self.optimize_hyperparameters(X_train, y_train, X_val, y_val)
        else:
            best_params = XGBOOST_BASE_PARAMS
        
        # Create final model with best parameters
        self.model = xgb.XGBRegressor(**best_params)
        
        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        self.is_fitted = True
        
        # Training metrics
        train_pred = self.model.predict(X_train)
        self.metrics['train_rmse'] = np.sqrt(mean_squared_error(y_train, train_pred))
        self.metrics['train_mae'] = mean_absolute_error(y_train, train_pred)
        self.metrics['train_r2'] = r2_score(y_train, train_pred)
        
        # Validation metrics
        val_pred = self.model.predict(X_val)
        self.metrics['val_rmse'] = np.sqrt(mean_squared_error(y_val, val_pred))
        self.metrics['val_mae'] = mean_absolute_error(y_val, val_pred)
        self.metrics['val_r2'] = r2_score(y_val, val_pred)
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=CV_N_SPLITS)
        cv_scores = cross_val_score(
            self.model, X_train, y_train,
            cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=N_JOBS
        )
        self.metrics['cv_rmse_mean'] = -cv_scores.mean()
        self.metrics['cv_rmse_std'] = cv_scores.std()
        
        logger.info(f"Training completed. Train RMSE: {self.metrics['train_rmse']:.2f}")
        logger.info(f"Validation RMSE: {self.metrics['val_rmse']:.2f}")
        
        return self.metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance on test data."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.predict(X_test)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        # Custom metric: percentage within 10 μg/m³
        within_10 = np.abs(y_test - y_pred) <= 10
        metrics['pct_within_10'] = np.mean(within_10) * 100
        
        logger.info(f"Test RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.3f}")
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'best_params': self.best_params,
            'metrics': self.metrics,
            'training_params': self.training_params
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model from disk."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.best_params = model_data['best_params']
        self.metrics = model_data['metrics']
        self.training_params = model_data['training_params']
        self.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance from XGBoost model."""
        if not self.is_fitted:
            raise ValueError("Model must be trained to get feature importance")
        
        importance = self.model.feature_importances_
        importance_series = pd.Series(importance, index=self.feature_names)
        importance_series = importance_series.sort_values(ascending=False)
        
        return importance_series
    
    def get_shap_values(self, X: pd.DataFrame, max_samples: int = 1000) -> Tuple[np.ndarray, pd.DataFrame]:
        """Calculate SHAP values for model interpretability."""
        try:
            import shap
            
            # Create SHAP explainer
            explainer = shap.TreeExplainer(self.model)
            
            # Sample data if too large
            if len(X) > max_samples:
                X_sample = X.sample(n=max_samples, random_state=RANDOM_STATE)
            else:
                X_sample = X
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_sample)
            
            return shap_values, X_sample
            
        except ImportError:
            logger.warning("SHAP not available. Install with: pip install shap")
            return None, None


def compare_models(baseline_model: BaselineModel, optimized_model: OptimizedModel,
                  X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Compare performance of baseline and optimized models.
    
    Args:
        baseline_model: Trained baseline model
        optimized_model: Trained optimized model
        X_test: Test features
        y_test: Test target
        
    Returns:
        Comparison DataFrame
    """
    baseline_metrics = baseline_model.evaluate(X_test, y_test)
    optimized_metrics = optimized_model.evaluate(X_test, y_test)
    
    comparison = pd.DataFrame({
        'Baseline (Linear Regression)': baseline_metrics,
        'Optimized (XGBoost)': optimized_metrics
    })
    
    # Add improvement percentage
    improvement = {}
    for metric in baseline_metrics.keys():
        if metric in ['rmse', 'mae', 'mape']:  # Lower is better
            improvement[metric] = ((baseline_metrics[metric] - optimized_metrics[metric]) / baseline_metrics[metric]) * 100
        else:  # Higher is better
            improvement[metric] = ((optimized_metrics[metric] - baseline_metrics[metric]) / baseline_metrics[metric]) * 100
    
    comparison['Improvement (%)'] = pd.Series(improvement)
    
    return comparison


def main():
    """Main function for testing the models."""
    
    from pathlib import Path
    
    # Load engineered data
    train_file = Path("../data/cleaned/train_engineered_optimized.csv")
    val_file = Path("../data/cleaned/val_engineered_optimized.csv")
    test_file = Path("../data/cleaned/test_engineered_optimized.csv")
    
    if not all([train_file.exists(), val_file.exists(), test_file.exists()]):
        print("Engineered data files not found. Please run feature engineering first.")
        return
    
    # Load data
    train_df = pd.read_csv(train_file, index_col=0, parse_dates=True)
    val_df = pd.read_csv(val_file, index_col=0, parse_dates=True)
    test_df = pd.read_csv(test_file, index_col=0, parse_dates=True)
    
    print(f"Data loaded - Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
    
    # Prepare features and targets
    baseline_model = BaselineModel()
    optimized_model = OptimizedModel()
    
    # Baseline model with raw features
    X_train_baseline = baseline_model.prepare_features(train_df, feature_type='raw')
    X_val_baseline = baseline_model.prepare_features(val_df, feature_type='raw')
    X_test_baseline = baseline_model.prepare_features(test_df, feature_type='raw')
    
    y_train = train_df[TARGET_VARIABLE]
    y_val = val_df[TARGET_VARIABLE]
    y_test = test_df[TARGET_VARIABLE]
    
    # Train baseline model
    print("\nTraining baseline model...")
    baseline_metrics = baseline_model.train(X_train_baseline, y_train, X_val_baseline, y_val)
    
    # Optimized model with engineered features
    X_train_opt = optimized_model.prepare_features(train_df)
    X_val_opt = optimized_model.prepare_features(val_df)
    X_test_opt = optimized_model.prepare_features(test_df)
    
    # Train optimized model
    print("\nTraining optimized model...")
    optimized_metrics = optimized_model.train(X_train_opt, y_train, X_val_opt, y_val, optimize=True)
    
    # Compare models
    print("\nComparing models...")
    comparison = compare_models(baseline_model, optimized_model, X_test_opt, y_test)
    print("\nModel Comparison:")
    print(comparison.round(3))
    
    # Save models
    if SAVE_MODELS:
        baseline_model.save_model(str(RESULTS_DIR / 'baseline_model.pkl'))
        optimized_model.save_model(str(RESULTS_DIR / 'optimized_model.pkl'))
        
        # Save best parameters
        import json
        with open(RESULTS_DIR / 'best_params.json', 'w') as f:
            json.dump(optimized_model.best_params, f, indent=2)
    
    print("\nModel development completed!")


if __name__ == "__main__":
    main()
# Final Project

# Final Project Module

