"""
Configuration settings for the Air Pollution Prediction project.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEANED_DATA_DIR = DATA_DIR / "cleaned"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Create directories if they don't exist
for directory in [CLEANED_DATA_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data configuration
TARGET_VARIABLE = 'PM2.5'
STATION_COLUMN = 'station'
DATE_TIME_COLUMNS = ['year', 'month', 'day', 'hour']

# Pollutant columns (our main features)
POLLUTANT_COLUMNS = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

# Meteorological columns
METEOROLOGICAL_COLUMNS = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']

# Wind direction column (categorical)
WIND_DIRECTION_COLUMN = 'wd'

# All feature columns (excluding target and metadata)
FEATURE_COLUMNS = POLLUTANT_COLUMNS + METEOROLOGICAL_COLUMNS + [WIND_DIRECTION_COLUMN]

# Time series configuration
FORECAST_HORIZON = 24  # Predict 24 hours ahead
MIN_LAG_HOURS = 1
MAX_LAG_HOURS = 24  # Use up to 24 hours of historical data
ROLLING_WINDOW_SIZE = 24  # 24-hour rolling statistics

# Data quality thresholds
PM25_MIN = 0  # Minimum physically possible PM2.5
PM25_MAX = 500  # Maximum reasonable PM2.5 (based on data distribution)
PM10_MIN = 0
PM10_MAX = 1000
OUTLIER_IQR_MULTIPLIER = 1.5  # For IQR-based outlier detection

# Missing data handling
MISSING_DATA_THRESHOLD = 0.1  # 10% threshold for dropping features
FORWARD_FILL_LIMIT = 6  # Maximum hours to forward fill (6 hours)

# Train/validation/test split (temporal)
TRAIN_START = '2013-03-01'
TRAIN_END = '2015-12-31'
VAL_START = '2016-01-01' 
VAL_END = '2016-12-31'
TEST_START = '2017-01-01'
TEST_END = '2017-02-28'

# Model configuration
RANDOM_STATE = 42
N_JOBS = -1  # Use all available CPU cores

# Baseline model (Linear Regression)
LINEAR_REGRESSION_PARAMS = {
    'fit_intercept': True,
    'copy_X': True,
    'n_jobs': N_JOBS
}

# XGBoost model
XGBOOST_BASE_PARAMS = {
    'random_state': RANDOM_STATE,
    'n_jobs': N_JOBS,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
}

# Hyperparameter optimization
OPTUNA_N_TRIALS = 50
OPTUNA_TIMEOUT = 3600  # 1 hour timeout
OPTUNA_N_JOBS = 1  # Sequential optimization for reproducibility

# XGBoost hyperparameter search space
XGBOOST_SEARCH_SPACE = {
    'max_depth': {'type': 'int', 'low': 3, 'high': 10},
    'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
    'n_estimators': {'type': 'int', 'low': 100, 'high': 1000},
    'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
    'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
    'reg_alpha': {'type': 'float', 'low': 0.0, 'high': 1.0},
    'reg_lambda': {'type': 'float', 'low': 0.0, 'high': 1.0}
}

# Evaluation metrics
EVALUATION_METRICS = ['RMSE', 'MAE', 'R2', 'MAPE']

# Visualization settings
PLOT_STYLE = 'seaborn-v0_8'
FIGURE_SIZE = (12, 8)
DPI = 300

# Feature importance
N_TOP_FEATURES = 15  # Number of top features to display

# Cross-validation
CV_N_SPLITS = 5  # TimeSeriesSplit
CV_MAX_TRAIN_SIZE = None  # No limit on training size

# Early stopping
EARLY_STOPPING_ROUNDS = 50
EARLY_STOPPING_MIN_DELTA = 0.001

# Logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Results saving
SAVE_MODELS = True
SAVE_PREDICTIONS = True
SAVE_PLOTS = True

# Academic reporting
CONFIDENCE_LEVEL = 0.95  # 95% confidence intervals
SIGNIFICANCE_LEVEL = 0.05  # Statistical significance threshold
# Final Project
