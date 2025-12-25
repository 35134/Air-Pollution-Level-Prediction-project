"""
Optimized feature engineering with better handling of edge cases and NaN values.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

from config import (
    TARGET_VARIABLE, POLLUTANT_COLUMNS, METEOROLOGICAL_COLUMNS,
    WIND_DIRECTION_COLUMN, STATION_COLUMN, MAX_LAG_HOURS, ROLLING_WINDOW_SIZE
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizedFeatureEngineer:
    """Optimized feature engineering with better NaN handling."""
    
    def __init__(self):
        self.feature_list = []
        self.engineering_steps = []
        
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features safely."""
        logger.info("Creating temporal features")
        
        # Hour of day (cyclical encoding)
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        
        # Day of week (cyclical encoding)
        df['day_of_week'] = df.index.dayofweek
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Month of year (cyclical encoding)
        df['month'] = df.index.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Season indicators
        df['season'] = df['month'].map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'autumn', 10: 'autumn', 11: 'autumn'
        })
        
        # One-hot encode season
        season_dummies = pd.get_dummies(df['season'], prefix='season')
        df = pd.concat([df, season_dummies], axis=1)
        
        # Weekend/weekday indicator
        df['is_weekend'] = (df['day_of_week'].isin([5, 6])).astype(int)
        
        # Time of day categories
        df['time_of_day'] = pd.cut(df.index.hour, 
                                  bins=[0, 6, 12, 18, 24], 
                                  labels=['night', 'morning', 'afternoon', 'evening'],
                                  include_lowest=True)
        
        # One-hot encode time of day
        time_dummies = pd.get_dummies(df['time_of_day'], prefix='time')
        df = pd.concat([df, time_dummies], axis=1)
        
        self.feature_list.extend([
            'hour_sin', 'hour_cos', 'day_of_week', 'dow_sin', 'dow_cos',
            'month_sin', 'month_cos', 'is_weekend'
        ])
        
        # Add dummy variables to feature list
        self.feature_list.extend([col for col in season_dummies.columns])
        self.feature_list.extend([col for col in time_dummies.columns])
        
        return df
    
    def create_lag_features_safe(self, df: pd.DataFrame, target_col: str = TARGET_VARIABLE) -> pd.DataFrame:
        """Create lag features with safe handling of edge cases."""
        logger.info("Creating lag features safely")
        
        # Only create lag features for data that has sufficient history
        # Start with shorter lags and gradually increase
        
        # Essential lag features (1, 6, 12, 24 hours)
        essential_lags = [1, 6, 12, 24]
        
        for lag in essential_lags:
            if lag <= MAX_LAG_HOURS:
                df[f'{target_col}_lag_{lag}h'] = df[target_col].shift(lag)
                self.feature_list.append(f'{target_col}_lag_{lag}h')
        
        # Lag features for other pollutants (only 1 and 6 hours to avoid overfitting)
        pollutant_cols = [col for col in POLLUTANT_COLUMNS if col in df.columns and col != target_col]
        
        for col in pollutant_cols:
            for lag in [1, 6]:
                df[f'{col}_lag_{lag}h'] = df[col].shift(lag)
                self.feature_list.append(f'{col}_lag_{lag}h')
        
        # Lag features for key meteorological variables
        key_met_vars = ['TEMP', 'PRES', 'WSPM']
        meteo_cols = [col for col in key_met_vars if col in df.columns]
        
        for col in meteo_cols:
            for lag in [1, 6]:
                df[f'{col}_lag_{lag}h'] = df[col].shift(lag)
                self.feature_list.append(f'{col}_lag_{lag}h')
        
        return df
    
    def create_rolling_features_safe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window statistics safely."""
        logger.info("Creating rolling window features safely")
        
        # Rolling statistics for target variable (shorter windows to preserve data)
        windows = [6, 12, 24]  # Reduced windows
        
        for window in windows:
            # Mean (most important)
            df[f'{TARGET_VARIABLE}_rolling_mean_{window}h'] = df[TARGET_VARIABLE].rolling(window=window, min_periods=window//2).mean()
            self.feature_list.append(f'{TARGET_VARIABLE}_rolling_mean_{window}h')
            
            # Standard deviation (only for larger windows)
            if window >= 12:
                df[f'{TARGET_VARIABLE}_rolling_std_{window}h'] = df[TARGET_VARIABLE].rolling(window=window, min_periods=window//2).std()
                self.feature_list.append(f'{TARGET_VARIABLE}_rolling_std_{window}h')
        
        # Rolling statistics for other pollutants (24h only)
        pollutant_cols = [col for col in POLLUTANT_COLUMNS if col in df.columns and col != TARGET_VARIABLE]
        
        for col in pollutant_cols:
            df[f'{col}_rolling_mean_24h'] = df[col].rolling(window=24, min_periods=12).mean()
            self.feature_list.append(f'{col}_rolling_mean_24h')
        
        # Rolling statistics for key meteorological variables
        meteo_cols = [col for col in ['TEMP', 'PRES'] if col in df.columns]
        
        for col in meteo_cols:
            df[f'{col}_rolling_mean_24h'] = df[col].rolling(window=24, min_periods=12).mean()
            self.feature_list.append(f'{col}_rolling_mean_24h')
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features."""
        logger.info("Creating interaction features")
        
        # Temperature-Humidity interaction
        if 'TEMP' in df.columns and 'DEWP' in df.columns:
            df['temp_dewp_interaction'] = df['TEMP'] * df['DEWP']
            self.feature_list.append('temp_dewp_interaction')
        
        # Wind-Pollution interactions (only for target variable)
        if 'WSPM' in df.columns:
            df[f'{TARGET_VARIABLE}_wind_interaction'] = df[TARGET_VARIABLE] * df['WSPM']
            self.feature_list.append(f'{TARGET_VARIABLE}_wind_interaction')
        
        # Pressure-Temperature interaction
        if 'PRES' in df.columns and 'TEMP' in df.columns:
            df['pres_temp_interaction'] = df['PRES'] * df['TEMP']
            self.feature_list.append('pres_temp_interaction')
        
        return df
    
    def create_station_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create station-specific features."""
        logger.info("Creating station features")
        
        if STATION_COLUMN not in df.columns:
            logger.warning("Station column not found, skipping station features")
            return df
        
        # One-hot encode station
        station_dummies = pd.get_dummies(df[STATION_COLUMN], prefix='station')
        df = pd.concat([df, station_dummies], axis=1)
        
        # Add station dummy columns to feature list
        self.feature_list.extend([col for col in station_dummies.columns])
        
        return df
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features."""
        logger.info("Creating advanced features")
        
        # PM ratios
        if 'PM2.5' in df.columns and 'PM10' in df.columns:
            df['pm25_pm10_ratio'] = df['PM2.5'] / (df['PM10'] + 1e-8)
            self.feature_list.append('pm25_pm10_ratio')
        
        # Temperature change rate
        if 'TEMP_lag_1h' in df.columns:
            df['temp_change_rate'] = df['TEMP'] - df['TEMP_lag_1h']
            self.feature_list.append('temp_change_rate')
        
        # Wind direction encoding
        if WIND_DIRECTION_COLUMN in df.columns:
            wind_direction_map = {
                'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
                'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
                'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
                'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
            }
            
            df['wind_direction_numeric'] = df[WIND_DIRECTION_COLUMN].map(wind_direction_map)
            df['wind_dir_sin'] = np.sin(np.radians(df['wind_direction_numeric']))
            df['wind_dir_cos'] = np.cos(np.radians(df['wind_direction_numeric']))
            
            self.feature_list.extend(['wind_dir_sin', 'wind_dir_cos'])
        
        return df
    
    def clean_data_after_feature_creation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data after feature creation, removing only problematic NaN values."""
        logger.info("Cleaning data after feature creation")
        
        initial_rows = len(df)
        
        # Only remove rows where critical features are missing
        critical_features = [
            TARGET_VARIABLE,
            f'{TARGET_VARIABLE}_lag_1h',
            f'{TARGET_VARIABLE}_lag_6h',
            f'{TARGET_VARIABLE}_lag_12h',
            f'{TARGET_VARIABLE}_lag_24h'
        ]
        
        # Only check for existing columns
        existing_critical_features = [f for f in critical_features if f in df.columns]
        
        if existing_critical_features:
            # Count NaN values in critical features
            nan_counts = df[existing_critical_features].isnull().sum()
            logger.info(f"NaN counts in critical features: {nan_counts.to_dict()}")
            
            # Remove rows with any NaN in critical features
            df_cleaned = df.dropna(subset=existing_critical_features)
            
            final_rows = len(df_cleaned)
            rows_removed = initial_rows - final_rows
            
            logger.info(f"Removed {rows_removed} rows ({rows_removed/initial_rows*100:.1f}%) with critical missing values")
            logger.info(f"Final dataset shape: {df_cleaned.shape}")
            
            return df_cleaned
        
        return df
    
    def engineer_features_optimized(self, df: pd.DataFrame, target_col: str = TARGET_VARIABLE,
                                   create_station_features: bool = True) -> pd.DataFrame:
        """Optimized feature engineering pipeline."""
        logger.info("Starting optimized feature engineering pipeline")
        
        initial_shape = df.shape
        
        # Create temporal features
        df = self.create_temporal_features(df)
        
        # Create lag features (safely)
        df = self.create_lag_features_safe(df, target_col)
        
        # Create rolling features (safely)
        df = self.create_rolling_features_safe(df)
        
        # Create interaction features
        df = self.create_interaction_features(df)
        
        # Create station features
        if create_station_features and STATION_COLUMN in df.columns:
            df = self.create_station_features(df)
        
        # Create advanced features
        df = self.create_advanced_features(df)
        
        # Clean data (remove only critical NaN values)
        df = self.clean_data_after_feature_creation(df)
        
        logger.info("Optimized feature engineering pipeline completed")
        logger.info(f"Initial shape: {initial_shape}, Final shape: {df.shape}")
        logger.info(f"Features created: {len(self.feature_list)}")
        
        return df


def main():
    """Main function for testing the optimized feature engineering."""
    
    from pathlib import Path
    
    # Load cleaned training data
    train_file = Path("data/cleaned/train_cleaned.csv")
    
    if not train_file.exists():
        print(f"Cleaned training data not found: {train_file}")
        return
    
    # Load data
    train_df = pd.read_csv(train_file, index_col=0, parse_dates=True)
    print(f"Loaded training data: {train_df.shape}")
    
    # Initialize feature engineer
    engineer = OptimizedFeatureEngineer()
    
    # Engineer features
    train_engineered = engineer.engineer_features_optimized(train_df)
    
    print(f"\nOptimized feature engineering completed!")
    print(f"Original shape: {train_df.shape}")
    print(f"Engineered shape: {train_engineered.shape}")
    print(f"Features created: {len(engineer.feature_list)}")
    
    # Get feature importance
    feature_importance = engineer.get_feature_importance_ranking(train_engineered)
    
    print(f"\nTop 15 most important features:")
    for i, (feature, importance) in enumerate(feature_importance.head(15).items(), 1):
        print(f"{i:2d}. {feature}: {importance:.3f}")
    
    # Save engineered data
    output_file = Path("data/cleaned/train_engineered_optimized.csv")
    train_engineered.to_csv(output_file)
    print(f"\nOptimized engineered data saved to: {output_file}")
    
    # Process validation and test data with the same approach
    print("\nProcessing validation and test data...")
    
    for dataset in ['val', 'test']:
        data_file = Path(f"data/cleaned/{dataset}_cleaned.csv")
        if data_file.exists():
            df = pd.read_csv(data_file, index_col=0, parse_dates=True)
            df_engineered = engineer.engineer_features_optimized(df)
            
            output_file = Path(f"data/cleaned/{dataset}_engineered_optimized.csv")
            df_engineered.to_csv(output_file)
            print(f"Processed {dataset} data: {df.shape} -> {df_engineered.shape}")
    
    print("\nAll optimized engineered data saved!")


if __name__ == "__main__":
    main()
# Final Project
