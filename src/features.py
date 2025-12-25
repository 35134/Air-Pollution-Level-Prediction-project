"""
Feature engineering functions for the Air Pollution Prediction project.

This module creates:
1. Temporal features (cyclical encoding, season indicators)
2. Lag features (historical values)
3. Rolling window statistics
4. Interaction terms
5. Station-specific features
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import logging

from config import (
    TARGET_VARIABLE, FEATURE_COLUMNS, POLLUTANT_COLUMNS, METEOROLOGICAL_COLUMNS,
    WIND_DIRECTION_COLUMN, STATION_COLUMN, FORECAST_HORIZON, MIN_LAG_HOURS,
    MAX_LAG_HOURS, ROLLING_WINDOW_SIZE, DATE_TIME_COLUMNS
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Comprehensive feature engineering for air quality time series data.
    """
    
    def __init__(self):
        self.feature_list = []
        self.engineering_steps = []
        
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features from datetime index.
        
        Features created:
        - Hour of day (cyclical encoding)
        - Day of week (cyclical encoding)
        - Month of year (cyclical encoding)
        - Season indicators
        - Weekend/weekday indicator
        - Holiday indicators (basic implementation)
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            DataFrame with temporal features
        """
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
        
        # Basic holiday indicators (weekends + some fixed holidays)
        # Note: This is a simplified implementation
        df['is_holiday'] = df['is_weekend'].copy()
        
        # Add some fixed date holidays (approximate)
        holiday_dates = [
            (1, 1),   # New Year
            (2, 14),  # Valentine's Day (for demonstration)
            (5, 1),   # Labor Day
            (10, 1),  # National Day
            (12, 25)  # Christmas
        ]
        
        for month, day in holiday_dates:
            holiday_mask = (df.index.month == month) & (df.index.day == day)
            df.loc[holiday_mask, 'is_holiday'] = 1
        
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
            'month_sin', 'month_cos', 'season', 'is_weekend', 'is_holiday',
            'time_of_day'
        ])
        
        # Add dummy variables to feature list
        self.feature_list.extend([col for col in season_dummies.columns])
        self.feature_list.extend([col for col in time_dummies.columns])
        
        self.engineering_steps.append("Created temporal features (cyclical encoding, seasons, weekends)")
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = TARGET_VARIABLE) -> pd.DataFrame:
        """
        Create lag features for the target variable and other pollutants.
        
        Args:
            df: DataFrame with time series data
            target_col: Target variable column name
            
        Returns:
            DataFrame with lag features
        """
        logger.info("Creating lag features")
        
        # Lag features for target variable
        for lag in range(MIN_LAG_HOURS, MAX_LAG_HOURS + 1):
            df[f'{target_col}_lag_{lag}h'] = df[target_col].shift(lag)
            self.feature_list.append(f'{target_col}_lag_{lag}h')
        
        # Lag features for other pollutants (shorter lags to avoid overfitting)
        pollutant_cols = [col for col in POLLUTANT_COLUMNS if col in df.columns and col != target_col]
        
        for col in pollutant_cols:
            for lag in [1, 3, 6, 12, 24]:
                df[f'{col}_lag_{lag}h'] = df[col].shift(lag)
                self.feature_list.append(f'{col}_lag_{lag}h')
        
        # Lag features for meteorological variables
        meteo_cols = [col for col in METEOROLOGICAL_COLUMNS if col in df.columns]
        
        for col in meteo_cols:
            for lag in [1, 6, 24]:
                df[f'{col}_lag_{lag}h'] = df[col].shift(lag)
                self.feature_list.append(f'{col}_lag_{lag}h')
        
        self.engineering_steps.append(f"Created lag features (1-{MAX_LAG_HOURS}h for target, various lags for other variables)")
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling window statistics.
        
        Args:
            df: DataFrame with time series data
            
        Returns:
            DataFrame with rolling features
        """
        logger.info("Creating rolling window features")
        
        # Rolling statistics for target variable
        windows = [6, 12, 24, 48]  # 6 hours, 12 hours, 1 day, 2 days
        
        for window in windows:
            # Mean
            df[f'{TARGET_VARIABLE}_rolling_mean_{window}h'] = df[TARGET_VARIABLE].rolling(window=window, min_periods=1).mean()
            self.feature_list.append(f'{TARGET_VARIABLE}_rolling_mean_{window}h')
            
            # Standard deviation
            df[f'{TARGET_VARIABLE}_rolling_std_{window}h'] = df[TARGET_VARIABLE].rolling(window=window, min_periods=1).std()
            self.feature_list.append(f'{TARGET_VARIABLE}_rolling_std_{window}h')
            
            # Maximum
            df[f'{TARGET_VARIABLE}_rolling_max_{window}h'] = df[TARGET_VARIABLE].rolling(window=window, min_periods=1).max()
            self.feature_list.append(f'{TARGET_VARIABLE}_rolling_max_{window}h')
            
            # Minimum
            df[f'{TARGET_VARIABLE}_rolling_min_{window}h'] = df[TARGET_VARIABLE].rolling(window=window, min_periods=1).min()
            self.feature_list.append(f'{TARGET_VARIABLE}_rolling_min_{window}h')
        
        # Rolling statistics for other pollutants
        pollutant_cols = [col for col in POLLUTANT_COLUMNS if col in df.columns]
        
        for col in pollutant_cols:
            for window in [12, 24]:
                # Mean
                df[f'{col}_rolling_mean_{window}h'] = df[col].rolling(window=window, min_periods=1).mean()
                self.feature_list.append(f'{col}_rolling_mean_{window}h')
                
                # Standard deviation
                df[f'{col}_rolling_std_{window}h'] = df[col].rolling(window=window, min_periods=1).std()
                self.feature_list.append(f'{col}_rolling_std_{window}h')
        
        # Rolling statistics for meteorological variables
        meteo_cols = [col for col in METEOROLOGICAL_COLUMNS if col in df.columns]
        
        for col in meteo_cols:
            for window in [6, 24]:
                # Mean
                df[f'{col}_rolling_mean_{window}h'] = df[col].rolling(window=window, min_periods=1).mean()
                self.feature_list.append(f'{col}_rolling_mean_{window}h')
                
                # Standard deviation (for variables that make sense)
                if col in ['TEMP', 'PRES', 'WSPM']:
                    df[f'{col}_rolling_std_{window}h'] = df[col].rolling(window=window, min_periods=1).std()
                    self.feature_list.append(f'{col}_rolling_std_{window}h')
        
        self.engineering_steps.append(f"Created rolling features (windows: {windows})")
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between variables.
        
        Args:
            df: DataFrame with base features
            
        Returns:
            DataFrame with interaction features
        """
        logger.info("Creating interaction features")
        
        # Temperature-Humidity interaction (heat index proxy)
        if 'TEMP' in df.columns and 'DEWP' in df.columns:
            df['temp_dewp_interaction'] = df['TEMP'] * df['DEWP']
            self.feature_list.append('temp_dewp_interaction')
        
        # Wind-Pollution interactions
        if 'WSPM' in df.columns:
            for col in POLLUTANT_COLUMNS:
                if col in df.columns:
                    df[f'{col}_wind_interaction'] = df[col] * df['WSPM']
                    self.feature_list.append(f'{col}_wind_interaction')
        
        # Pressure-Temperature interaction (weather system indicator)
        if 'PRES' in df.columns and 'TEMP' in df.columns:
            df['pres_temp_interaction'] = df['PRES'] * df['TEMP']
            self.feature_list.append('pres_temp_interaction')
        
        # Lag interactions
        if f'{TARGET_VARIABLE}_lag_1h' in df.columns and 'TEMP' in df.columns:
            df[f'{TARGET_VARIABLE}_temp_lag_interaction'] = df[f'{TARGET_VARIABLE}_lag_1h'] * df['TEMP']
            self.feature_list.append(f'{TARGET_VARIABLE}_temp_lag_interaction')
        
        # Seasonal interactions
        season_cols = [col for col in df.columns if col.startswith('season_')]
        for season_col in season_cols:
            if 'TEMP' in df.columns:
                df[f'{season_col}_temp_interaction'] = df[season_col] * df['TEMP']
                self.feature_list.append(f'{season_col}_temp_interaction')
        
        self.engineering_steps.append("Created interaction features (temp-humidity, wind-pollution, pressure-temp)")
        
        return df
    
    def create_station_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create station-specific features.
        
        Args:
            df: DataFrame with station information
            
        Returns:
            DataFrame with station features
        """
        logger.info("Creating station features")
        
        if STATION_COLUMN not in df.columns:
            logger.warning("Station column not found, skipping station features")
            return df
        
        # One-hot encode station
        station_dummies = pd.get_dummies(df[STATION_COLUMN], prefix='station')
        df = pd.concat([df, station_dummies], axis=1)
        
        # Add station dummy columns to feature list
        self.feature_list.extend([col for col in station_dummies.columns])
        
        # Create station-specific statistics (if we have enough data)
        stations = df[STATION_COLUMN].unique()
        
        for station in stations:
            station_mask = df[STATION_COLUMN] == station
            
            # Station-specific mean pollution level
            if TARGET_VARIABLE in df.columns:
                station_mean = df.loc[station_mask, TARGET_VARIABLE].mean()
                df.loc[station_mask, f'{station}_mean_{TARGET_VARIABLE}'] = station_mean
                self.feature_list.append(f'{station}_mean_{TARGET_VARIABLE}')
        
        self.engineering_steps.append("Created station features (one-hot encoding, station-specific statistics)")
        
        return df
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced features like trend, change rates, and ratios.
        
        Args:
            df: DataFrame with base features
            
        Returns:
            DataFrame with advanced features
        """
        logger.info("Creating advanced features")
        
        # Change rates (hour-over-hour)
        if f'{TARGET_VARIABLE}_lag_1h' in df.columns:
            df[f'{TARGET_VARIABLE}_change_rate'] = (
                df[TARGET_VARIABLE] - df[f'{TARGET_VARIABLE}_lag_1h']
            ) / (df[f'{TARGET_VARIABLE}_lag_1h'] + 1e-8)  # Add small epsilon to avoid division by zero
            self.feature_list.append(f'{TARGET_VARIABLE}_change_rate')
        
        # PM ratios (PM2.5/PM10 ratio is important for air quality)
        if 'PM2.5' in df.columns and 'PM10' in df.columns:
            df['pm25_pm10_ratio'] = df['PM2.5'] / (df['PM10'] + 1e-8)
            self.feature_list.append('pm25_pm10_ratio')
        
        # Temperature change rate
        if 'TEMP_lag_1h' in df.columns:
            df['temp_change_rate'] = df['TEMP'] - df['TEMP_lag_1h']
            self.feature_list.append('temp_change_rate')
        
        # Pressure trend (important for weather prediction)
        if 'PRES_lag_6h' in df.columns:
            df['pres_trend'] = df['PRES'] - df['PRES_lag_6h']
            self.feature_list.append('pres_trend')
        
        # Wind direction encoding (convert to numeric for better ML performance)
        if WIND_DIRECTION_COLUMN in df.columns:
            # Create wind direction sine/cosine encoding
            wind_direction_map = {
                'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
                'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
                'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
                'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
            }
            
            df['wind_direction_numeric'] = df[WIND_DIRECTION_COLUMN].map(wind_direction_map)
            
            # Convert to radians and create cyclical features
            df['wind_dir_sin'] = np.sin(np.radians(df['wind_direction_numeric']))
            df['wind_dir_cos'] = np.cos(np.radians(df['wind_direction_numeric']))
            
            self.feature_list.extend(['wind_dir_sin', 'wind_dir_cos'])
        
        # Composite air quality index (simplified)
        pollutants_for_aqi = [col for col in ['PM2.5', 'PM10', 'NO2', 'SO2'] if col in df.columns]
        if len(pollutants_for_aqi) >= 2:
            # Normalize pollutants to 0-1 scale
            normalized_pollutants = []
            for col in pollutants_for_aqi:
                normalized = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-8)
                normalized_pollutants.append(normalized)
            
            df['composite_aqi'] = np.mean(normalized_pollutants, axis=0)
            self.feature_list.append('composite_aqi')
        
        self.engineering_steps.append("Created advanced features (change rates, ratios, wind encoding, composite AQI)")
        
        return df
    
    def select_features(self, df: pd.DataFrame, target_col: str = TARGET_VARIABLE, 
                       max_features: int = 50) -> List[str]:
        """
        Select most important features using correlation and domain knowledge.
        
        Args:
            df: DataFrame with all features
            target_col: Target variable name
            max_features: Maximum number of features to select
            
        Returns:
            List of selected feature names
        """
        logger.info("Selecting features")
        
        # Calculate correlation with target
        correlations = df.corr()[target_col].abs().sort_values(ascending=False)
        
        # Remove target variable itself
        correlations = correlations[correlations.index != target_col]
        
        # Select top correlated features
        top_corr_features = correlations.head(max_features).index.tolist()
        
        # Ensure we include domain-important features
        important_features = [
            # Target lags
            *[f'{target_col}_lag_{i}h' for i in [1, 6, 12, 24] if f'{target_col}_lag_{i}h' in df.columns],
            
            # Key meteorological variables
            'TEMP', 'PRES', 'WSPM', 'DEWP',
            
            # Temporal features
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'is_weekend',
            
            # Rolling statistics
            f'{target_col}_rolling_mean_24h', f'{target_col}_rolling_std_24h',
            
            # Interactions
            'temp_dewp_interaction', 'pm25_pm10_ratio'
        ]
        
        # Combine correlation-based and domain-important features
        selected_features = []
        
        # First, add domain-important features if they exist
        for feature in important_features:
            if feature in df.columns and feature not in selected_features:
                selected_features.append(feature)
        
        # Then add top correlated features until we reach max_features
        for feature in top_corr_features:
            if feature not in selected_features and len(selected_features) < max_features:
                selected_features.append(feature)
        
        logger.info(f"Selected {len(selected_features)} features out of {len(df.columns)}")
        
        return selected_features
    
    def engineer_features(self, df: pd.DataFrame, target_col: str = TARGET_VARIABLE,
                         create_station_features: bool = True) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Args:
            df: Base DataFrame
            target_col: Target variable name
            create_station_features: Whether to create station-specific features
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering pipeline")
        
        initial_shape = df.shape
        
        # Create temporal features
        df = self.create_temporal_features(df)
        
        # Create lag features
        df = self.create_lag_features(df, target_col)
        
        # Create rolling features
        df = self.create_rolling_features(df)
        
        # Create interaction features
        df = self.create_interaction_features(df)
        
        # Create station features (if requested and data available)
        if create_station_features and STATION_COLUMN in df.columns:
            df = self.create_station_features(df)
        
        # Create advanced features
        df = self.create_advanced_features(df)
        
        # Remove rows with NaN values only for the most critical features
        # Keep rows that have the target variable and key features
        initial_rows = len(df)
        
        # Only remove rows where the target variable or key lag features are missing
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
            df = df.dropna(subset=existing_critical_features)
        
        final_rows = len(df)
        
        logger.info(f"Removed {initial_rows - final_rows} rows with critical missing values")
        
        logger.info("Feature engineering pipeline completed")
        logger.info(f"Initial shape: {initial_shape}, Final shape: {df.shape}")
        logger.info(f"Features created: {len(self.feature_list)}")
        
        return df
    
    def get_feature_importance_ranking(self, df: pd.DataFrame, target_col: str = TARGET_VARIABLE) -> pd.Series:
        """
        Get feature importance ranking based on correlation with target.
        
        Args:
            df: DataFrame with features and target
            target_col: Target variable name
            
        Returns:
            Series with feature importance scores
        """
        logger.info("Calculating feature importance ranking")
        
        # Calculate absolute correlation with target
        correlations = df.corr()[target_col].abs()
        
        # Remove target variable itself
        correlations = correlations[correlations.index != target_col]
        
        # Sort by importance
        correlations = correlations.sort_values(ascending=False)
        
        return correlations
    
    def generate_feature_report(self) -> str:
        """
        Generate a comprehensive feature engineering report.
        
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("FEATURE ENGINEERING REPORT")
        report.append("=" * 60)
        report.append("")
        
        report.append("Engineering Steps Performed:")
        for i, step in enumerate(self.engineering_steps, 1):
            report.append(f"{i}. {step}")
        
        report.append(f"\nTotal Features Created: {len(self.feature_list)}")
        
        report.append("\nFeature Categories:")
        
        # Count features by category
        temporal_features = [f for f in self.feature_list if any(x in f for x in ['hour', 'day', 'month', 'season', 'weekend', 'holiday', 'time'])]
        lag_features = [f for f in self.feature_list if 'lag' in f]
        rolling_features = [f for f in self.feature_list if 'rolling' in f]
        interaction_features = [f for f in self.feature_list if 'interaction' in f]
        station_features = [f for f in self.feature_list if 'station' in f]
        advanced_features = [f for f in self.feature_list if any(x in f for x in ['change_rate', 'ratio', 'trend', 'wind_dir', 'composite'])]
        
        report.append(f"  - Temporal features: {len(temporal_features)}")
        report.append(f"  - Lag features: {len(lag_features)}")
        report.append(f"  - Rolling features: {len(rolling_features)}")
        report.append(f"  - Interaction features: {len(interaction_features)}")
        report.append(f"  - Station features: {len(station_features)}")
        report.append(f"  - Advanced features: {len(advanced_features)}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


def main():
    """Main function for testing the feature engineering pipeline"""
    
    from pathlib import Path
    
    # Load cleaned training data
    data_file = Path("data/cleaned/train_cleaned.csv")
    
    if not data_file.exists():
        print(f"Cleaned data file not found: {data_file}")
        print("Please run data preprocessing first.")
        return
    
    # Load data
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    
    print(f"Loaded training data: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Engineer features
    df_engineered = engineer.engineer_features(df)
    
    print(f"\nFeature engineering completed!")
    print(f"Original shape: {df.shape}")
    print(f"Engineered shape: {df_engineered.shape}")
    print(f"Features created: {len(engineer.feature_list)}")
    
    # Get feature importance
    feature_importance = engineer.get_feature_importance_ranking(df_engineered)
    
    print(f"\nTop 10 most important features:")
    for i, (feature, importance) in enumerate(feature_importance.head(10).items(), 1):
        print(f"{i:2d}. {feature}: {importance:.3f}")
    
    # Save engineered data
    output_file = Path("data/cleaned/train_engineered.csv")
    df_engineered.to_csv(output_file)
    print(f"\nEngineered data saved to: {output_file}")
    
    # Save feature report
    report_file = Path("data/cleaned/feature_engineering_report.txt")
    with open(report_file, 'w') as f:
        f.write(engineer.generate_feature_report())
    print(f"Feature engineering report saved to: {report_file}")


if __name__ == "__main__":
    main()
# Final Project
