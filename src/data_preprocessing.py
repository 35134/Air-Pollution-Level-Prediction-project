"""
Data preprocessing functions for the Air Pollution Prediction project.

This module handles:
1. Data cleaning (missing values, outliers, invalid data)
2. Temporal alignment and validation
3. Quality control and filtering
4. Basic feature validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import logging

from config import (
    TARGET_VARIABLE, FEATURE_COLUMNS, POLLUTANT_COLUMNS, METEOROLOGICAL_COLUMNS,
    DATE_TIME_COLUMNS, STATION_COLUMN, WIND_DIRECTION_COLUMN, PM25_MIN, PM25_MAX,
    PM10_MIN, PM10_MAX, OUTLIER_IQR_MULTIPLIER, FORWARD_FILL_LIMIT,
    MISSING_DATA_THRESHOLD, TRAIN_START, TRAIN_END, VAL_START, VAL_END,
    TEST_START, TEST_END, CLEANED_DATA_DIR
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Comprehensive data preprocessor for air quality time series data.
    """
    
    def __init__(self):
        self.data_quality_report = {}
        self.preprocessing_steps = []
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading data from {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def create_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create datetime index from year, month, day, hour columns.
        
        Args:
            df: DataFrame with temporal columns
            
        Returns:
            DataFrame with datetime index
        """
        logger.info("Creating datetime index")
        
        # Create datetime column
        df['datetime'] = pd.to_datetime(df[DATE_TIME_COLUMNS])
        
        # Sort by datetime
        df = df.sort_values('datetime')
        
        # Set datetime as index
        df = df.set_index('datetime')
        
        self.preprocessing_steps.append("Created datetime index")
        logger.info(f"Datetime index created. Range: {df.index.min()} to {df.index.max()}")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using appropriate strategies:
        - Forward fill for temporal data (limited to 6 hours)
        - Linear interpolation for meteorological variables
        - Drop rows with excessive missing data
        
        Args:
            df: DataFrame with missing values
            
        Returns:
            DataFrame with handled missing values
        """
        logger.info("Handling missing values")
        
        initial_missing = df.isnull().sum().sum()
        logger.info(f"Initial missing values: {initial_missing}")
        
        # Separate different types of variables
        pollutant_cols = [col for col in POLLUTANT_COLUMNS if col in df.columns]
        meteo_cols = [col for col in METEOROLOGICAL_COLUMNS if col in df.columns]
        
        # Handle missing values for each station separately to avoid cross-contamination
        stations = df[STATION_COLUMN].unique() if STATION_COLUMN in df.columns else [None]
        
        processed_dfs = []
        
        for station in stations:
            if station is not None:
                station_df = df[df[STATION_COLUMN] == station].copy()
            else:
                station_df = df.copy()
            
            # Forward fill for pollutants (temporal continuity)
            for col in pollutant_cols:
                if col in station_df.columns:
                    station_df[col] = station_df[col].ffill(limit=FORWARD_FILL_LIMIT)
            
            # Linear interpolation for meteorological variables
            for col in meteo_cols:
                if col in station_df.columns:
                    station_df[col] = station_df[col].interpolate(method='linear', limit_direction='both')
            
            # Handle wind direction separately (categorical)
            if WIND_DIRECTION_COLUMN in station_df.columns:
                # Forward fill wind direction
                station_df[WIND_DIRECTION_COLUMN] = station_df[WIND_DIRECTION_COLUMN].fillna(method='ffill', limit=FORWARD_FILL_LIMIT)
            
            processed_dfs.append(station_df)
        
        # Combine processed dataframes
        df = pd.concat(processed_dfs)
        
        # Drop rows with excessive missing data
        threshold = len(df.columns) * MISSING_DATA_THRESHOLD
        df = df.dropna(thresh=len(df.columns) - threshold)
        
        final_missing = df.isnull().sum().sum()
        logger.info(f"Final missing values: {final_missing}")
        logger.info(f"Missing values reduced by: {((initial_missing - final_missing) / initial_missing * 100):.1f}%")
        
        self.preprocessing_steps.append(f"Handled missing values (reduced by {((initial_missing - final_missing) / initial_missing * 100):.1f}%)")
        
        return df
    
    def detect_and_handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and handle outliers using IQR method and physical constraints.
        
        Args:
            df: DataFrame with potential outliers
            
        Returns:
            DataFrame with outliers handled
        """
        logger.info("Detecting and handling outliers")
        
        outlier_summary = {}
        
        # Physical constraints for pollutants
        physical_constraints = {
            'PM2.5': (PM25_MIN, PM25_MAX),
            'PM10': (PM10_MIN, PM10_MAX),
            'SO2': (0, 1000),  # μg/m³
            'NO2': (0, 1000),  # μg/m³
            'CO': (0, 100),    # mg/m³
            'O3': (0, 1000),   # μg/m³
            'TEMP': (-50, 60),  # °C
            'PRES': (900, 1100),  # hPa
            'DEWP': (-50, 40),  # °C
            'RAIN': (0, 200),   # mm
            'WSPM': (0, 50)     # m/s
        }
        
        for col, (min_val, max_val) in physical_constraints.items():
            if col in df.columns:
                # Count values outside physical constraints
                physical_outliers = ((df[col] < min_val) | (df[col] > max_val)).sum()
                
                if physical_outliers > 0:
                    logger.info(f"Found {physical_outliers} physical outliers in {col}")
                    # Cap values at physical limits
                    df[col] = df[col].clip(lower=min_val, upper=max_val)
                    outlier_summary[f"{col}_physical"] = physical_outliers
        
        # IQR method for additional outlier detection
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in df.columns and col not in [STATION_COLUMN]:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - OUTLIER_IQR_MULTIPLIER * IQR
                upper_bound = Q3 + OUTLIER_IQR_MULTIPLIER * IQR
                
                iqr_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                
                if iqr_outliers > 0:
                    logger.info(f"Found {iqr_outliers} IQR outliers in {col}")
                    # Cap extreme outliers (beyond 3*IQR)
                    extreme_lower = Q1 - 3 * IQR
                    extreme_upper = Q3 + 3 * IQR
                    
                    # Cap only extreme outliers
                    mask = (df[col] < extreme_lower) | (df[col] > extreme_upper)
                    df.loc[mask, col] = df.loc[mask, col].clip(lower=extreme_lower, upper=extreme_upper)
                    
                    outlier_summary[f"{col}_iqr"] = iqr_outliers
        
        self.data_quality_report['outlier_summary'] = outlier_summary
        self.preprocessing_steps.append(f"Handled outliers for {len(outlier_summary)} variable combinations")
        
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate timestamps within each station.
        
        Args:
            df: DataFrame with potential duplicates
            
        Returns:
            DataFrame with duplicates removed
        """
        logger.info("Removing duplicates")
        
        initial_count = len(df)
        
        if STATION_COLUMN in df.columns:
            # Remove duplicates within each station
            df = df.reset_index()
            df = df.drop_duplicates(subset=['datetime', STATION_COLUMN], keep='first')
            df = df.set_index('datetime')
        else:
            # Remove duplicates based on index
            df = df[~df.index.duplicated(keep='first')]
        
        final_count = len(df)
        duplicates_removed = initial_count - final_count
        
        logger.info(f"Removed {duplicates_removed} duplicate records")
        self.preprocessing_steps.append(f"Removed {duplicates_removed} duplicates")
        
        return df
    
    def validate_data_quality(self, df: pd.DataFrame) -> bool:
        """
        Validate data quality and completeness.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if data quality is acceptable
        """
        logger.info("Validating data quality")
        
        is_valid = True
        
        # Check for negative pollutant concentrations
        for col in POLLUTANT_COLUMNS:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    logger.warning(f"Found {negative_count} negative values in {col}")
                    is_valid = False
        
        # Check for excessive missing data
        missing_pct = (df.isnull().sum() / len(df)) * 100
        high_missing = missing_pct[missing_pct > 10]
        
        if len(high_missing) > 0:
            logger.warning(f"Variables with >10% missing data: {list(high_missing.index)}")
        
        # Check temporal continuity
        if len(df) > 0:
            expected_hours = (df.index.max() - df.index.min()).total_seconds() / 3600 + 1
            actual_hours = len(df)
            continuity_pct = (actual_hours / expected_hours) * 100
            
            logger.info(f"Temporal continuity: {continuity_pct:.1f}%")
            
            if continuity_pct < 80:
                logger.warning("Poor temporal continuity (<80%)")
        
        # Check data range
        if TARGET_VARIABLE in df.columns:
            target_range = df[TARGET_VARIABLE].max() - df[TARGET_VARIABLE].min()
            logger.info(f"Target variable range: {target_range:.1f}")
            
            if target_range < 10:
                logger.warning("Very small target variable range")
        
        self.data_quality_report['validation_passed'] = is_valid
        return is_valid
    
    def split_temporal_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data temporally into train, validation, and test sets.
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Splitting data temporally")
        
        train_df = df[df.index <= TRAIN_END]
        val_df = df[(df.index >= VAL_START) & (df.index <= VAL_END)]
        test_df = df[df.index >= TEST_START]
        
        logger.info(f"Train set: {len(train_df)} records ({train_df.index.min()} to {train_df.index.max()})")
        logger.info(f"Validation set: {len(val_df)} records ({val_df.index.min()} to {val_df.index.max()})")
        logger.info(f"Test set: {len(test_df)} records ({test_df.index.min()} to {test_df.index.max()})")
        
        self.preprocessing_steps.append(f"Temporal split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    def generate_preprocessing_report(self) -> str:
        """
        Generate a comprehensive preprocessing report.
        
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("DATA PREPROCESSING REPORT")
        report.append("=" * 60)
        report.append("")
        
        report.append("Preprocessing Steps Performed:")
        for i, step in enumerate(self.preprocessing_steps, 1):
            report.append(f"{i}. {step}")
        
        report.append("")
        report.append("Data Quality Summary:")
        
        if 'outlier_summary' in self.data_quality_report:
            report.append("\nOutlier Handling:")
            for key, value in self.data_quality_report['outlier_summary'].items():
                report.append(f"  - {key}: {value} outliers handled")
        
        if 'validation_passed' in self.data_quality_report:
            validation_status = "PASSED" if self.data_quality_report['validation_passed'] else "FAILED"
            report.append(f"\nData Validation: {validation_status}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def preprocess_pipeline(self, df: pd.DataFrame, save_cleaned: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Raw DataFrame
            save_cleaned: Whether to save cleaned data to file
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Starting preprocessing pipeline")
        
        # Store initial shape
        initial_shape = df.shape
        
        # Create datetime index
        df = self.create_datetime_index(df)
        
        # Remove duplicates
        df = self.remove_duplicates(df)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Handle outliers
        df = self.detect_and_handle_outliers(df)
        
        # Validate data quality
        is_valid = self.validate_data_quality(df)
        
        if not is_valid:
            logger.warning("Data validation failed - proceeding with caution")
        
        # Split data temporally
        train_df, val_df, test_df = self.split_temporal_data(df)
        
        # Save cleaned data if requested
        if save_cleaned:
            CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            
            train_df.to_csv(CLEANED_DATA_DIR / "train_cleaned.csv")
            val_df.to_csv(CLEANED_DATA_DIR / "val_cleaned.csv")
            test_df.to_csv(CLEANED_DATA_DIR / "test_cleaned.csv")
            
            logger.info("Cleaned data saved to data/cleaned/")
        
        # Generate and save preprocessing report
        report = self.generate_preprocessing_report()
        
        with open(CLEANED_DATA_DIR / "preprocessing_report.txt", 'w') as f:
            f.write(report)
        
        logger.info("Preprocessing pipeline completed")
        logger.info(f"Initial shape: {initial_shape}, Final shape: {df.shape}")
        logger.info(f"Data reduction: {((initial_shape[0] - df.shape[0]) / initial_shape[0] * 100):.1f}%")
        
        return train_df, val_df, test_df


def main():
    """Main function for testing the preprocessing pipeline"""
    
    from pathlib import Path
    
    # Load the combined dataset
    data_file = Path("data/raw/beijing_air_quality_combined.csv")
    
    if not data_file.exists():
        print(f"Data file not found: {data_file}")
        return
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load data
    df = preprocessor.load_data(str(data_file))
    
    # Run preprocessing pipeline
    train_df, val_df, test_df = preprocessor.preprocess_pipeline(df)
    
    print("\nPreprocessing completed successfully!")
    print(f"Train shape: {train_df.shape}")
    print(f"Validation shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")


if __name__ == "__main__":
    main()
# Final Project

# Final Project Module

