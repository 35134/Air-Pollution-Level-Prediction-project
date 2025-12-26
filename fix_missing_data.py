#!/usr/bin/env python3
"""
Script to create basic engineered features from cleaned data since Git LFS files are not accessible.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_basic_features(df):
    """Create basic engineered features from cleaned data."""
    df_eng = df.copy()
    
    # Temporal features
    df_eng['day_of_week'] = df_eng.index.dayofweek
    df_eng['day_of_year'] = df_eng.index.dayofyear
    df_eng['is_weekend'] = (df_eng['day_of_week'] >= 5).astype(int)
    
    # Lag features (1-6 hours)
    for lag in range(1, 7):
        df_eng[f'PM2.5_lag_{lag}'] = df_eng['PM2.5'].shift(lag)
        df_eng[f'TEMP_lag_{lag}'] = df_eng['TEMP'].shift(lag)
        df_eng[f'PRES_lag_{lag}'] = df_eng['PRES'].shift(lag)
    
    # Rolling statistics (24-hour window)
    window = 24
    df_eng[f'PM2.5_rolling_mean_{window}'] = df_eng['PM2.5'].rolling(window=window, min_periods=1).mean()
    df_eng[f'PM2.5_rolling_std_{window}'] = df_eng['PM2.5'].rolling(window=window, min_periods=1).std()
    df_eng[f'TEMP_rolling_mean_{window}'] = df_eng['TEMP'].rolling(window=window, min_periods=1).mean()
    df_eng[f'PRES_rolling_mean_{window}'] = df_eng['PRES'].rolling(window=window, min_periods=1).mean()
    
    # Pollutant ratios
    df_eng['PM_ratio'] = df_eng['PM2.5'] / (df_eng['PM10'] + 1e-6)  # Avoid division by zero
    df_eng['NO2_SO2_ratio'] = df_eng['NO2'] / (df_eng['SO2'] + 1e-6)
    
    # Interaction terms
    df_eng['TEMP_x_PRES'] = df_eng['TEMP'] * df_eng['PRES']
    df_eng['WIND_x_TEMP'] = df_eng['WSPM'] * df_eng['TEMP']
    
    return df_eng

def main():
    """Main function to create engineered datasets."""
    print("Creating basic engineered features from cleaned data...")
    
    # Load cleaned data
    print("Loading cleaned data...")
    train_clean = pd.read_csv('datasets/data/cleaned/train_cleaned.csv', index_col=0, parse_dates=True)
    val_clean = pd.read_csv('datasets/data/cleaned/val_cleaned.csv', index_col=0, parse_dates=True)
    test_clean = pd.read_csv('datasets/data/cleaned/test_cleaned.csv', index_col=0, parse_dates=True)
    
    print(f"Original data shapes:")
    print(f"  Train: {train_clean.shape}")
    print(f"  Val: {val_clean.shape}")
    print(f"  Test: {test_clean.shape}")
    
    # Create engineered features
    print("Creating engineered features...")
    train_eng = create_basic_features(train_clean)
    val_eng = create_basic_features(val_clean)
    test_eng = create_basic_features(test_clean)
    
    print(f"Engineered data shapes:")
    print(f"  Train: {train_eng.shape}")
    print(f"  Val: {val_eng.shape}")
    print(f"  Test: {test_eng.shape}")
    
    # Save engineered data
    print("Saving engineered data...")
    train_eng.to_csv('datasets/data/cleaned/train_engineered_optimized.csv')
    val_eng.to_csv('datasets/data/cleaned/val_engineered_optimized.csv')
    test_eng.to_csv('datasets/data/cleaned/test_engineered_optimized.csv')
    
    print("Basic engineered datasets created successfully!")
    print("Note: These are simplified features compared to the original engineered datasets.")

if __name__ == "__main__":
    main()