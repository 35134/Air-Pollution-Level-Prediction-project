# Data Management Strategy

## Large Files Handling

Due to GitHub's 100MB file size limit, large datasets are managed as follows:

### Files Excluded from Repository:
- `train_engineered_optimized.csv` (206MB) - Contains engineered features
- `val_engineered_optimized.csv` (73MB) - Validation set with engineered features
- `*.dll` files - XGBoost and related libraries

### Files Included:
- `train_cleaned.csv` (30MB) - Original cleaned training data
- `val_cleaned.csv` (11MB) - Original cleaned validation data
- `test_cleaned.csv` (12MB) - Original cleaned test data
- `test_engineered_optimized.csv` (12MB) - Test set with engineered features

## How to Use This Project

### Option 1: Use Included Data (Recommended)
The project runs perfectly with the included cleaned datasets. The defense notebook demonstrates the complete methodology.

### Option 2: Generate Engineered Features
To reproduce the engineered features, run the feature engineering pipeline in the notebook.

### Option 3: Use Git LFS (Advanced)
If you need the full engineered datasets, install Git LFS and use the .gitattributes configuration.

## Data Sources
- Original data: Beijing air quality monitoring stations (2013-2017)
- Feature engineering: Advanced temporal and interaction features
- Cleaning: Comprehensive missing value handling with 95%+ retention

## Performance Summary
- Baseline: Linear Regression with 15 features
- Optimized: XGBoost with 63 engineered features
- Improvement: 86.2% reduction in RMSE (24.30 → 3.35 μg/m³)
- Model Accuracy: R² = 0.999
---\n\n**Note**: This documentation was finalized as part of the final project submission.

---
**Final Project**: Complete and ready for presentation

