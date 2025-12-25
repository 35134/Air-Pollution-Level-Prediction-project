# Air Pollution Level Prediction Project

##  Key Results

- **Dataset**: 420,768 hourly observations from 12 Beijing monitoring stations (2013-2017)
- **Baseline Model**: Linear Regression with RMSE = 27.19 μg/m³
- **Optimized Model**: XGBoost with RMSE = 3.06 μg/m³  
- **Performance Improvement**: **88.8% reduction in prediction error**
- **Model Accuracy**: R² = 0.999 for the optimized model

## 🚀 Quick Start

### Prerequisites
- Python 3.13 or compatible version
- Windows PowerShell or Command Prompt

### Installation
```bash
# Clone or download the project
cd Air-Pollution-Level-Prediction

# Install dependencies
pip install -r requirements.txt

# Run the complete project
python main.py

# Verify installation (optional)
python verify_clean.py
```

### Expected Output
```
============================================================
AIR POLLUTION LEVEL PREDICTION PROJECT
============================================================

Step 1: Loading and preparing data...
Data loaded successfully!
   Training set: (294827, 18)
   Validation set: (104587, 18)
   Test set: (16983, 18)

Step 2: Training baseline model...
Baseline model trained!
   Test RMSE: 27.19 ug/m3
   Test R2: 0.927

Step 3: Training optimized model with consistent features...
Using 80 common numeric features across all datasets
Optimized model trained!
   Test RMSE: 3.06 ug/m3
   Test R2: 0.999

Performance Improvement:
   RMSE improvement: 88.8%
   R2 improvement: 7.8%

PROJECT EXECUTION COMPLETED SUCCESSFULLY!
```

## 📋 Technical Details

### Models
- **Baseline**: Linear Regression with StandardScaler
- **Optimized**: XGBoost Regressor with 80 engineered features
- **Features**: Temporal, lag, rolling window, and interaction terms

### Data Processing
- Time-series split (2013-2015 train, 2016 validation, 2017 test)
- Missing value imputation
- Outlier detection using IQR method
- Feature engineering with temporal awareness

### Key Dependencies
- pandas 2.3.0
- scikit-learn 1.7.0
- XGBoost 3.1.2
- numpy 1.24.3
- matplotlib 3.7.1
- seaborn 0.12.2

## 🔧 Verification

The project includes a verification script to ensure all files are present:
```bash
python verify_clean.py
```

## 🎯 Project Highlights

✅ **Clean Architecture**: Minimal, production-ready codebase  
✅ **Robust Validation**: Temporal train/validation/test split  
✅ **Feature Engineering**: 80 engineered features for optimization  
✅ **Performance**: 88.8% improvement over baseline  
✅ **Reproducible**: Complete documentation and verification  
✅ **Efficient**: Only essential files, no clutter  

## 📈 Applications

- **Environmental Management**: Real-time air quality forecasting
- **Public Health**: Early warning systems for pollution episodes  
- **Policy Making**: Evidence-based environmental regulations
- **Urban Planning**: Infrastructure development decisions
