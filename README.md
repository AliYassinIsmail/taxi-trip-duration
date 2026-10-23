# 🚕 NYC Taxi Trip Duration Prediction

A Machine Learning project to predict NYC taxi trip duration using Ridge Regression and geospatial feature engineering.

## 📌 Features & Pipeline

- **Feature Engineering:** Haversine & Manhattan distances, time-based features (rush hour, cycles), and seasonal indicators.
- **Preprocessing:** Log transformations (`log1p`), outlier removal, and `StandardScaler` normalization.
- **Model:** Ridge Regression evaluation using RMSE and $R^2$ metrics.

## 📁 Project Structure

```text
├── project.py         # Project class & feature engineering logic
├── finalmodel.py            # Execution script using argparse
├── json.json          # Configuration file for dataset path
└── requirements.txt   # Python dependenciesS
```

scores:
Rmse for the test set 0.4896199598284822
r2 score for test set 0.6215591487446235
