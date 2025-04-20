# Jane-Street-Real-Time-Market-Data-Forecasting
## 1. Link of the project
https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/overview
## 2. Requirements
```
numpy
pandas
matplotlib
seaborn
polars
gc
xgboost
scikit-learin
```
## 3. Method
1. Data Process
+ Remove high related features
+ Remove features with null value precentage >= 5%
+ Filling null values with 0
2. Split Dataset
+ Training set: date_id ≤ 1200 (800 trading days)
+ Validation set: 1200 < date_id ≤ 1500 (300 trading days)
+ Test set: date_id > 1500 (29 trading days)
3. Training
+ Two metric: $\text{R}^2$ and RMSE
## 4. Visualization of feature analysis
### 4.1 Missing Values
Here is the visualization of null value precentage of features.
![](.\imgs\missing_values.png)
### 4.2 Distribution of some festures
![](.\imgs\feature_00_distribution.png)
![](.\imgs\feature_10_distribution.png)
![](.\imgs\feature_20_distribution.png)
### 4.3 Distribution of responders
![](.\imgs\responder_2_distribution.png)
![](.\imgs\responder_6_distribution.png)
![](.\imgs\responder_8_distribution.png)
### 4.4 Data anslysis in time series
![](.\imgs\responder_6_rolling_mean.png)
### 4.5 Feature correlation
![](.\imgs\all_feature_correlation_heatmap.png)
![](.\imgs\feature_correlation_heatmap.png)
![](.\imgs\feature_correlation_heatmap_2.png)
## 5. Training Loss and Valization Loss
![](.\imgs\training_metrics_evolution.png)
## 6. Prediction
```
Xgboost R2 score: 0.011455476967631495
```