import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import optuna


import gc
import xgboost as xgb
from sklearn.model_selection import train_test_split

sample_pl = []

for i in range(9):
    df_pl = pl.read_parquet(f'data/train.parquet/partition_id={i}/part-0.parquet')
    sample_pl.append(df_pl)


df_pl = pl.concat(sample_pl)

# high related and missing values
Features_to_remove = ['feature_74','feature_78','feature_12','feature_70','feature_13','feature_14','feature_69', \
                      'feature_30', 'feature_17','feature_29','feature_21','feature_02','feature_43','feature_41', \
                        'feature_34','feature_32', \
                        'feature_26', 'feature_27', 'feature_31', 'feature_21', 'feature_39', \
                        'feature_42', 'feature_50', 'feature_53', 'feature_03', 'feature_01', 'feature_02', \
                        'feature_04', 'feature_00'] # missing values|

# Remove the specified features from the dataset
df_pl_clean = df_pl.drop(Features_to_remove)

# Verify the features have been removed
print(f"Number of features after removal: {len(df_pl_clean.columns)}")
print("Remaining features:", df_pl_clean.columns)

# fill the missing values in features with 0
fill_cols = df_pl_clean.columns[4:56]
df_pl_clean = df_pl_clean.with_columns(
    pl.col(fill_cols).fill_null(0),
)

# Split the data into train and test data.
# train and validation: 1200+300 days
# test data: 29 days
Test_data = df_pl_clean.filter(pl.col('date_id')>1500)
# Test_data = Test_data.sample(n=1000, seed = 0)

Train_data = df_pl_clean.filter(pl.col('date_id')<=1500)

feature_names = df_pl_clean.columns[4:56]

X_train = Train_data.filter(pl.col('date_id')<=1200).select(feature_names)
Y_train = Train_data.filter(pl.col('date_id')<=1200).select(['responder_6'])
W_train = Train_data.filter(pl.col('date_id')<=1200).select(['weight'])


X_val = Train_data.filter((pl.col('date_id')>1200) & (pl.col('date_id')<=1500)).select(feature_names)
Y_val = Train_data.filter((pl.col('date_id')>1200) & (pl.col('date_id')<=1500)).select(['responder_6'])
W_val = Train_data.filter((pl.col('date_id')>1200) & (pl.col('date_id')<=1500)).select(['weight'])


X_test = Test_data.select(feature_names)
Y_test = Test_data.select(['responder_6'])
W_test = Test_data.select(['weight'])


# Custom R2 metric for XGBoost
def r2_xgb(y_true, y_pred, sample_weight):
    # Convert Polars DataFrames to numpy arrays if needed
    y_true_np = y_true.to_numpy() if hasattr(y_true, 'to_numpy') else y_true
    y_pred_np = y_pred.to_numpy() if hasattr(y_pred, 'to_numpy') else y_pred
    sample_weight_np = sample_weight.to_numpy() if hasattr(sample_weight, 'to_numpy') else sample_weight
    sample_weight_np = sample_weight_np.reshape(y_pred_np.shape)
    # Calculate weighted R2 score
    numerator = np.average((y_pred_np - y_true_np) ** 2, weights=sample_weight_np)
    denominator = np.average((y_true_np) ** 2, weights=sample_weight_np) + 1e-6
    r2 = 1 - numerator / denominator
    return -r2


from sklearn.metrics import root_mean_squared_error

model = xgb.XGBRegressor(n_estimators=1600,
                 learning_rate=0.02,
                 max_depth=12,
                 tree_method='hist',
                 device="cuda",
                 objective='reg:squarederror',
                 eval_metric=r2_xgb,
                 verbosity=3,
                 # disable_default_eval_metric=True, 
                 early_stopping_rounds=50)

# Train XGBoost model with early stopping and verbose logging
model.fit(X_train, Y_train, sample_weight=W_train, 
          eval_set=[(X_train, Y_train), (X_val, Y_val)], 
          sample_weight_eval_set=[W_train, W_val], 
          verbose=True)

# Save evaluation results for plotting
results = model.evals_result()
np.save('training_results.npy', results)

train_rmse = results['validation_0']['rmse']
val_rmse = results['validation_1']['rmse']
train_r2 = -1 * np.array(results['validation_0']['r2_xgb'])
val_r2 = -1 * np.array(results['validation_1']['r2_xgb'])

# Create plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot RMSE curves
ax1.set_xlabel('Boosting Rounds')
ax1.set_ylabel('RMSE')
ax1.plot(train_rmse, label='Train RMSE', linestyle='--', color='tab:orange')
ax1.plot(val_rmse, label='Val RMSE', linestyle='-', color='tab:blue')
ax1.set_title('RMSE Evolution')
ax1.legend(loc='upper right')
ax1.grid(True)

# Plot R2 curves 
ax2.set_xlabel('Boosting Rounds')
ax2.set_ylabel('R2 Score')
ax2.plot(train_r2, label='Train R2', linestyle='--', color='tab:orange')
ax2.plot(val_r2, label='Val R2', linestyle='-', color='tab:blue')
ax2.set_title('R2 Evolution')
ax2.legend(loc='upper right')
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_metrics_evolution.png')
plt.show()

# Make predictions using best iteration
best_iteration = model.best_iteration
Y_pred = model.predict(X_test, iteration_range=(0, best_iteration))
Y_pred = pl.from_numpy(Y_pred, schema=["Pred"])

# Evaluate model
r2_xgb = r2_xgb(Y_test, Y_pred, W_test)
print(f"Xgboost R2 score: {-1*r2_xgb}")

