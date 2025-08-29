import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('/Users/kassywang/Documents/Summer Project/R/without outliers/continuous.csv')
df = df.dropna(subset=["ratio_before", "ratio_after"])  # ensure targets are available
df.columns = df.columns.str.lower()

# Define features and targets
drop_cols = ["study_id", "match_id", "calculated_classification", "ratio_before", "ratio_after"]
X = df.drop(columns=[col for col in drop_cols if col in df.columns])
X = pd.get_dummies(X, drop_first=True)
X.columns = X.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)


y = df["ratio_after"]

# Clean NaNs
X = X.dropna()

y = y.loc[X.index]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42),
    "LightGBM": LGBMRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "AdaBoost": AdaBoostRegressor(random_state=42),
    "Extra Trees": ExtraTreesRegressor(random_state=42),
    "KNN": KNeighborsRegressor(),
    "SVR": SVR(),
    "MLP": MLPRegressor(random_state=42, max_iter=1000)
}

# Evaluation function
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        "MAE": mean_absolute_error(y_test, preds),
        "MSE": mean_squared_error(y_test, preds),
        "RMSE": mean_squared_error(y_test, preds, squared=False),
        "R2": r2_score(y_test, preds)
    }

# Collect results
results = {"Before Outlier Removal": {}, "After Outlier Removal": {}}
for name, model in models.items():
    results["After Outlier Removal"][name] = evaluate_model(model, X_train, X_test, y_train, y_test)

# Display results
before_df = pd.DataFrame(results["Before Outlier Removal"]).T.round(4)
after_df = pd.DataFrame(results["After Outlier Removal"]).T.round(4)

print("\n📊 Performance with ratio_before:")
print(before_df)

print("\n📊 Performance with ratio_after:")
print(after_df)


fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

best_model = SVR()

## Plot
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

axes[0].scatter(y_test, y_pred, color='black', label='Predictions')
axes[0].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'r--', label='Ideal')
axes[0].set_title(f"Observed vs Predicted (ratio_before)\n$R^2$ = {r2_score(y_test, y_pred):.3f}", fontsize=12)
axes[0].set_xlabel("Observed", fontsize=11)
axes[0].set_ylabel("Predicted", fontsize=11)
axes[0].legend()
axes[0].grid(True)

residuals = y_pred - y_test

axes[1].scatter(y_test, residuals, color='blue', alpha=0.7)
axes[1].axhline(0, color='red', linestyle='--')
axes[1].set_title("Residual Plot (ratio_before)", fontsize=12)
axes[1].set_xlabel("Observed", fontsize=11)
axes[1].set_ylabel("Residual (Pred - Obs)", fontsize=11)
axes[1].grid(True)

# Optional overall title (if submitting as a figure in thesis)
fig.suptitle("Model Calibration and Residual Diagnostics for ratio_before", fontsize=14, y=1.05)

# Show the plot
plt.show()