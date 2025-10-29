from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split # Importing the missing function
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # Import metrics here
import xgboost as xgb
import lightgbm as lgb
import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv("Minor Genus.csv")
unique_circuit = df['circuit'].unique()

# Split data before encoding to prevent data leakage
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Drop original 'circuit' column
train_df.drop(columns=['circuit'], inplace=True)
test_df.drop(columns=['circuit'], inplace=True)

# Features and target variable
X_train = train_df.drop(columns=["Power_total(nW)"])
y_train = train_df["Power_total(nW)"]

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Compute evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {"MAE": mae, "MSE": mse, "R² Score": r2}

# Convert results to DataFrame for easy comparison
results_df = pd.DataFrame(results).T

# Print results
print(results_df)

X_test = test_df.drop(columns=["Power_total(nW)"])
y_test = test_df["Power_total(nW)"]

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "MLP Regressor (Neural Network)": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42),
    "XGBoost": xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "LightGBM": lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42),

    "SVR": SVR(kernel="rbf", C=1.0, epsilon=0.1)
}

# Define the hyperparameter grid for tuning
param_grid = {
    'hidden_layer_sizes': [(64, 32), (128, 64, 32), (256, 128, 64)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],  # L2 regularization
    'learning_rate_init': [0.001, 0.01],  # Initial learning rate
    'max_iter': [1000, 2000]
}


# Create and train the Ridge Regression model
ridge_model = Ridge()
ridge_model.fit(X_train, y_train)

# Make predictions
y_pred_ridge = ridge_model.predict(X_test)

# Evaluate the model
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print("Ridge Regression Results:")
print(f"MAE: {mae_ridge}")
print(f"MSE: {mse_ridge}")
print(f"R² Score: {r2_ridge}")


# Assuming y_test and y_pred are already defined from your previous code Linear plot
plt.figure(figsize=(10, 6))
plt.plot(X_test.index, y_test, label='Actual', marker='o', linestyle='-', color='blue')  # Actual values
plt.plot(X_test.index, y_pred, label='Predicted', marker='x', linestyle='--', color='red')  # Predicted values

plt.xlabel("X_test Index") # Assuming X_test has an index
plt.ylabel("Power_total(nW)")
plt.title("Actual vs. Predicted Values")
plt.legend()
plt.grid(True)
plt.show()

# Bar PLot of Actual and predicted value

def plot_test_vs_pred(X_test, y_test, y_pred, outlier_threshold=1.5):
    # Calculate the IQR for outlier detection
    q1 = np.percentile(y_test, 25)
    q3 = np.percentile(y_test, 75)
    iqr = q3 - q1
    lower_bound = q1 - outlier_threshold * iqr
    upper_bound = q3 + outlier_threshold * iqr

    # Filter out outliers
    inliers_mask = (y_test >= lower_bound) & (y_test <= upper_bound)
    X_test_inliers = X_test[inliers_mask]
    y_test_inliers = y_test[inliers_mask]
    y_pred_inliers = y_pred[inliers_mask]

    # Generate bar positions
    indices = np.arange(len(y_test_inliers))  # X-axis positions
    bar_width = 0.4  # Width of each bar

    # Plotting
    plt.figure(figsize=(12, 6))

    # Actual values (shifted left)
    plt.bar(indices - bar_width / 2, y_test_inliers, width=bar_width, label='Actual', color='blue', alpha=0.7)

    # Predicted values (shifted right)
    plt.bar(indices + bar_width / 2, y_pred_inliers, width=bar_width, label='Predicted', color='red', alpha=0.7)

    plt.xlabel("X_test Index")
    plt.ylabel("Power_total(nW)")
    plt.title("Actual vs. Predicted Values (Outliers Removed)")
    plt.xticks(indices)  # Ensure correct index alignment
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()

# Call the function
plot_test_vs_pred(X_test, y_test, y_pred)
