import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import os

# Ensure the 'models' directory exists
os.makedirs('data/models', exist_ok=True)

# Load the processed data
file_path1 = 'data/processed/loaded_firstchart.csv'
df1 = pd.read_csv(file_path1)

# Print column names to verify
print("Columns in df1:")
print(df1.columns)

# Convert target variable to numeric
df1['Percentage of age\'s total identity theft reports, 2022'] = df1['Percentage of age\'s total identity theft reports, 2022'].str.rstrip('%').astype(float) / 100.0

# Select features and target
X = df1[['Number of reports, 2024 Q1-Q2']]  # Feature(s)
y = df1['Percentage of age\'s total identity theft reports, 2022']  # Target

# Load the saved model
model_path = 'data/models/linear_regression_model.pkl'
model = joblib.load(model_path)

# Make predictions using the loaded model
y_pred = model.predict(X)

# Evaluate the model
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Output the performance metrics
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Save predictions
predictions_df = pd.DataFrame({
    'Actual': y,
    'Predicted': y_pred
})
predictions_path = 'data/models/evaluated_predictions.csv'
predictions_df.to_csv(predictions_path, index=False)
print(f"Evaluated predictions saved to {predictions_path}")
