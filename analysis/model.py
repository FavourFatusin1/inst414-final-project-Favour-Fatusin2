import pandas as pd
import numpy as np
import os
import joblib  # For saving the model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# Ensure the 'models' directory exists
os.makedirs('data/models', exist_ok=True)

# Load the processed data
file_path1 = 'data/processed/loaded_firstchart.csv'
file_path2 = 'data/processed/loaded_secondchart.csv'

df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)

# Example Data Preparation (assuming df1 has 'feature' and 'target' columns)
X = df1[['feature']]  # Features
y = df1['target']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Save the model to disk
model_path = 'data/linear_regression_model.pkl'
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

# Optionally, save predictions
predictions_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
predictions_path = 'data/predictions.csv'
predictions_df.to_csv(predictions_path, index=False)
print(f"Predictions saved to {predictions_path}")

