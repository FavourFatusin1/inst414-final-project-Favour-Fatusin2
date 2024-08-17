import pandas as pd
import joblib  # For saving the model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

# Ensure the 'models' directory exists
os.makedirs('data/models', exist_ok=True)

# Load the data
file_path1 = 'data/processed/loaded_firstchart.csv'
df1 = pd.read_csv(file_path1)

# Display column names to identify correct columns
print("Columns in df1:")
print(df1.columns)

# Data Preparation
# Remove percentage signs and convert to numeric
df1['Percentage of age\'s total identity theft reports, 2022'] = df1['Percentage of age\'s total identity theft reports, 2022'].str.rstrip('%').astype('float') / 100.0

# Define features and target
X = df1[['Number of reports, 2024 Q1-Q2']]  # Features
y = df1['Percentage of age\'s total identity theft reports, 2022']  # Target

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
model_path = 'data/models/linear_regression_model.pkl'
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

# Optionally, save predictions
predictions_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
predictions_path = 'data/models/predictions.csv'
predictions_df.to_csv(predictions_path, index=False)
print(f"Predictions saved to {predictions_path}")
