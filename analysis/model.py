import pandas as pd
import joblib  # For saving the model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

# Ensure the 'models' directory exists
def create_models_directory():
    os.makedirs('data/models', exist_ok=True)

# Load the dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Prepare the data by cleaning and selecting features/target
def prepare_data(df):
    # Remove percentage signs and convert to numeric
    df['percentage_of_age\'s_total_identity_theft_reports,_2022'] = df['percentage_of_age\'s_total_identity_theft_reports,_2022'].str.rstrip('%').astype('float') / 100.0

    # Define features and target
    X = df[['number_of_reports,_2024_q1-q2']]  # Features
    y = df['percentage_of_age\'s_total_identity_theft_reports,_2022']  # Target
    return X, y

# Split the data into training and testing sets
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Train the Linear Regression model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Evaluate the model's performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2, y_pred

# Save the model to disk
def save_model(model, model_path):
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

# Save predictions to a CSV file
def save_predictions(y_test, y_pred, predictions_path):
    predictions_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")
