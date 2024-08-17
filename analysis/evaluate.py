import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import os

# Ensure the 'models' directory exists
def create_models_directory():
    os.makedirs('data/models', exist_ok=True)

# Load the processed data
def load_data(file_path):
    return pd.read_csv(file_path)

# Prepare the data by converting the target variable to numeric
def prepare_data(df):
    df['percentage_of_age\'s_total_identity_theft_reports,_2022'] = df['percentage_of_age\'s_total_identity_theft_reports,_2022'].str.rstrip('%').astype(float) / 100.0
    X = df[['number_of_reports,_2024_q1-q2']]  # Feature(s)
    y = df['percentage_of_age\'s_total_identity_theft_reports,_2022']  # Target
    return X, y

# Load the saved model
def load_model(model_path):
    return joblib.load(model_path)

# Make predictions using the loaded model
def make_predictions(model, X):
    return model.predict(X)

# Evaluate the model's performance
def evaluate_model(y, y_pred):
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return mse, r2

# Save predictions to a CSV file
def save_predictions(y, y_pred, predictions_path):
    predictions_df = pd.DataFrame({
        'Actual': y,
        'Predicted': y_pred
    })
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Evaluated predictions saved to {predictions_path}")
