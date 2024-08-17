import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import os

def evaluate_and_save_predictions():
    # Define file paths
    file_path = 'data/processed/loaded_firstchart.csv'
    model_path = 'data/models/linear_regression_model.pkl'
    predictions_path = 'data/models/evaluated_predictions.csv'

    # Ensure the 'models' directory exists
    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)

    # Load the processed data
    df = pd.read_csv(file_path)

    # Prepare the data by converting the target variable to numeric
    df['percentage_of_age\'s_total_identity_theft_reports,_2022'] = df['percentage_of_age\'s_total_identity_theft_reports,_2022'].str.rstrip('%').astype(float) / 100.0
    X = df[['number_of_reports,_2024_q1-q2']]  # Feature(s)
    y = df['percentage_of_age\'s_total_identity_theft_reports,_2022']  # Target

    # Load the saved model
    model = joblib.load(model_path)

    # Make predictions using the loaded model
    y_pred = model.predict(X)

    # Evaluate the model's performance
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    # Save predictions to a CSV file
    predictions_df = pd.DataFrame({
        'Actual': y,
        'Predicted': y_pred
    })
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Evaluated predictions saved to {predictions_path}")

