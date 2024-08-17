import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

def train_and_save_model():
    # Define file paths
    file_path = 'data/processed/loaded_firstchart.csv'
    model_path = 'data/models/linear_regression_model.pkl'
    predictions_path = 'data/models/predictions.csv'

    # Ensure the 'models' directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Load the dataset
    df = pd.read_csv(file_path)

    # Prepare the data by cleaning and selecting features/target
    df['percentage_of_age\'s_total_identity_theft_reports,_2022'] = df['percentage_of_age\'s_total_identity_theft_reports,_2022'].str.rstrip('%').astype('float') / 100.0
    X = df[['number_of_reports,_2024_q1-q2']]  # Features
    y = df['percentage_of_age\'s_total_identity_theft_reports,_2022']  # Target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model's performance
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print the performance metrics
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    # Save the model to disk
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Save predictions to a CSV file
    predictions_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")


