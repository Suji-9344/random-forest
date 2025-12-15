import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib # For saving and loading the model efficiently

def main():
    print("--- Random Forest Classifier Application ---")

    # 1. Generate Synthetic Data
    print("Generating synthetic data...")
    X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"Data generated: X_train shape {X_train.shape}, y_train shape {y_train.shape}")

    # 2. Initialize and Train the Random Forest Classifier
    print("Training Random Forest Classifier...")
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    print("Training complete.")

    # 3. Evaluate the Model
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on test set: {accuracy:.2f}")

    # 4. Save the trained model
    model_filename = 'random_forest_model.pkl'
    joblib.dump(rf_classifier, model_filename)
    print(f"Trained model saved to '{model_filename}'")

    # 5. Load the model (demonstration of loading for future use)
    print(f"Loading model from '{model_filename}'...")
    loaded_rf_classifier = joblib.load(model_filename)
    print("Model loaded successfully.")

    # 6. Make predictions using the loaded model
    # Using X_test as new data for demonstration
    new_predictions = loaded_rf_classifier.predict(X_test)
    print(f"Predictions made using loaded model: {new_predictions[:5]}...")

    # 7. Save predictions to a CSV file
    predictions_df = pd.DataFrame(new_predictions)
    output_file_path = 'predictions_output.csv'
    predictions_df.to_csv(output_file_path, index=False, header=False)
    print(f"Predictions saved to '{output_file_path}'")

    # 8. Analyze prediction distribution (optional, but good for app output)
    print("\nDistribution of Predictions:")
    display(predictions_df[0].value_counts())


if __name__ == "__main__":
    main()
