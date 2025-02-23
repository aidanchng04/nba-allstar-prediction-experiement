import pandas as pd
import os
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np
import shap



def load_data(path, name):
    model = joblib.load(os.path.join(path, name))
    return model


def load_test_data(path, x, y):
    X_test = pd.read_csv(os.path.join(path, x))
    y_test = pd.read_csv(os.path.join(path, y))   
    return X_test, y_test


def evaluate_model(model, x, y):
    y_pred = model.predict(x)
    y_prob = model.predict_proba(x)[:, 1]

    print("Classification Report:")
    print(classification_report(y, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

    print("ROC-AUC score: ", roc_auc_score(y, y_prob))



def compare_baseline(x, y):
    threshold = np.percentile(x["points"], 95)
    baseline_preds = (x["points"] > threshold).astype(int)

    print("Baseline Model Classication Report:")
    print(classification_report(y, baseline_preds))




def prediction_test(model, file_path, file_name):
    stats = pd.read_csv(os.path.join(file_path, file_name))

    prediction = model.predict(stats)
    probability = model.predict_proba(stats)[:, 1]

    # Remove file extension
    file_name = file_name.replace("_stats.csv", "")

    # Split by underscore
    firstName, lastName, seasonYear = file_name.split("_")

    print(f"{firstName} {lastName} Predicted All-Star Stutus in {seasonYear}: ", "Yes" if prediction[0] == 1 else "No")
    print("Model Confidence:", round(probability[0] * 100, 2), "%")


def explain_model(model, X_test):
    # """Uses SHAP values to explain model predictions."""
    # explainer = shap.Explainer(model, X_test)
    # shap_values = explainer(X_test)

    # print(X_test.shape)  # Check shape
    # print(X_test.columns)  # Check column names (if DataFrame)

    # print(shap_values.shape)  # Check shape of SHAP values

    # shap.summary_plot(shap_values[1], X_test)  # Example: Use SHAP values for class 1
    """Uses SHAP values to explain model predictions."""
    explainer = shap.Explainer(model, X_test)
    shap_values = explainer.shap_values(X_test, check_additivity=False)

    shap_values_class1 = shap_values[:, :, 1]  # Select SHAP values for class 1
    shap.summary_plot(shap_values_class1, X_test)  # Visualization in notebook, remove for production script

    # # Ensure X_test is a DataFrame with proper column names
    # if not isinstance(X_test, pd.DataFrame):
    #     X_test = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
    
    # # Plot SHAP summary
    # shap.summary_plot(shap_values[1], X_test)

    # shap.summary_plot(shap_values, X_test)
    


def main():
    model_path = "models"
    model_name = "optimal_model.pkl"

    processed_path = "data/processed"
    player_path = "data/players"
    X_test_name = "X_test_scaled.csv"
    y_test_name = "y_test.csv"
    player_file_name = "Mikal_Bridges_2025_stats.csv"

    model = load_data(model_path, model_name)
    X_test, y_test = load_test_data(processed_path, X_test_name, y_test_name)
    print(X_test)
    evaluate_model(model, X_test, y_test)
    compare_baseline(X_test, y_test)
    prediction_test(model, player_path, player_file_name)
    explain_model(model, X_test)


    print("everything works so far✅")
    return 


if __name__ == "__main__":
    main()
