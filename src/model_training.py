import pandas as pd
from sklearn.model_selection import train_test_split
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import shap
from sklearn.preprocessing import StandardScaler
import joblib



# Function to load and split data
def load_data(file_path, file_name):
    df = pd.read_csv(os.path.join(file_path, file_name))
    return df



# Function to split the data loaded
def split_data(df):
    X = df.drop(columns="allStar")
    y = df['allStar']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler() # Initialize the scaler

    data_to_scale = [
        "points", "freeThrowsMade", "fieldGoalsMade", "plusMinusPoints",
        "minutes", "gamesPlayed", "reboundsDefensive", "assists", "turnovers"
    ] # List of columns required for scaling 

    print(X_test.var())

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    # Fit only on training data)
    X_train_scaled[data_to_scale] = scaler.fit_transform(X_train_scaled[data_to_scale])

    # Transform test data using the same scaler (but do NOT fit again!)
    X_test_scaled[data_to_scale] = scaler.transform(X_test_scaled[data_to_scale])


    # Save the scaler for later use (for new player predictions)
    joblib.dump(scaler, os.path.join("models", "scaler_latest.pkl"))

    print("successfully split data")

    # print("Training set size:", X_train.shape)
    # print("Testing set size:", X_test.shape)

    return X_train_scaled, X_test_scaled, y_train, y_test, X_test



# Function to train models
def train_model(X_train, y_train):
    models = {
        "Logical Regression" : LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest" : RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost" : XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        models[name] = model

    return models



# Function to evaluate model
def evaluate_model(models, X_test, y_test):
    for name, model in models.items():
        preds = model.predict(X_test)
        print(f"{name} Metrics:")
        print(classification_report(y_test, preds))
"""
Logistic Regression has the best overall performance

Note that recall are still relatively low across all models,
meaning some true allStars are being misclassified as non_allstars
"""



# Function to conduct Hyperparameter tuning
def hyperparameter_tuning(X_train, y_train):
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15],
        "min_samples_split": [2, 5, 10]
    }

    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring="f1")
    grid_search.fit(X_train, y_train)

    print("Best parameters:", grid_search.best_params_)
    print("Best Estimator:", grid_search.best_estimator_)

    return grid_search.best_estimator_



# Function to explain model
def explain_model(model, X_train, X_test):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer.shap_values(X_test, check_additivity=False)

    shap_values_class1 = shap_values[:, :, 1]  # Select SHAP values for class 1
    shap.summary_plot(shap_values_class1, X_test)  # Visualization in notebook, remove for production script



# Function to save model
def save_model(model, X_test_scaled, X_test, y_test, model_path, processed_path):

    path = os.path.join(model_path, "optimal_model.pkl")
    joblib.dump(model, path) # Save scaler for future use

    # Convert back to a DataFrame (restore original column names)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    print(X_test_scaled_df)
    print(X_test_scaled_df.var())

    X_test_scaled_df.to_csv(os.path.join(processed_path, "X_test_scaled.csv"), index=False)
    y_test.to_csv(os.path.join(processed_path, "y_test.csv"), index=False)

    print("Best model saved successfully!")



# Main function to execute feature engineering pipeline.
def main():
    processed_path = "data/processed"
    model_path = "models"
    file_name = "processed_features.csv"

    df = load_data(processed_path, file_name)
    X_train_scaled, X_test_scaled, y_train, y_test, X_test = split_data(df)
    models = train_model(X_train_scaled, y_train)
    evaluate_model(models, X_test, y_test)
    optimal_model = hyperparameter_tuning(X_train_scaled, y_train)
    explain_model(optimal_model, X_train_scaled, X_test)
    save_model(optimal_model, X_test_scaled, X_test, y_test, model_path, processed_path)
    print("everything works so far✅")


if __name__ == "__main__":
    main()