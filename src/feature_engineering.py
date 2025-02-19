import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
import joblib
import os



# Function to load data
def load_data(file_path, indiv_df_file, team_df_file):
    indiv_stats_sum_file = os.path.join(file_path, indiv_df_file)
    team_stats_sum_file = os.path.join(file_path, team_df_file)

    indiv_stats_sum = pd.read_csv(indiv_stats_sum_file)
    team_stats_sum = pd.read_csv(team_stats_sum_file)

    return indiv_stats_sum, team_stats_sum



# Function to filter columns
def filter_columns(indiv_df, team_df, selected_features):
    new_df = indiv_df[['playerName'] + ['season_year'] + ['teamTricode'] + selected_features + ["allStar"]]

    usage_team_stats = ['fieldGoalsAttempted', 'freeThrowsAttempted', 'turnovers', 'minutes']
    usage_team_stats_sum = team_df[usage_team_stats].rename(columns=lambda x: x + "_team")
    usage_team_stats_sum['season_year'] = team_df['season_year']
    usage_team_stats_sum['teamTricode'] = team_df['teamTricode']
    
    indiv_df = indiv_df.merge(usage_team_stats_sum, on=['teamTricode', 'season_year'], how='left')
    return new_df, indiv_df



# Function to create new features
def create_features(df, indiv_stats_sum):
    df["PER"] = (
        (indiv_stats_sum['fieldGoalsMade'] * 85.910) +
        (indiv_stats_sum['steals'] * 53.897) +
        (indiv_stats_sum['threePointersMade'] * 51.757) +
        (indiv_stats_sum['freeThrowsMade'] * 46.845) +
        (indiv_stats_sum['blocks'] * 39.190) +
        (indiv_stats_sum['reboundsOffensive'] * 39.910) +
        (indiv_stats_sum['assists'] * 34.677) +
        (indiv_stats_sum['reboundsDefensive'] * 14.707) -
        (indiv_stats_sum['foulsPersonal'] * 17.174) -
        ((indiv_stats_sum['freeThrowsAttempted'] - indiv_stats_sum['freeThrowsMade']) * 20.091) -
        ((indiv_stats_sum['fieldGoalsAttempted'] - indiv_stats_sum['fieldGoalsMade']) * 39.190) -
        (indiv_stats_sum['turnovers'] * 53.897)
    ) / (
        indiv_stats_sum['minutes']) # Computing Player Efficiency Rating

    df["USG%"] = (
        (indiv_stats_sum["fieldGoalsAttempted"] + (0.44 * indiv_stats_sum["freeThrowsAttempted"]) + indiv_stats_sum["turnovers"]) * (indiv_stats_sum['minutes_team'] / 5
    ) / (
        (indiv_stats_sum["fieldGoalsAttempted_team"] + (0.44 * indiv_stats_sum["freeThrowsAttempted_team"]) + indiv_stats_sum["turnovers_team"]) * (indiv_stats_sum['minutes'])
    ) * 100 ) # Computing Usage Rate %
    return df



# Function to handle missing data for numerical columns only
def handle_missing_data(df):
    """Fill missing values in numerical columns with column means."""
    numeric_cols = df.select_dtypes(include=['number']).columns  # Select only numerical columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    return df



# Function to scale features
def scale_features(df, scaled_features):
    """Scale numerical features."""
    scaler = StandardScaler()

    df[scaled_features] = scaler.fit_transform(df[scaled_features])
    model_path = os.path.join("models", "scaler.pkl")
    joblib.dump(scaler, model_path) # Save scaler for future use

    return df



# Function to select features
def select_features(df):
    """Select the most important features using SHAP."""
    # Train a simple model to determine feature importance
    X = df.drop(columns=["season_year", "playerName", "teamTricode", "allStar"])
    y = df["allStar"]

    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)


    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X, check_additivity=False)

    # shap_values_class1 = shap_values[:, :, 1]  # Select SHAP values for class 1
    # shap.summary_plot(shap_values_class1, X)  # Visualization in notebook, remove for production script

    # Update the best features based on the latest SHAP summary
    latest_features = [
        "freeThrowsMade", "points", "plusMinusPoints", "fieldGoalsMade", "PER",
        "turnovers", "USG%", "gamesPlayed", "reboundsDefensive", "minutes", "assists"
    ]

    return df[latest_features + ["allStar"]]



# Function to save data 
def save_data(df, file_path):
    """Save processed dataset."""
    df.to_csv(os.path.join(file_path, "processed_features.csv"), index=False)



# Main function to execute feature engineering pipeline.
def main():
    file_path = "data/processed"
    indiv_df_file = "indiv_stats_sum.csv"
    team_df_file = "team_stats_sum.csv"

    selected_features = [
        "points", "freeThrowsMade", "fieldGoalsMade", "plusMinusPoints",
        "minutes", "gamesPlayed", "reboundsDefensive", "assists", "turnovers"
    ]

    indiv, team = load_data(file_path, indiv_df_file, team_df_file)
    df, indiv = filter_columns(indiv, team, selected_features)
    df = create_features(df, indiv)
    df = handle_missing_data(df)
    df = scale_features(df, selected_features)
    df = select_features(df)
    save_data(df, file_path)

    print("Feature Engineering Completed!")


if __name__ == "__main__":
    main()