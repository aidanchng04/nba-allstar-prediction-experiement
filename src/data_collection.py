# Importing Libraries
import pandas as pd
import os
import unidecode 
import matplotlib.pyplot as plt
import seaborn as sns


# Establishing paths
raw_path = "data/raw"
processed_path = "data/processed"


# Creating file names
p1 = os.path.join(raw_path, "regular_season_box_scores_2010_2024_part_1.csv")
p2 = os.path.join(raw_path, "regular_season_box_scores_2010_2024_part_2.csv")
p3 = os.path.join(raw_path, "regular_season_box_scores_2010_2024_part_3.csv")
all_star_file = os.path.join(raw_path, "all_star_selections_complete.csv")


# Using pandas to read 3 different csv files
df1 = pd.read_csv(p1)
df2 = pd.read_csv(p2)
df3 = pd.read_csv(p3)
all_star_df = pd.read_csv(all_star_file)


# Combining all 3 dataframes into one 
df = pd.concat([df1, df2, df3], ignore_index=True)


# Altering the season column from "2010-11" to "2011"
df["season_year"] = df["season_year"].apply(lambda x: int("20" + x.split("-")[1]))


# Creating and applying function to convert MM:SS format to total seconds
def convert_to_minutes(time_str):
    if pd.isna(time_str):  # Handle NaN values
        return None
    try:
        minutes, seconds = map(int, time_str.split(":"))
        return minutes + (seconds / 60)  # Convert to decimal minutes
    except ValueError:  # If conversion fails
        return 

df['minutes'] = df['minutes'].apply(convert_to_minutes)


# Define game stats list used for Aggregation
game_stats_list = [
    'minutes', 'fieldGoalsMade', 'fieldGoalsAttempted', 'fieldGoalsPercentage', 'threePointersMade', 'threePointersAttempted',
    'threePointersPercentage', 'freeThrowsMade', 'freeThrowsAttempted', 'freeThrowsPercentage',
    'reboundsOffensive', 'reboundsDefensive', 'reboundsTotal', 'assists', 'steals', 'blocks',
    'turnovers', 'foulsPersonal', 'points', 'plusMinusPoints'
]


# Using Aggregation to find season totals
agg_methods_sum = {col: lambda x: x.sum(skipna=True) for col in game_stats_list} # Defining Aggregation using sum

indiv_stats_sum = df.groupby(["playerName", "season_year", "teamTricode"]).agg(agg_methods_sum).reset_index() 
game_stats_sum = df.groupby(['season_year', 'teamTricode', 'gameId']).agg(agg_methods_sum).reset_index()
team_stats_sum = game_stats_sum.groupby(['season_year', 'teamTricode']).agg(agg_methods_sum).reset_index() # Perform aggregation to find season totals


# Creating gamesPlayed column 
indiv_games_played = df.groupby(['playerName', 'season_year', 'teamTricode'])['gameId'].nunique().reset_index()
team_games_played = df.groupby(['teamTricode', 'season_year'])['gameId'].nunique().reset_index() # Counting number of games played

indiv_games_played.rename(columns={'gameId': 'gamesPlayed'}, inplace=True)
team_games_played.rename(columns={'gameId': 'gamesPlayed'}, inplace=True) # Renaming columns to 'gamesPlayed'

indiv_stats_sum = indiv_stats_sum.merge(indiv_games_played, on=['playerName', 'season_year', 'teamTricode'], how='left')
team_stats_sum = team_stats_sum.merge(team_games_played, on=['teamTricode', 'season_year'], how='left') # Merging back to sums df


# Dividing sums by gamesPlayed to compute averages
indiv_stats_avg = indiv_stats_sum.copy()
team_stats_avg = team_stats_sum.copy() # copying sum dfs and storing em into new dfs

indiv_stats_avg[game_stats_list] = indiv_stats_sum[game_stats_list].div(indiv_stats_sum['gamesPlayed'], axis=0)
team_stats_avg[game_stats_list] = team_stats_sum[game_stats_list].div(team_stats_sum['gamesPlayed'], axis=0)


# Fixing shooting percentages
def calculate_shooting_percentages(df, fgm_col, fga_col):
    return (df[fgm_col] / df[fga_col] * 100).fillna(0)  # Function to calculate shooting percentage for given columns.

team_stats_sum['fieldGoalsPercentage'] = calculate_shooting_percentages(team_stats_sum, 'fieldGoalsMade', 'fieldGoalsAttempted')
team_stats_sum['threePointersPercentage'] = calculate_shooting_percentages(team_stats_sum, 'threePointersMade', 'threePointersAttempted')
team_stats_sum['freeThrowsPercentage'] = calculate_shooting_percentages(team_stats_sum, 'freeThrowsMade', 'freeThrowsAttempted')

team_stats_avg['fieldGoalsPercentage'] = calculate_shooting_percentages(team_stats_avg, 'fieldGoalsMade', 'fieldGoalsAttempted')
team_stats_avg['threePointersPercentage'] = calculate_shooting_percentages(team_stats_avg, 'threePointersMade', 'threePointersAttempted')
team_stats_avg['freeThrowsPercentage'] = calculate_shooting_percentages(team_stats_avg, 'freeThrowsMade', 'freeThrowsAttempted')

indiv_stats_sum['fieldGoalsPercentage'] = calculate_shooting_percentages(indiv_stats_sum, 'fieldGoalsMade', 'fieldGoalsAttempted')
indiv_stats_sum['threePointersPercentage'] = calculate_shooting_percentages(indiv_stats_sum, 'threePointersMade', 'threePointersAttempted')
indiv_stats_sum['freeThrowsPercentage'] = calculate_shooting_percentages(indiv_stats_sum, 'freeThrowsMade', 'freeThrowsAttempted')

indiv_stats_avg['fieldGoalsPercentage'] = calculate_shooting_percentages(indiv_stats_avg, 'fieldGoalsMade', 'fieldGoalsAttempted')
indiv_stats_avg['threePointersPercentage'] = calculate_shooting_percentages(indiv_stats_avg, 'threePointersMade', 'threePointersAttempted')
indiv_stats_avg['freeThrowsPercentage'] = calculate_shooting_percentages(indiv_stats_avg, 'freeThrowsMade', 'freeThrowsAttempted')


# Adding an "allStar" column to the dataframe
indiv_stats_avg["allStar"] = None
indiv_stats_sum["allStar"] = None

all_star_df["playerName"] = all_star_df["playerName"].apply(lambda x: unidecode.unidecode(x))
indiv_stats_avg["playerName"] = indiv_stats_avg["playerName"].apply(lambda x: unidecode.unidecode(x))
indiv_stats_sum["playerName"] = indiv_stats_sum["playerName"].apply(lambda x: unidecode.unidecode(x)) # Apply normalization to remove special characters for European Names

all_star_df["season_year"] = all_star_df["season_year"].astype(int) 
indiv_stats_avg["season_year"] = indiv_stats_avg["season_year"].astype(int)  
indiv_stats_sum["season_year"] = indiv_stats_sum["season_year"].astype(int) # Ensure column names match for merging

all_star_set = set(zip(all_star_df["playerName"], all_star_df["season_year"])) # Create a dictionary of All-Star selections for quick lookup

indiv_stats_avg["allStar"] = indiv_stats_avg.apply(
    lambda row: 1 if (row["playerName"], row["season_year"]) in all_star_set else 0, axis=1)
indiv_stats_sum["allStar"] = indiv_stats_sum.apply(
    lambda row: 1 if (row["playerName"], row["season_year"]) in all_star_set else 0, axis=1)# Update the allStar column: 1 if player is in All-Star list, otherwise 0


# Rounding the numbers up to 1 decimal places
team_stats_sum = team_stats_sum.round(1)
team_stats_avg = team_stats_avg.round(1)
indiv_stats_avg = indiv_stats_avg.round(1)
indiv_stats_sum = indiv_stats_sum.round(1)


# Downloading season_avg_df into a csv file and storing it under "data/processed"
os.makedirs(processed_path, exist_ok=True)

team_stats_sum.to_csv(os.path.join(processed_path, "team_stats_sum.csv"), index=False)
team_stats_avg.to_csv(os.path.join(processed_path, "team_stats_avg.csv"), index=False)
indiv_stats_sum.to_csv(os.path.join(processed_path, "indiv_stats_sum.csv"), index=False)
indiv_stats_avg.to_csv(os.path.join(processed_path, "indiv_stats_avg.csv"), index=False)

print("data_collection.py ran successfully!!\n")


# to find specific players
specific_stats = indiv_stats_avg[indiv_stats_avg["playerName"] == "Jimmy Butler"]
print(specific_stats)


# # Heat map to show missing values from dataset
# plt.figure(figsize=(12, 6))
# sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
# plt.title("Missing Values Heatmap")
# plt.show()






# IGNORE
# all_star_count = (indiv_stats_avg["allStar"] == 1).sum()
# non_all_star_count = (indiv_stats_avg["allStar"] == 0).sum()
# total_rows = len(indiv_stats_avg) # Cross-checking the number of all stars 

# # print(indiv_stats_avg)
# print(f"\nAll-Star players marked as 1: {all_star_count}")  # Should be 336
# print(f"Non All-Star players marked as 0: {non_all_star_count}")  # Should be 6938
# print(f"Total rows in DataFrame: {total_rows}\n")  # Should be 7274
