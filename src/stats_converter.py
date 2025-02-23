import pandas as pd
import os 
import joblib
from sklearn.preprocessing import StandardScaler

information_list = ["firstName", "lastName", "season"]

stats_names_list = [
    "gamesPlayed", "minutes", "points", "assists", "steals", "blocks", 
    "fieldGoalsMade", "fieldGoalsAttempted", "threePointersMade", 
    "freeThrowsMade", "freeThrowsAttempted", "reboundsOffensive",
    "reboundsDefensive", "turnovers", "foulsPersonal", 
    "plusMinusPoints", "USG%"
]

information_data = {}
stats_data = {}

# Use StatMuse for easiest input process
for x in information_list:
    value = (input(f"{x}: "))
    # value = 20000
    information_data[x] = [value]

for x in stats_names_list:
    value = float(input(f"{x}: "))
    # value = 20000
    stats_data[x] = [value]

information = pd.DataFrame(information_data)
firstName = information["firstName"].values[0]
lastName = information["lastName"].values[0]
season = information["season"].values[0]
    
stats = pd.DataFrame(stats_data)


print(firstName)
# print(lastName)
# print(season)

stats["PER"] = (
    (stats['fieldGoalsMade'] * 85.910) +
    (stats['steals'] * 53.897) +
    (stats['threePointersMade'] * 51.757) +
    (stats['freeThrowsMade'] * 46.845) +
    (stats['blocks'] * 39.190) +
    (stats['reboundsOffensive'] * 39.910) +
    (stats['assists'] * 34.677) +
    (stats['reboundsDefensive'] * 14.707) -
    (stats['foulsPersonal'] * 17.174) -
    ((stats['freeThrowsAttempted'] - stats['freeThrowsMade']) * 20.091) -
    ((stats['fieldGoalsAttempted'] - stats['fieldGoalsMade']) * 39.190) -
    (stats['turnovers'] * 53.897)
) / (
    stats['minutes']) # Computing Player Efficiency Rating


# Data used for models
model_data_to_divide = ['freeThrowsMade', 'points', 'fieldGoalsMade', 'turnovers', 'reboundsDefensive', 'assists']

num_season_games = int(input("How many games has this player's team played this season: "))

stats[model_data_to_divide] = stats[model_data_to_divide].div(num_season_games) * 82

print(stats)

scaler = joblib.load(os.path.join("models","scaler_latest.pkl"))
# scaler = StandardScaler()

model_data_to_scale = [
    "points", "freeThrowsMade", "fieldGoalsMade", "plusMinusPoints",
    "minutes", "gamesPlayed", "reboundsDefensive", "assists", "turnovers"
]

stats[model_data_to_scale] = scaler.transform(stats[model_data_to_scale])

#Scale data using scaler and we're ready to use this for model

model_data = [
    'freeThrowsMade', 'points', 'plusMinusPoints', 'fieldGoalsMade', 'PER', 
    'turnovers', 'USG%', 'gamesPlayed','reboundsDefensive', 'minutes', 'assists'
    ]

final_model_stats = stats[model_data]

# team_stats_sum.to_csv(os.path.join(processed_path, "team_stats_sum.csv"), index=False)
file_name = f"{firstName}_{lastName}_{season}_stats.csv"

final_model_stats.to_csv(os.path.join("data/players", file_name), index=False)

print(final_model_stats)