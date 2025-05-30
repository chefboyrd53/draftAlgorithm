import nfl_data_py as nfl
from localStorage import storePlayerRoster, saveToFiles
import sys
import math

# get roster
years = []
for i in range(1, len(sys.argv)):
    years.append(int(sys.argv[i]))
allPlayers = nfl.import_seasonal_rosters(years)

# offensive players (kickers too)
for _, row in allPlayers.iterrows():
    position = row['position']

    if position == 'QB' or position == 'RB' or position == 'WR' or position =='TE' or position == 'K':
        playerId = row['player_id']
        name = row['player_name']
        team = row['team']
        yearsExp = row['years_exp']
        age = row['age']
        status = row['status']

        playerInfo = {
            "name": name,
            "position": position,
            "team": team,
            "yearsExp": yearsExp,
            "age": age,
            "status": status
        }

        if math.isnan(age): 
            continue
        storePlayerRoster(playerId, playerInfo)

print("initialized", years, "rosters into local storage")
saveToFiles()