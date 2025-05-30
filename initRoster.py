import nfl_data_py as nfl
from localStorage import storePlayerRoster, saveToFiles
import sys
# get roster
year = int(sys.argv[1])
allPlayers = nfl.import_seasonal_rosters([year])

# offensive players (kickers too)
for _, row in allPlayers.iterrows():
    position = row['position']

    if position == 'QB' or position == 'RB' or position == 'WR' or position =='TE' or position == 'K':
        playerId = row['player_id']
        name = row['player_name']
        team = row['team']

        playerInfo = {
            "name": name,
            "position": position,
            "team": team
        }

        storePlayerRoster(playerId, playerInfo)

print("initialized", year, "rosters into local storage")
saveToFiles()