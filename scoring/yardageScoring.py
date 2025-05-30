from localStorage import storePlayerData

def scoreYardage(playerRows, week, year):
    for _, row in playerRows.iterrows():
        score = 0
        playerId = row['player_id']
        passYds = row['passing_yards']
        rushYds = row['rushing_yards']
        recYds = row['receiving_yards']
        numTwoPoints = row['passing_2pt_conversions'] + row['rushing_2pt_conversions'] + row['receiving_2pt_conversions']

        # passing yards
        if 200 <= passYds <= 299:
            score += 6
        elif 300 <= passYds <= 399:
            score += 9
        elif 400 <= passYds <= 499:
            score += 12
        elif 500 <= passYds:
            score += 15

        # rushing yards
        if 50 <= rushYds <= 74:
            score += 3
        elif 75 <= rushYds <= 99:
            score += 6
        elif 100 <= rushYds <= 124:
            score += 9
        elif 125 <= rushYds <= 150:
            score += 12
        elif 150 <= rushYds <= 199:
            score += 15
        elif 200 <= rushYds:
            score += 18

        # receiving yards
        if 50 <= recYds <= 74:
            score += 3
        elif 75 <= recYds <= 99:
            score += 6
        elif 100 <= recYds <= 124:
            score += 9
        elif 125 <= recYds <= 150:
            score += 12
        elif 150 <= recYds <= 199:
            score += 15
        elif 200 <= recYds:
            score += 18

        # combined yardage
        if rushYds >= 20 and recYds >= 20 and (rushYds + recYds) >= 150:
            score += 6

        # 2 point conversions
        score += (numTwoPoints * 2)

        playerData = {
            "points": score,
            "passYards": passYds,
            "rushYards": rushYds,
            "recYards": recYds,
            "passTds": 0,
            "rushTds": 0,
            "recTds": 0,
            "fgm": 0,
            "epm": 0,
            "2pConvs": numTwoPoints
        }
        storePlayerData(playerId, year, week, playerData)