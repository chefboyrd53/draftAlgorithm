from localStorage import storePlayerData, getPlayerData, getPlayerRoster

def score(touchdownRows, week, year):
    for _, row in touchdownRows.iterrows():
        passerScore = 0
        carrierScore = 0

        if row['pass_touchdown']:
            passerId = row['passer_player_id']
            receiverId = row['td_player_id']
            yards = row['yards_gained']

            if yards <= 9:
                carrierScore = 6
                passerScore = 6
            elif 10 <= yards <= 39:
                carrierScore = 9
                passerScore = 9
            elif 40 <= yards <= 69:
                carrierScore = 12
                passerScore = 12
            elif 70 <= yards:
                carrierScore = 15
                passerScore = 15
            
            passerPosition = getPlayerRoster(passerId).get("position")
            receiverPosition = getPlayerRoster(receiverId).get("position")

            if passerPosition != 'QB':
                passerScore *= 2
            if receiverPosition != 'WR' and receiverPosition != 'TE':
                carrierScore *= 2

            # Update passer data
            passer_data = getPlayerData(passerId, year, week)
            passer_data.update({
                "points": passer_data.get("points", 0) + passerScore,
                "passTds": passer_data.get("passTds", 0) + 1
            })
            storePlayerData(passerId, year, week, passer_data)

            # Update receiver data
            receiver_data = getPlayerData(receiverId, year, week)
            receiver_data.update({
                "points": receiver_data.get("points", 0) + carrierScore,
                "recTds": receiver_data.get("recTds", 0) + 1
            })
            storePlayerData(receiverId, year, week, receiver_data)

        elif row['rush_touchdown']:
            rusherId = row['td_player_id']
            yards = row['yards_gained']

            if yards <= 9:
                carrierScore = 6
            elif 10 <= yards <= 39:
                carrierScore = 9
            elif 40 <= yards <= 69:
                carrierScore = 12
            elif 70 <= yards:
                carrierScore = 15

            position = getPlayerRoster(rusherId).get("position")
            if position != 'RB':
                carrierScore *= 2

            # Update rusher data
            rusher_data = getPlayerData(rusherId, year, week)
            rusher_data.update({
                "points": rusher_data.get("points", 0) + carrierScore,
                "rushTds": rusher_data.get("rushTds", 0) + 1
            })
            storePlayerData(rusherId, year, week, rusher_data)
