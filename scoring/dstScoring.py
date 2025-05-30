from localStorage import storeDefenseData, getDefenseData
import re

def getLateralReturnYards(description):
    # Normalize case
    desc = description.lower()
    if 'lateral' not in desc:
        return None  # No lateral, skip

    # Find all "to [Team Abbrev] [Yard Line] for X yards" patterns
    yard_chunks = re.findall(r'for (\d+) yards?', desc)
    
    # Convert to integers and sum
    if yard_chunks:
        return sum(int(y) for y in yard_chunks)
    
    return None

def scoreDST(playByPlaydf, week, year):
    # defensive touchdown
    touchdownRows = playByPlaydf[(playByPlaydf['return_touchdown'] == 1) & (playByPlaydf['week'] == week)]
    for _, row in touchdownRows.iterrows():
        defTeam = row['td_team']
        score = 10

        defData = getDefenseData(defTeam, year, week)
        defData.update({
            "points": defData.get("points", 0) + score,
            "touchdowns": defData.get("touchdowns", 0) + 1
        })
        storeDefenseData(defTeam, year, week, defData)


    # turnovers
    turnoverRows = playByPlaydf[
        (playByPlaydf['week'] == week) & (
            (playByPlaydf['interception'] == 1) |
            (
                (playByPlaydf['fumble_lost'] == 1) & (
                    (playByPlaydf['fumble_recovery_1_team'] == playByPlaydf['defteam']) |
                    (
                        # Special teams muff recovery:
                        (playByPlaydf['kickoff_attempt'] == 1) |
                        (playByPlaydf['punt_attempt'] == 1)
                    ) &
                    (playByPlaydf['fumble_recovery_1_team'] == playByPlaydf['posteam'])  # posteam = kicking team on kicks
                )
            )
        )
    ]
    for _, row in turnoverRows.iterrows():
        # Use fumble recovery team as the defense if it's on special teams
        if row['fumble_lost'] == 1 and row['fumble_recovery_1_team'] != row['defteam']:
            defTeam = row['fumble_recovery_1_team']
        else:
            defTeam = row['defteam']
        score = 2

        defData = getDefenseData(defTeam, year, week)
        defData.update({
            "points": defData.get("points", 0) + score,
            "turnovers": defData.get("turnovers", 0) + 1
        })
        storeDefenseData(defTeam, year, week, defData)

    # sacks
    sackRows = playByPlaydf[(playByPlaydf['sack'] == 1) & (playByPlaydf['week'] == week)]
    for _, row in sackRows.iterrows():
        defTeam = row['defteam']
        score = 1

        defData = getDefenseData(defTeam, year, week)
        defData.update({
            "points": defData.get("points", 0) + score,
            "sacks": defData.get("sacks", 0) + 1
        })
        storeDefenseData(defTeam, year, week, defData)

    # safety
    safetyRows = playByPlaydf[(playByPlaydf['safety'] == 1) & (playByPlaydf['week'] == week)]
    for _, row in safetyRows.iterrows():
        defTeam = row['defteam']
        score = 12

        defData = getDefenseData(defTeam, year, week)
        defData.update({
            "points": defData.get("points", 0) + score,
            "safeties": defData.get("safeties", 0) + 1
        })
        storeDefenseData(defTeam, year, week, defData)

    # defense blocks PAT and returns it for PAT, defense intercepts or recovers fumble of 2 pt conversion and returns it
    specialRows = playByPlaydf[(playByPlaydf['defensive_extra_point_conv'] == 1) & (playByPlaydf['week'] == week)]
    for _, row in specialRows.iterrows():
        defTeam = row['defteam']
        score = 12

        defData = getDefenseData(defTeam, year, week)
        defData.update({
            "points": defData.get("points", 0) + score,
            "returned2pts": defData.get("returned2pts", 0) + 1
        })
        storeDefenseData(defTeam, year, week, defData)

    # combined punt + kickoff return yards (no point updates)
    returnRows = playByPlaydf[((playByPlaydf['punt_attempt'] == 1) | (playByPlaydf['kickoff_attempt'] == 1)) & (playByPlaydf['week'] == week)]
    for _, row in returnRows.iterrows():
        if row['return_team'] is not None:
            returnTeam = row['return_team']
            yards = row['return_yards']
            
            if row['lateral_return'] == 1:   
                description = row['desc']
                lateralYards = getLateralReturnYards(description)
                if lateralYards is not None:
                    yards = lateralYards

            defData = getDefenseData(returnTeam, year, week)
            defData.update({
                "returnYards": defData.get("returnYards", 0) + yards
            })
            storeDefenseData(returnTeam, year, week, defData)

    # end of game check for points allowed and add points for return yards
    weekGames = playByPlaydf[(playByPlaydf['week'] == week)]
    finalPlays = weekGames.sort_values(['game_id', 'play_id']).groupby('game_id').tail(1)
    for _, row in finalPlays.iterrows():
        homeTeam = row['home_team']
        awayTeam = row['away_team']
        homeTeamPoints = row['home_score']
        awayTeamPoints = row['away_score']
        homeScore = 0
        awayScore = 0
        
        if homeTeamPoints == 0:
            awayScore = 12
        elif 1 <= homeTeamPoints <= 3:
            awayScore = 9
        elif 4 <= homeTeamPoints <= 6:
            awayScore = 6
        elif 7 <= homeTeamPoints <= 10:
            awayScore = 3

        if awayTeamPoints == 0:
            homeScore = 12
        elif 1 <= awayTeamPoints <= 3:
            homeScore = 9
        elif 4 <= awayTeamPoints <= 6:
            homeScore = 6
        elif 7 <= awayTeamPoints <= 10:
            homeScore = 3

        # Update home team data
        homeData = getDefenseData(homeTeam, year, week)
        homeData.update({
            "points": homeData.get("points", 0) + homeScore,
            "pointsAllowed": awayTeamPoints
        })
        storeDefenseData(homeTeam, year, week, homeData)

        # Update away team data
        awayData = getDefenseData(awayTeam, year, week)
        awayData.update({
            "points": awayData.get("points", 0) + awayScore,
            "pointsAllowed": homeTeamPoints
        })
        storeDefenseData(awayTeam, year, week, awayData)

        # Add points for return yards
        totalReturnYards = homeData.get("returnYards", 0)
        if 75 <= totalReturnYards <= 99:
            homeData["points"] = homeData.get("points", 0) + 3
        elif 100 <= totalReturnYards <= 149:
            homeData["points"] = homeData.get("points", 0) + 6
        elif 150 <= totalReturnYards:
            homeData["points"] = homeData.get("points", 0) + 9
        storeDefenseData(homeTeam, year, week, homeData)

        totalReturnYards = awayData.get("returnYards", 0)
        if 75 <= totalReturnYards <= 99:
            awayData["points"] = awayData.get("points", 0) + 3
        elif 100 <= totalReturnYards <= 149:
            awayData["points"] = awayData.get("points", 0) + 6
        elif 150 <= totalReturnYards:
            awayData["points"] = awayData.get("points", 0) + 9
        storeDefenseData(awayTeam, year, week, awayData)
    

    


