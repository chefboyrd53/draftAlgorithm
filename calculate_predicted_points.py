import json

rbYearsExp = [0, 36.3, -1.5, -0.6, -10.4, 11.3, -23.8, -24.1, -11.4, -0.8, -0.8, -8.1, -6.4, 4.7, -26.1]
wrYearsExp = [0, 30.9, 14.0, -7.5, 19.6, -7.4, -7.6, -7.3, -9.2, -18.8, -11, 4.4, -15.7, 12.2, -9.8]
teYearsExp = [0, 58.6, -2.5, 3.5, 8.8, -1.0, -18.3, -3.0, -15.8, 10.0, -14.5, 4.9, -14.5, -11.8, 17.9]

def calculate_points_stats(player_data):
    """Calculate points statistics from the 2024 season."""
    total_points = 0
    games_played = 0
    
    if 'scoring' in player_data and '2023' in player_data['scoring']:
        for week_data in player_data['scoring']['2023'].values():
            # Count the game if the week exists in scoring data
            games_played += 1
            total_points += week_data.get('points', 0)
    
    # Calculate average points
    avg_points = total_points / games_played if games_played > 0 else 0
    
    return {
        'total_points': total_points,
        'games_played': games_played,
        'avg_points': avg_points
    }

def get_multiplier(position, years_exp):
    """Get the appropriate multiplier based on position and years of experience."""
    # Convert years of experience to index (0-based)
    index = years_exp
    
    if position == 'RB':
        multipliers = rbYearsExp
    elif position == 'WR':
        multipliers = wrYearsExp
    elif position == 'TE':
        multipliers = teYearsExp
    else:
        return 0  # Default multiplier for other positions
    
    # Ensure index is within bounds
    if index >= len(multipliers):
        index = len(multipliers) - 1
    
    return multipliers[index]

def main():
    # Load the players data
    with open('local_data/players.json', 'r') as f:
        players = json.load(f)
    
    # Create a list to store player data with predicted points
    players_with_predictions = []
    
    # Process each player
    for player_id, player_data in players.items():
        # Get position
        position = player_data['roster']['position']
        
        # Skip players that aren't RB, WR, or TE
        if position not in ['RB', 'WR', 'TE']:
            continue
            
        # Calculate points statistics from last season
        stats = calculate_points_stats(player_data)
        
        # Skip players with less than 7 games played
        if stats['games_played'] < 7:
            continue
        
        # Get years of experience
        years_exp = player_data['roster']['yearsExp']
        
        # Calculate predicted points
        multiplier = get_multiplier(position, years_exp)
        predicted_avg_points = stats['avg_points'] * (1 + multiplier/100)  # Convert percentage to decimal
        predicted_total_points = predicted_avg_points * 14  # Assuming 15-game season
        
        # Create new player data with only roster information
        new_player_data = {
            'roster': {
                'name': player_data['roster']['name'],
                'position': position,
                'team': player_data['roster']['team'],
                'yearsExp': years_exp,
                'age': player_data['roster']['age'],
                'status': player_data['roster']['status'],
                'lastSeasonAvgPoints': round(stats['avg_points'], 2),
                'lastSeasonTotalPoints': round(stats['total_points'], 2),
                'lastSeasonGamesPlayed': stats['games_played'],
                'predictedAvgPoints': round(predicted_avg_points, 2),
                'predictedTotalPoints': round(predicted_total_points, 2)
            }
        }
        
        # Add player to list with their ID and predicted points
        players_with_predictions.append((player_id, new_player_data, predicted_avg_points))
    
    # Sort players by predicted average points in descending order
    players_with_predictions.sort(key=lambda x: x[2], reverse=True)
    
    # Create new sorted dictionary
    sorted_players = {player_id: player_data for player_id, player_data, _ in players_with_predictions}
    
    # Save the updated data to a new file
    with open('local_data/predicted_points.json', 'w') as f:
        json.dump(sorted_players, f, indent=2)

if __name__ == "__main__":
    main() 