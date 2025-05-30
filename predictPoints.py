import json

rbYearsExp = [0, 36.3, -1.5, -0.6, -10.4, 11.3, -23.8, -24.1, -11.4, -0.8, -0.8, -8.1, -6.4, 4.7, -26.1]
wrYearsExp = [0, 30.9, 14.0, -7.5, 19.6, -7.4, -7.6, -7.3, -9.2, -18.8, 4.4, -15.7, 12.2, -9.8]
teYearsExp = [0, 58.6, -2.5, 3.5, 8.8, -1.0, -18.3, -3.0, -15.8, 10.0, -14.5, 4.9, -14.5, -11.8, 17.9]

def calculate_total_points(player_data):
    """Calculate total points from the 2024 season."""
    total_points = 0
    if 'scoring' in player_data and '2024' in player_data['scoring']:
        for week_data in player_data['scoring']['2024'].values():
            total_points += week_data.get('points', 0)
    return total_points

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
        return 1.0  # Default multiplier for other positions
    
    # Ensure index is within bounds
    if index >= len(multipliers):
        index = len(multipliers) - 1
    
    return multipliers[index]

def main():
    # Load the players data
    with open('local_data/players.json', 'r') as f:
        players = json.load(f)
    
    # Process each player
    for player_id, player_data in players.items():
        # Calculate total points from last season
        total_points = calculate_total_points(player_data)
        
        # Get position and years of experience
        position = player_data['roster']['position']
        years_exp = player_data['roster']['yearsExp']
        
        # Calculate predicted points
        multiplier = get_multiplier(position, years_exp)
        predicted_points = total_points * (1 + multiplier/100)  # Convert percentage to decimal
        
        # Add predicted points to player data
        if 'scoring' not in player_data:
            player_data['scoring'] = {}
        if '2024' not in player_data['scoring']:
            player_data['scoring']['2024'] = {}
        
        # Add predicted points to the first week of 2024
        if '1' not in player_data['scoring']['2024']:
            player_data['scoring']['2024']['1'] = {}
        player_data['scoring']['2024']['1']['predictedPoints'] = round(predicted_points, 2)
    
    # Save the updated data
    with open('local_data/players.json', 'w') as f:
        json.dump(players, f, indent=2)

if __name__ == "__main__":
    main() 