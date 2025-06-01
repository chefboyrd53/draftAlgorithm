import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# Create directory for plots
PLOT_DIR = "model_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# === Load and Process Data ===
with open("local_data/players.json") as f:
    data = json.load(f)

# === Convert JSON to DataFrame with enhanced processing ===
records = []
for player_id, player_data in data.items():
    roster = player_data['roster']
    scoring = player_data.get('scoring', {})
    for season, weeks in scoring.items():
        season_int = int(season)
        season_stats = {
            'player_id': player_id,
            'name': roster['name'],
            'position': roster['position'],
            'team': roster['team'],
            'yearsExp': roster.get('yearsExp', 0),
            'age': roster['age'],
            'status': roster['status'],
            'season': season_int,
            'total_points': 0,
            'games_played': 0,
            'passYards': 0,
            'rushYards': 0,
            'recYards': 0,
            'passTds': 0,
            'rushTds': 0,
            'recTds': 0,
            'attempts': 0,
            'completions': 0,
            'carries': 0,
            'targets': 0,
            'fgm': 0,
            'epm': 0,
            '2pConvs': 0
        }
        for week, stats in weeks.items():
            season_stats['total_points'] += stats.get('points', 0)
            season_stats['games_played'] += 1
            for stat in ['passYards', 'rushYards', 'recYards', 'passTds', 'rushTds', 'recTds',
                         'attempts', 'completions', 'carries', 'targets', 'fgm', 'epm', '2pConvs']:
                season_stats[stat] += stats.get(stat, 0)
        records.append(season_stats)

full_df = pd.DataFrame(records)

# === Feature Engineering ===
def create_advanced_features(df):
    # Per game stats
    df['ppg'] = df['total_points'] / df['games_played'].replace(0, 1)
    df['passYards_pg'] = df['passYards'] / df['games_played'].replace(0, 1)
    df['rushYards_pg'] = df['rushYards'] / df['games_played'].replace(0, 1)
    df['recYards_pg'] = df['recYards'] / df['games_played'].replace(0, 1)
    
    # Efficiency metrics
    df['td_rate'] = df['passTds'] / df['attempts'].replace(0, 1)
    df['completion_rate'] = df['completions'] / df['attempts'].replace(0, 1)
    df['yards_per_attempt'] = df['passYards'] / df['attempts'].replace(0, 1)
    df['yards_per_carry'] = df['rushYards'] / df['carries'].replace(0, 1)
    df['yards_per_target'] = df['recYards'] / df['targets'].replace(0, 1)
    
    # Advanced career metrics
    df['career_games'] = df.groupby('player_id')['games_played'].cumsum()
    df['career_avg_ppg'] = df.groupby('player_id')['ppg'].expanding().mean().shift(1).reset_index(level=0, drop=True)
    
    # Recency-weighted metrics
    def weighted_avg(series, weights=[0.5, 0.3, 0.2]):
        if len(series) == 0:
            return np.nan
        weights = weights[:len(series)]
        return np.sum(series * weights) / np.sum(weights)
    
    df['weighted_ppg'] = df.groupby('player_id')['ppg'].transform(
        lambda x: x.rolling(window=3, min_periods=1).apply(weighted_avg)
    )
    
    # Age and experience analysis
    df['age_squared'] = df['age'] ** 2
    df['exp_squared'] = df['yearsExp'] ** 2
    
    # Injury and durability metrics
    df['games_played_pct'] = df['games_played'] / 17
    df['injury_flag'] = (df['games_played'] < 14).astype(int)
    
    # League-adjusted metrics
    for stat in ['ppg', 'passYards_pg', 'rushYards_pg', 'recYards_pg']:
        df[f'{stat}_league_adj'] = df.groupby('season')[stat].transform(
            lambda x: (x - x.mean()) / x.std()
        )
    
    return df

full_df = create_advanced_features(full_df)

# === PyTorch Dataset and Model ===
class FantasyDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class FantasyNet(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout_rate=0.2):
        super(FantasyNet, self).__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_sizes)-1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def train_position_model(position, df, device='cuda' if torch.cuda.is_available() else 'cpu'):
    pos_df = df[df['position'] == position].copy()
    
    # Position-specific features
    if position == 'QB':
        features = ['ppg', 'passYards_pg', 'td_rate', 'completion_rate', 'yards_per_attempt',
                   'weighted_ppg', 'career_avg_ppg', 'age', 'yearsExp', 'games_played_pct',
                   'ppg_league_adj', 'passYards_pg_league_adj']
    elif position == 'RB':
        features = ['ppg', 'rushYards_pg', 'recYards_pg', 'yards_per_carry', 'yards_per_target',
                   'weighted_ppg', 'career_avg_ppg', 'age', 'yearsExp', 'games_played_pct',
                   'ppg_league_adj', 'rushYards_pg_league_adj']
    elif position == 'WR':
        features = ['ppg', 'recYards_pg', 'yards_per_target', 'weighted_ppg', 'career_avg_ppg',
                   'age', 'yearsExp', 'games_played_pct', 'ppg_league_adj', 'recYards_pg_league_adj']
    elif position == 'TE':
        features = ['ppg', 'weighted_ppg', 'career_avg_ppg', 'age', 'yearsExp', 'games_played_pct',
                   'ppg_league_adj', 'recYards_pg', 'yards_per_target']
    else:
        features = ['ppg', 'weighted_ppg', 'career_avg_ppg', 'age', 'yearsExp', 'games_played_pct',
                   'ppg_league_adj']
    
    # Prepare training data
    train_df = pos_df[pos_df['season'] < 2024].copy()
    future_df = pos_df[pos_df['season'] == 2024].copy()
    
    # Drop rows with NaN in features
    train_df.dropna(subset=features + ['total_points'], inplace=True)
    future_df.dropna(subset=features, inplace=True)
    
    if train_df.empty or len(train_df) < 2:
        print(f"Skipping {position} model training due to insufficient data.")
        return None
    
    X_train = train_df[features]
    y_train = train_df['total_points']
    
    if len(X_train) < 2:
        print(f"Skipping {position} model training due to insufficient data for train-test split.")
        return None
    
    # Train-test split
    X_train_part, X_test_part, y_train_part, y_test_part = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, shuffle=True
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_part)
    X_test_scaled = scaler.transform(X_test_part)
    
    # Create datasets and dataloaders
    train_dataset = FantasyDataset(X_train_scaled, y_train_part.values)
    test_dataset = FantasyDataset(X_test_scaled, y_test_part.values)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = FantasyNet(input_size=len(features)).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    n_epochs = 100
    best_mse = float('inf')
    patience = 10
    patience_counter = 0
    train_losses = []
    test_losses = []
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                test_loss += criterion(outputs.squeeze(), batch_y).item()
        
        train_loss /= len(train_loader)
        test_loss /= len(test_loader)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        scheduler.step(test_loss)
        
        # Early stopping
        if test_loss < best_mse:
            best_mse = test_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f'best_model_{position}.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Validation Loss')
    plt.title(f'Training History - {position}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, f'training_history_{position}.png'))
    plt.close()
    
    # Load best model and evaluate
    model.load_state_dict(torch.load(f'best_model_{position}.pth'))
    model.eval()
    
    with torch.no_grad():
        test_predictions = []
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            test_predictions.extend(outputs.cpu().numpy())
    
    mse = mean_squared_error(y_test_part, test_predictions)
    print(f"{position} Neural Network Model Test MSE: {mse:.2f}")
    
    # Predict future season
    if not future_df.empty:
        X_future = future_df[features]
        X_future_scaled = scaler.transform(X_future)
        future_dataset = FantasyDataset(X_future_scaled, np.zeros(len(X_future_scaled)))
        future_loader = DataLoader(future_dataset, batch_size=32, shuffle=False)
        
        model.eval()
        with torch.no_grad():
            future_predictions = []
            for batch_X, _ in future_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                future_predictions.extend(outputs.cpu().numpy())
        
        future_df['predicted_points_2024'] = future_predictions
        return future_df[['player_id', 'name', 'position', 'team', 'predicted_points_2024']]
    return None

# === Train models for each position and combine predictions ===
positions = full_df['position'].unique()
all_predictions = []

for position in positions:
    print(f"\nTraining neural network model for {position}...")
    predictions = train_position_model(position, full_df)
    if predictions is not None:
        all_predictions.append(predictions)

if all_predictions:
    final_predictions = pd.concat(all_predictions)
    final_predictions = final_predictions.sort_values('predicted_points_2024', ascending=False)
    final_predictions.to_csv("2024_predictions_nn.csv", index=False)
    print("\nSaved 2024 player predictions to 2024_predictions_nn.csv")
else:
    print("No valid predictions could be made for 2024.") 