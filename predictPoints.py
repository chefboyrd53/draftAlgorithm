import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import shap
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
import matplotlib.pyplot as plt
import os
warnings.filterwarnings('ignore')

# Create directory for SHAP plots
SHAP_PLOT_DIR = "shap_plots"
os.makedirs(SHAP_PLOT_DIR, exist_ok=True)

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

# === League-wide Trend Analysis ===
def calculate_league_trends(df, stat_col, window=3):
    league_avg = df.groupby('season')[stat_col].mean()
    trend = league_avg.rolling(window=window, min_periods=1).mean()
    return trend

# Calculate league-wide trends for key metrics
league_trends = {}
for stat in ['passYards', 'rushYards', 'recYards', 'total_points']:
    league_trends[stat] = calculate_league_trends(full_df, stat)

# === Advanced Feature Engineering ===
def create_advanced_features(df):
    # Per game stats with league adjustment
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
    
    # Recency-weighted metrics (more weight to recent seasons)
    def weighted_avg(series, weights=[0.5, 0.3, 0.2]):
        if len(series) == 0:
            return np.nan
        weights = weights[:len(series)]
        return np.sum(series * weights) / np.sum(weights)
    
    # Calculate weighted PPG using rolling window
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

# === Position-specific Modeling ===
def train_position_model(position, df):
    pos_df = df[df['position'] == position].copy()
    
    # Define position-specific parameters and features
    if position == 'QB':
        features = ['ppg', 'passYards_pg', 'td_rate', 'completion_rate', 'yards_per_attempt',
                   'weighted_ppg', 'career_avg_ppg', 'age', 'yearsExp', 'games_played_pct',
                   'ppg_league_adj', 'passYards_pg_league_adj']
        params = {
            'n_estimators': 2000, 
            'learning_rate': 0.01,
            'num_leaves': 48,
            'max_depth': 7, 
            'min_child_samples': 15,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_split_gain': 0.05, 
            'min_child_weight': 0.5,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'verbose': -1
        }
        early_stopping_rounds = 100 
    elif position == 'RB':
        features = ['ppg', 'rushYards_pg', 'recYards_pg', 'yards_per_carry', 'yards_per_target',
                   'weighted_ppg', 'career_avg_ppg', 'age', 'yearsExp', 'games_played_pct',
                   'ppg_league_adj', 'rushYards_pg_league_adj']
        params = {
            'n_estimators': 2000,
            'learning_rate': 0.01,
            'num_leaves': 48,
            'max_depth': 7,
            'min_child_samples': 15,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_split_gain': 0.05,
            'min_child_weight': 0.5,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'verbose': -1
        }
        early_stopping_rounds = 100 
    elif position == 'WR':
        features = ['ppg', 'recYards_pg', 'yards_per_target', 'weighted_ppg', 'career_avg_ppg',
                   'age', 'yearsExp', 'games_played_pct', 'ppg_league_adj', 'recYards_pg_league_adj']
        params = {
            'n_estimators': 1500, 
            'learning_rate': 0.01,
            'num_leaves': 40, 
            'max_depth': 6, 
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_split_gain': 0.1,
            'min_child_weight': 1,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'verbose': -1
        }
        early_stopping_rounds = 50
    elif position == 'TE':
        features = ['ppg', 'weighted_ppg', 'career_avg_ppg', 'age', 'yearsExp', 'games_played_pct',
                   'ppg_league_adj', 'recYards_pg', 'yards_per_target'] 
        params = {
            'n_estimators': 1500, 
            'learning_rate': 0.01,
            'num_leaves': 40, 
            'max_depth': 6, 
            'min_child_samples': 15,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_split_gain': 0.05, 
            'min_child_weight': 0.5, 
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'verbose': -1
        }
        early_stopping_rounds = 75 
    else: # For positions with less data or different needs
        features = ['ppg', 'weighted_ppg', 'career_avg_ppg', 'age', 'yearsExp', 'games_played_pct',
                   'ppg_league_adj']
        params = {
            'n_estimators': 1000,
            'learning_rate': 0.01,
            'num_leaves': 31,
            'max_depth': 5,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_split_gain': 0.1,
            'min_child_weight': 1,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'verbose': -1
        }
        early_stopping_rounds = 50
    
    # Prepare training data
    train_df = pos_df[pos_df['season'] < 2023].copy()
    future_df = pos_df[pos_df['season'] == 2023].copy()
    
    # Drop rows with NaN in features *after* creating future_df
    train_df.dropna(subset=features + ['total_points'], inplace=True)
    future_df.dropna(subset=features, inplace=True)
    
    # Skip modeling if not enough data after dropping NaNs or for train-test split
    if train_df.empty or len(train_df) < 2:
        print(f"Skipping {position} model training due to insufficient data.")
        return None
            
    X_train = train_df[features]
    y_train = train_df['total_points']
    
    # Ensure enough samples for split
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
    
    # Train model with position-specific parameters
    model = lgb.LGBMRegressor(**params)
    
    model.fit(
        X_train_scaled, y_train_part,
        eval_set=[(X_test_scaled, y_test_part)],
        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds), lgb.log_evaluation(100)]
    )
    
    # Evaluate
    y_pred_test = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test_part, y_pred_test)
    print(f"{position} Model Test MSE: {mse:.2f}")
    
    # SHAP analysis with improved visualization
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test_scaled)
    
    # Create a figure for SHAP summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_part, show=False)
    plt.title(f'SHAP Summary Plot - {position}')
    plt.tight_layout()
    plt.savefig(os.path.join(SHAP_PLOT_DIR, f'shap_summary_{position}.png'))
    plt.close()
    
    # Create a figure for SHAP dependence plots for top features
    # Ensure there are enough features for dependence plots
    if len(features) > 0:
        top_features = np.abs(shap_values.values).mean(0).argsort()[-min(len(features), 3):][::-1]
        for idx in top_features:
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(idx, shap_values.values, X_test_scaled, 
                               feature_names=features, show=False)
            plt.title(f'SHAP Dependence Plot - {position} - {features[idx]}')
            plt.tight_layout()
            plt.savefig(os.path.join(SHAP_PLOT_DIR, f'shap_dependence_{position}_{features[idx]}.png'))
            plt.close()
    else:
        print(f"Not enough features for dependence plots for {position}")
    
    # Predict future season
    if not future_df.empty:
        X_future = future_df[features]
        
        # Skip prediction if no data after dropping NaNs
        if X_future.empty:
            print(f"Skipping {position} future prediction due to insufficient data after dropping NaNs.")
            return None
            
        X_future_scaled = scaler.transform(X_future)
        preds_2025 = model.predict(X_future_scaled)
        future_df['predicted_points_2025'] = preds_2025
        return future_df[['player_id', 'name', 'position', 'team', 'predicted_points_2025']]
    return None

# === Train models for each position and combine predictions ===
positions = full_df['position'].unique()
all_predictions = []

for position in positions:
    print(f"\nTraining model for {position}...")
    predictions = train_position_model(position, full_df)
    if predictions is not None:
        all_predictions.append(predictions)

if all_predictions:
    final_predictions = pd.concat(all_predictions)
    final_predictions = final_predictions.sort_values('predicted_points_2025', ascending=False)
    final_predictions.to_csv("2025_predictions.csv", index=False)
    print("\nSaved 2025 player predictions to 2025_predictions.csv")
else:
    print("No valid predictions could be made for 2025.")
