"""
feature_engineering.py
======================
PURPOSE: Transform raw match data + Elo ratings into ML-ready features.

WHY THIS FILE EXISTS:
- Machine learning models cannot learn from raw text like team names
- We need to convert everything into numbers
- We need to capture football-specific patterns:
  * Recent form (last 5 matches)
  * Goal scoring ability
  * Goal conceding tendency
  * Home advantage
  * Team strength (Elo + Squad rating)

INPUTS:
  - Raw match DataFrame (from data_loader.py)
  - Pre-match Elo ratings (from elo.py)
  - Squad strength CSV (data/squad_strength.csv)

OUTPUTS:
  - features.csv saved to data/processed/
  - A DataFrame ready for machine learning

FEATURE LIST:
  1.  home_elo              - Home team Elo rating before match
  2.  away_elo              - Away team Elo rating before match
  3.  elo_difference        - home_elo - away_elo
  4.  home_form             - Home team points from last 5 matches (0-1 scale)
  5.  away_form             - Away team points from last 5 matches (0-1 scale)
  6.  home_goals_scored_avg - Home team avg goals scored (last 5)
  7.  home_goals_conceded_avg - Home team avg goals conceded (last 5)
  8.  away_goals_scored_avg - Away team avg goals scored (last 5)
  9.  away_goals_conceded_avg - Away team avg goals conceded (last 5)
  10. home_strength         - Squad strength score for home team
  11. away_strength         - Squad strength score for away team
  12. strength_difference   - home_strength - away_strength
  13. is_neutral            - 1 if neutral venue, 0 if not
  14. target                - Match result (0=Away Win, 1=Draw, 2=Home Win)
"""

import pandas as pd
import numpy as np
from pathlib import Path


class FeatureEngineer:
    """
    Builds a feature matrix from raw match data and Elo ratings.
    
    HOW TO USE:
        fe = FeatureEngineer()
        features_df = fe.build_features(raw_df, pre_match_elo_df, squad_df)
        fe.save_features(features_df)
    """
    
    def __init__(self, window: int = 5):
        """
        Set up the feature engineer.
        
        Parameters:
            window : Number of recent matches to consider for form/goals (default: 5)
                     This means we look at each team's last 5 matches
        """
        self.window = window
        
        # These will be filled as we compute features
        self.team_match_history = {}   # team -> list of match records
    
    # ================================================================
    # TEAM HISTORY BUILDER
    # ================================================================
    
    def _build_team_histories(self, df: pd.DataFrame) -> dict:
        """
        Build a match history for every team sorted by date.
        
        WHY: To compute rolling stats (form, goals), we need to know
        each team's previous matches in order.
        
        For each team we track:
        - Date of match
        - Goals scored
        - Goals conceded
        - Match result (win/draw/loss)
        - Points earned (Win=3, Draw=1, Loss=0)
        
        Parameters:
            df : Raw match DataFrame
        
        Returns:
            dict : {team_name: [list of match records]}
        """
        print("  Building team match histories...")
        
        histories = {}  # team_name -> list of dicts
        
        # Sort by date so history is chronological
        df_sorted = df.sort_values('date').reset_index(drop=True)
        
        for _, row in df_sorted.iterrows():
            home = row['home_team']
            away = row['away_team']
            h_score = row['home_score']
            a_score = row['away_score']
            date = row['date']
            
            # Initialize history lists if team not seen before
            if home not in histories:
                histories[home] = []
            if away not in histories:
                histories[away] = []
            
            # --------------------------------------------------------
            # Determine result for each team
            # --------------------------------------------------------
            if h_score > a_score:
                home_result = 'win'
                away_result = 'loss'
                home_points = 3
                away_points = 0
            elif h_score < a_score:
                home_result = 'loss'
                away_result = 'win'
                home_points = 0
                away_points = 3
            else:
                home_result = 'draw'
                away_result = 'draw'
                home_points = 1
                away_points = 1
            
            # --------------------------------------------------------
            # Record match for home team
            # --------------------------------------------------------
            histories[home].append({
                'date': date,
                'goals_scored': h_score,
                'goals_conceded': a_score,
                'result': home_result,
                'points': home_points
            })
            
            # --------------------------------------------------------
            # Record match for away team
            # --------------------------------------------------------
            histories[away].append({
                'date': date,
                'goals_scored': a_score,
                'goals_conceded': h_score,
                'result': away_result,
                'points': away_points
            })
        
        total_teams = len(histories)
        print(f"  Built histories for {total_teams} teams.")
        
        return histories
    
    # ================================================================
    # ROLLING STATISTICS CALCULATOR
    # ================================================================
    
    def _get_rolling_stats(self, 
                           team: str, 
                           match_date, 
                           histories: dict) -> dict:
        """
        Compute rolling statistics for a team BEFORE a given date.
        
        IMPORTANT: We only look at matches BEFORE match_date.
        This prevents "data leakage" - using future information to 
        predict the past, which would make results unrealistically good.
        
        DATA LEAKAGE EXAMPLE (bad):
            If we computed form using matches AFTER the match we're predicting,
            the model would "cheat" and achieve unrealistically high accuracy.
            In real life, you don't know future results when predicting.
        
        Parameters:
            team       : Team name (e.g., 'Brazil')
            match_date : The date of the match we're predicting (pd.Timestamp)
            histories  : Dict of all team histories
        
        Returns:
            dict : Rolling stats for this team before match_date
        """
        # Default values (used if team has no history yet)
        defaults = {
            'form': 0.5,              # Neutral form (50%)
            'goals_scored_avg': 1.5,  # Average goals scored
            'goals_conceded_avg': 1.5, # Average goals conceded
            'win_rate': 0.33,         # 33% win rate (random baseline)
        }
        
        # If team has no history at all, return defaults
        if team not in histories:
            return defaults
        
        team_history = histories[team]
        
        # Filter: only matches BEFORE this match's date
        past_matches = [
            m for m in team_history 
            if m['date'] < match_date
        ]
        
        # If fewer than 1 past match, return defaults
        if len(past_matches) == 0:
            return defaults
        
        # Take the last N matches (most recent form)
        recent = past_matches[-self.window:]
        
        # --------------------------------------------------------
        # CALCULATE FORM SCORE
        # Form = total points earned / maximum possible points
        # Maximum possible points = window * 3 (all wins)
        # This gives a value between 0 and 1
        # --------------------------------------------------------
        total_points = sum(m['points'] for m in recent)
        max_possible = len(recent) * 3
        form_score = total_points / max_possible if max_possible > 0 else 0.5
        
        # --------------------------------------------------------
        # CALCULATE GOAL AVERAGES
        # --------------------------------------------------------
        goals_scored = [m['goals_scored'] for m in recent]
        goals_conceded = [m['goals_conceded'] for m in recent]
        
        avg_scored = np.mean(goals_scored) if goals_scored else 1.5
        avg_conceded = np.mean(goals_conceded) if goals_conceded else 1.5
        
        # --------------------------------------------------------
        # CALCULATE WIN RATE
        # --------------------------------------------------------
        wins = sum(1 for m in recent if m['result'] == 'win')
        win_rate = wins / len(recent) if recent else 0.33
        
        return {
            'form': round(form_score, 4),
            'goals_scored_avg': round(avg_scored, 4),
            'goals_conceded_avg': round(avg_conceded, 4),
            'win_rate': round(win_rate, 4),
        }
    
    # ================================================================
    # MAIN FEATURE BUILDER
    # ================================================================
    
    def build_features(self,
                       raw_df: pd.DataFrame,
                       pre_match_elo_df: pd.DataFrame,
                       squad_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build the complete feature matrix for machine learning.
        
        Parameters:
            raw_df          : Raw match data from DataLoader
            pre_match_elo_df: Pre-match Elo ratings from EloRatingSystem
            squad_df        : Squad strength scores
        
        Returns:
            pd.DataFrame : Feature matrix ready for ML training
        """
        print("\n" + "=" * 60)
        print("  FEATURE ENGINEERING")
        print("=" * 60)
        
        # Sort by date
        df = raw_df.sort_values('date').reset_index(drop=True)
        
        # ----------------------------------------------------------------
        # STEP A: Prepare squad strength as a dictionary for fast lookup
        # ----------------------------------------------------------------
        # Dictionary lookup is much faster than searching a DataFrame
        # {team_name: strength_score}
        print("\nStep A: Preparing squad strength scores...")
        
        if squad_df is not None and len(squad_df) > 0:
            strength_map = dict(zip(squad_df['team'], squad_df['strength']))
        else:
            strength_map = {}
        
        default_strength = 60.0  # Default for teams not in our list
        print(f"  Loaded strength scores for {len(strength_map)} teams.")
        
        # ----------------------------------------------------------------
        # STEP B: Prepare Elo ratings as a dictionary for fast lookup
        # ----------------------------------------------------------------
        print("\nStep B: Merging Elo ratings...")
        
        # The pre_match_elo_df has one row per match with Elo ratings
        # We need to merge it with our main dataframe
        
        # Make sure date columns are the same type
        df['date'] = pd.to_datetime(df['date'])
        pre_match_elo_df['date'] = pd.to_datetime(pre_match_elo_df['date'])
        
        # Merge on date, home_team, away_team
        df = df.merge(
            pre_match_elo_df[['date', 'home_team', 'away_team', 
                               'home_elo_before', 'away_elo_before', 
                               'elo_difference']],
            on=['date', 'home_team', 'away_team'],
            how='left'  # Keep all rows from df, add Elo where available
        )
        
        # Fill missing Elo values with default
        df['home_elo_before'] = df['home_elo_before'].fillna(1500.0)
        df['away_elo_before'] = df['away_elo_before'].fillna(1500.0)
        df['elo_difference'] = df['elo_difference'].fillna(0.0)
        
        print(f"  Merged Elo ratings. Shape: {df.shape}")
        
        # ----------------------------------------------------------------
        # STEP C: Build team match histories (for rolling stats)
        # ----------------------------------------------------------------
        print("\nStep C: Building team match histories...")
        histories = self._build_team_histories(df)
        
        # ----------------------------------------------------------------
        # STEP D: Compute rolling features for every match
        # ----------------------------------------------------------------
        print("\nStep D: Computing rolling features for every match...")
        print("  This takes 1-3 minutes (processing each match)...")
        
        # Storage for our computed features
        feature_rows = []
        
        total = len(df)
        
        for idx, row in df.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            match_date = row['date']
            
            # Show progress
            if (idx + 1) % 3000 == 0:
                pct = (idx + 1) / total * 100
                print(f"  Progress: {idx+1:,}/{total:,} ({pct:.1f}%)")
            
            # --------------------------------------------------------
            # Get rolling stats for home and away teams
            # --------------------------------------------------------
            home_stats = self._get_rolling_stats(home_team, match_date, histories)
            away_stats = self._get_rolling_stats(away_team, match_date, histories)
            
            # --------------------------------------------------------
            # Get squad strength for both teams
            # --------------------------------------------------------
            home_strength = strength_map.get(home_team, default_strength)
            away_strength = strength_map.get(away_team, default_strength)
            strength_diff = home_strength - away_strength
            
            # --------------------------------------------------------
            # Handle neutral venue flag
            # Some columns may be named differently
            # --------------------------------------------------------
            if 'neutral' in row.index:
                is_neutral = 1 if row['neutral'] == True else 0
            else:
                is_neutral = 0
            
            # --------------------------------------------------------
            # Build the feature row
            # One row = one match = one training example
            # --------------------------------------------------------
            feature_row = {
                # Identifiers (not used for training, but useful for analysis)
                'date': match_date,
                'home_team': home_team,
                'away_team': away_team,
                'tournament': row.get('tournament', 'Unknown'),
                
                # Elo features
                'home_elo': row['home_elo_before'],
                'away_elo': row['away_elo_before'],
                'elo_difference': row['elo_difference'],
                
                # Form features (rolling last 5 matches)
                'home_form': home_stats['form'],
                'away_form': away_stats['form'],
                'form_difference': home_stats['form'] - away_stats['form'],
                
                # Goal features (rolling last 5 matches)
                'home_goals_scored_avg': home_stats['goals_scored_avg'],
                'home_goals_conceded_avg': home_stats['goals_conceded_avg'],
                'away_goals_scored_avg': away_stats['goals_scored_avg'],
                'away_goals_conceded_avg': away_stats['goals_conceded_avg'],
                
                # Win rate features
                'home_win_rate': home_stats['win_rate'],
                'away_win_rate': away_stats['win_rate'],
                
                # Squad strength features
                'home_strength': home_strength,
                'away_strength': away_strength,
                'strength_difference': strength_diff,
                
                # Venue feature
                'is_neutral': is_neutral,
                
                # Raw scores (for reference, NOT used as ML features)
                'home_score': row['home_score'],
                'away_score': row['away_score'],
                
                # TARGET: what we want to predict
                # 0 = Away Win, 1 = Draw, 2 = Home Win
                'target': row['target'],
            }
            
            feature_rows.append(feature_row)
        
        # ----------------------------------------------------------------
        # STEP E: Convert list of dicts to DataFrame
        # ----------------------------------------------------------------
        print("\nStep E: Creating features DataFrame...")
        features_df = pd.DataFrame(feature_rows)
        
        # ----------------------------------------------------------------
        # STEP F: Final cleanup
        # ----------------------------------------------------------------
        print("\nStep F: Final cleanup...")
        
        # Drop rows where target is missing
        features_df = features_df.dropna(subset=['target'])
        
        # Make sure target is integer
        features_df['target'] = features_df['target'].astype(int)
        
        # Sort by date
        features_df = features_df.sort_values('date').reset_index(drop=True)
        
        print(f"\nFeature engineering complete!")
        print(f"  Total matches with features: {len(features_df):,}")
        print(f"  Feature columns: {len(features_df.columns)}")
        
        return features_df
    
    def save_features(self, 
                      df: pd.DataFrame, 
                      filepath: str = "data/processed/features.csv") -> None:
        """
        Save the feature matrix to CSV.
        
        Parameters:
            df       : Feature DataFrame to save
            filepath : Where to save it
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"\nFeatures saved to: {filepath}")
    
    def get_feature_columns(self) -> list:
        """
        Return the list of column names to use for ML training.
        
        WHY THIS MATTERS: We separate feature columns (inputs) from
        identifier columns (date, team names) and the target (output).
        The model only trains on feature columns.
        
        Returns:
            list : Names of feature columns for ML
        """
        return [
            'home_elo',
            'away_elo',
            'elo_difference',
            'home_form',
            'away_form',
            'form_difference',
            'home_goals_scored_avg',
            'home_goals_conceded_avg',
            'away_goals_scored_avg',
            'away_goals_conceded_avg',
            'home_win_rate',
            'away_win_rate',
            'home_strength',
            'away_strength',
            'strength_difference',
            'is_neutral',
        ]
    
    def print_feature_summary(self, df: pd.DataFrame) -> None:
        """
        Print a summary of the feature matrix.
        
        Parameters:
            df : Feature DataFrame
        """
        feature_cols = self.get_feature_columns()
        
        print("\n" + "=" * 60)
        print("  FEATURE SUMMARY")
        print("=" * 60)
        print(f"\nTotal matches: {len(df):,}")
        print(f"Total features: {len(feature_cols)}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        print("\nTarget distribution:")
        target_counts = df['target'].value_counts().sort_index()
        labels = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
        for val, count in target_counts.items():
            pct = count / len(df) * 100
            print(f"  {labels[val]}: {count:,} ({pct:.1f}%)")
        
        print("\nFeature statistics:")
        print("-" * 60)
        summary = df[feature_cols].describe().round(3)
        print(summary.to_string())
        
        print("\nMissing values per feature:")
        missing = df[feature_cols].isnull().sum()
        if missing.sum() == 0:
            print("  No missing values! ✅")
        else:
            print(missing[missing > 0])


# ================================================================
# MAIN BLOCK
# Command: python -m src.feature_engineering
# ================================================================

if __name__ == "__main__":
    
    from src.data_loader import DataLoader
    from src.elo import EloRatingSystem
    
    print("=" * 60)
    print("  FEATURE ENGINEERING - DAY 3 TEST")
    print("=" * 60)
    
    # -----------------------------------------------
    # Step 1: Load raw data
    # -----------------------------------------------
    print("\nStep 1: Loading raw data...")
    loader = DataLoader()
    raw_data = loader.load_raw_data()
    
    # -----------------------------------------------
    # Step 2: Compute Elo ratings
    # -----------------------------------------------
    print("\nStep 2: Computing Elo ratings...")
    elo_system = EloRatingSystem()
    elo_system.compute_ratings(raw_data)
    pre_match_elo = elo_system.get_pre_match_ratings()
    
    print(f"Pre-match Elo records: {len(pre_match_elo):,}")
    
    # -----------------------------------------------
    # Step 3: Load squad strength
    # -----------------------------------------------
    print("\nStep 3: Loading squad strength...")
    squad_df = loader.load_squad_strength()
    print(f"Squad strength loaded: {len(squad_df)} teams")
    
    # -----------------------------------------------
    # Step 4: Build features
    # -----------------------------------------------
    print("\nStep 4: Building features...")
    fe = FeatureEngineer(window=5)
    features = fe.build_features(raw_data, pre_match_elo, squad_df)
    
    # -----------------------------------------------
    # Step 5: Print feature summary
    # -----------------------------------------------
    fe.print_feature_summary(features)
    
    # -----------------------------------------------
    # Step 6: Show sample rows
    # -----------------------------------------------
    print("\n" + "=" * 60)
    print("  SAMPLE FEATURE ROWS (first 5 matches)")
    print("=" * 60)
    
    feature_cols = fe.get_feature_columns()
    display_cols = ['date', 'home_team', 'away_team'] + feature_cols[:6] + ['target']
    print(features[display_cols].head().to_string(index=False))
    
    # -----------------------------------------------
    # Step 7: Show a specific match example
    # -----------------------------------------------
    print("\n" + "=" * 60)
    print("  FEATURE BREAKDOWN FOR ONE MATCH")
    print("=" * 60)
    
    sample = features.iloc[100]
    print(f"\nMatch: {sample['home_team']} vs {sample['away_team']}")
    print(f"Date: {sample['date']}")
    print(f"\nFeatures:")
    for col in feature_cols:
        print(f"  {col:<30} {sample[col]:.4f}")
    
    result_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
    print(f"\nActual Result: {result_map[sample['target']]} (target={sample['target']})")
    
    # -----------------------------------------------
    # Step 8: Save features
    # -----------------------------------------------
    print("\nStep 8: Saving features to disk...")
    fe.save_features(features)
    
    # Also save Elo files if not already saved
    elo_system.save_ratings()
    elo_system.save_pre_match_ratings()
    
    print("\n✅ Day 3 Complete! Feature engineering working correctly.")
    print("\nFiles created:")
    print("  data/processed/features.csv")
    print("  data/processed/elo_ratings.csv")
    print("  data/processed/pre_match_elo.csv")