"""
predictor.py
============
PURPOSE: Wrap the trained Random Forest model to predict any match.

WHY THIS FILE EXISTS:
- The simulator needs to predict hundreds of matches per simulation
- This file provides a clean, reusable interface to the model
- Handles feature building for any two teams on the fly
- Uses current Elo ratings and team statistics

INPUTS:  Trained rf_model.pkl + current Elo ratings + team stats
OUTPUTS: Match win/draw/loss probabilities as numpy array
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path


class MatchPredictor:
    """
    Predicts match outcomes using the trained Random Forest model.

    HOW TO USE:
        predictor = MatchPredictor()
        predictor.load_model()
        predictor.load_team_data(elo_ratings_df, features_df)
        probs = predictor.predict_match('Brazil', 'France', neutral=True)
        # probs = [away_win_prob, draw_prob, home_win_prob]
    """

    def __init__(self, model_dir: str = "models"):
        """
        Set up file paths.

        Parameters:
            model_dir : Folder containing saved model files
        """
        self.model_dir = Path(model_dir)
        self.model = None
        self.feature_columns = None

        # Team data lookups (filled when load_team_data() is called)
        self.elo_ratings = {}       # team -> current elo rating
        self.team_form = {}         # team -> recent form score
        self.team_goals_scored = {} # team -> avg goals scored
        self.team_goals_conceded = {}# team -> avg goals conceded
        self.team_win_rate = {}     # team -> win rate
        self.team_strength = {}     # team -> squad strength

        # Default values for teams with no data
        self.default_elo = 1500.0
        self.default_form = 0.5
        self.default_goals = 1.5
        self.default_win_rate = 0.33
        self.default_strength = 60.0

    # ================================================================
    # MODEL LOADING
    # ================================================================

    def load_model(self, filename: str = "rf_model.pkl") -> None:
        """
        Load the trained Random Forest model from disk.

        Parameters:
            filename : Model file name
        """
        model_path = self.model_dir / filename
        feature_path = self.model_dir / "feature_columns.txt"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Please run Day 4 first: python -m src.train_model"
            )

        self.model = joblib.load(model_path)
        print(f"Model loaded from: {model_path}")

        # Load feature column names
        if feature_path.exists():
            with open(feature_path, 'r') as f:
                self.feature_columns = [
                    line.strip() for line in f.readlines()
                ]
            print(f"Feature columns loaded: {len(self.feature_columns)} features")
        else:
            # Fallback: hardcode the feature columns
            self.feature_columns = [
                'home_elo', 'away_elo', 'elo_difference',
                'home_form', 'away_form', 'form_difference',
                'home_goals_scored_avg', 'home_goals_conceded_avg',
                'away_goals_scored_avg', 'away_goals_conceded_avg',
                'home_win_rate', 'away_win_rate',
                'home_strength', 'away_strength',
                'strength_difference', 'is_neutral',
            ]

    # ================================================================
    # TEAM DATA LOADING
    # ================================================================

    def load_team_data(self,
                       elo_df: pd.DataFrame,
                       features_df: pd.DataFrame,
                       squad_df: pd.DataFrame = None) -> None:
        """
        Load team statistics from computed data.

        We extract the MOST RECENT statistics for each team
        from the features dataframe. This gives us each team's
        current form, goals, and win rate going into the tournament.

        Parameters:
            elo_df      : DataFrame with columns [team, elo]
            features_df : Full feature DataFrame from Day 3
            squad_df    : DataFrame with columns [team, strength]
        """
        print("Loading team data for predictions...")

        # --------------------------------------------------------
        # Load Elo ratings into a dictionary
        # --------------------------------------------------------
        self.elo_ratings = dict(zip(elo_df['team'], elo_df['elo']))
        print(f"  Loaded Elo ratings for {len(self.elo_ratings)} teams")

        # --------------------------------------------------------
        # Load squad strength
        # --------------------------------------------------------
        if squad_df is not None and len(squad_df) > 0:
            self.team_strength = dict(
                zip(squad_df['team'], squad_df['strength'])
            )
        print(f"  Loaded strength for {len(self.team_strength)} teams")

        # --------------------------------------------------------
        # Extract most recent stats per team from features_df
        # We look at each team's most recent HOME and AWAY matches
        # and take their latest rolling statistics
        # --------------------------------------------------------
        features_df = features_df.sort_values('date')

        # Get most recent stats for home teams
        home_latest = features_df.groupby('home_team').last().reset_index()
        away_latest = features_df.groupby('away_team').last().reset_index()

        # Build form dictionary from home perspective
        for _, row in home_latest.iterrows():
            team = row['home_team']
            self.team_form[team] = row.get('home_form', self.default_form)
            self.team_goals_scored[team] = row.get(
                'home_goals_scored_avg', self.default_goals)
            self.team_goals_conceded[team] = row.get(
                'home_goals_conceded_avg', self.default_goals)
            self.team_win_rate[team] = row.get(
                'home_win_rate', self.default_win_rate)

        # Fill gaps with away perspective data
        for _, row in away_latest.iterrows():
            team = row['away_team']
            if team not in self.team_form:
                self.team_form[team] = row.get(
                    'away_form', self.default_form)
                self.team_goals_scored[team] = row.get(
                    'away_goals_scored_avg', self.default_goals)
                self.team_goals_conceded[team] = row.get(
                    'away_goals_conceded_avg', self.default_goals)
                self.team_win_rate[team] = row.get(
                    'away_win_rate', self.default_win_rate)

        print(f"  Loaded stats for {len(self.team_form)} teams")
        print("Team data loaded successfully!")

    # ================================================================
    # FEATURE BUILDING
    # ================================================================

    def _build_match_features(self,
                               home_team: str,
                               away_team: str,
                               neutral: bool = True) -> np.ndarray:
        """
        Build feature vector for a match between two teams.

        This recreates the same features we used during training,
        but using CURRENT team statistics (not historical).

        Parameters:
            home_team : Name of the home team (or team1 in neutral)
            away_team : Name of the away team (or team2 in neutral)
            neutral   : True if played at neutral venue

        Returns:
            np.ndarray : Feature vector ready for model prediction
        """
        # Get Elo ratings
        home_elo = self.elo_ratings.get(home_team, self.default_elo)
        away_elo = self.elo_ratings.get(away_team, self.default_elo)
        elo_diff = home_elo - away_elo

        # Get form scores
        home_form = self.team_form.get(home_team, self.default_form)
        away_form = self.team_form.get(away_team, self.default_form)
        form_diff = home_form - away_form

        # Get goal statistics
        home_gs = self.team_goals_scored.get(
            home_team, self.default_goals)
        home_gc = self.team_goals_conceded.get(
            home_team, self.default_goals)
        away_gs = self.team_goals_scored.get(
            away_team, self.default_goals)
        away_gc = self.team_goals_conceded.get(
            away_team, self.default_goals)

        # Get win rates
        home_wr = self.team_win_rate.get(
            home_team, self.default_win_rate)
        away_wr = self.team_win_rate.get(
            away_team, self.default_win_rate)

        # Get squad strength
        home_str = self.team_strength.get(
            home_team, self.default_strength)
        away_str = self.team_strength.get(
            away_team, self.default_strength)
        str_diff = home_str - away_str

        # Neutral venue flag
        is_neutral = 1 if neutral else 0

        # Build feature vector in EXACT same order as training
        features = {
            'home_elo': home_elo,
            'away_elo': away_elo,
            'elo_difference': elo_diff,
            'home_form': home_form,
            'away_form': away_form,
            'form_difference': form_diff,
            'home_goals_scored_avg': home_gs,
            'home_goals_conceded_avg': home_gc,
            'away_goals_scored_avg': away_gs,
            'away_goals_conceded_avg': away_gc,
            'home_win_rate': home_wr,
            'away_win_rate': away_wr,
            'home_strength': home_str,
            'away_strength': away_str,
            'strength_difference': str_diff,
            'is_neutral': is_neutral,
        }

        return np.array([
            features[col] for col in self.feature_columns
        ]).reshape(1, -1)

    # ================================================================
    # MATCH PREDICTION
    # ================================================================

    def predict_match(self,
                      team_a: str,
                      team_b: str,
                      neutral: bool = True) -> np.ndarray:
        """
        Predict win probabilities for a match.

        Parameters:
            team_a  : First team (treated as home/team1)
            team_b  : Second team (treated as away/team2)
            neutral : True = neutral venue (World Cup matches)

        Returns:
            np.ndarray : [away_win_prob, draw_prob, home_win_prob]
                         i.e. [team_b wins, draw, team_a wins]
        """
        if self.model is None:
            raise ValueError("Load model first using load_model()")

        features = self._build_match_features(team_a, team_b, neutral)
        probs = self.model.predict_proba(features)[0]

        return probs  # [P(away win), P(draw), P(home win)]

    def predict_winner(self,
                       team_a: str,
                       team_b: str,
                       neutral: bool = True,
                       allow_draw: bool = False) -> str:
        """
        Randomly sample a winner based on predicted probabilities.

        WHY RANDOM SAMPLING?
        In simulation we don't always pick the most likely outcome.
        We randomly choose based on probabilities. This means:
        - Strong teams WIN more often (as expected)
        - But upsets CAN happen (just like real football)

        Parameters:
            team_a      : First team
            team_b      : Second team
            neutral     : Neutral venue flag
            allow_draw  : If True, draws are possible (group stage)
                          If False, one team must win (knockout)

        Returns:
            str : 'team_a', 'team_b', or 'draw'
        """
        probs = self.predict_match(team_a, team_b, neutral)

        # probs = [away_win, draw, home_win]
        # home = team_a, away = team_b
        p_team_a_wins = probs[2]  # home win = team_a wins
        p_draw = probs[1]
        p_team_b_wins = probs[0]  # away win = team_b wins

        if allow_draw:
            # Group stage: draw is a valid result
            outcomes = ['team_a', 'draw', 'team_b']
            chosen = np.random.choice(
                outcomes,
                p=[p_team_a_wins, p_draw, p_team_b_wins]
            )
        else:
            # Knockout: no draw allowed, redistribute draw probability
            # Split draw probability proportionally between win/loss
            total = p_team_a_wins + p_team_b_wins
            if total == 0:
                p_a_adj = 0.5
                p_b_adj = 0.5
            else:
                p_a_adj = p_team_a_wins / total
                p_b_adj = p_team_b_wins / total

            outcomes = ['team_a', 'team_b']
            chosen = np.random.choice(
                outcomes, p=[p_a_adj, p_b_adj]
            )

        return chosen
    