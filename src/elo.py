"""
elo.py
======
PURPOSE: Compute Elo ratings for every national football team.

WHY THIS FILE EXISTS:
- Raw data only tells us scores. It doesn't tell us HOW STRONG each team is.
- Elo ratings give us a single number representing team strength at any point in time.
- This becomes one of our most important features for machine learning.

HOW ELO WORKS:
- Every team starts at 1500
- Win against strong team = big rating gain
- Win against weak team = small rating gain
- Lose to strong team = small rating loss
- Lose to weak team = big rating loss

INPUTS:  Raw match data (results.csv loaded by data_loader.py)
OUTPUTS: Dictionary of {team_name: elo_rating} + CSV file saved to disk
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path


class EloRatingSystem:
    """
    Computes and updates Elo ratings for national football teams.
    
    HOW TO USE:
        elo = EloRatingSystem()
        elo.compute_ratings(match_dataframe)
        print(elo.get_current_ratings())
    """
    
    def __init__(self, initial_rating: int = 1500, k_factor: int = 40):
        """
        Set up the Elo system with starting values.
        
        Parameters:
            initial_rating : Starting rating for every team (1500 = average)
            k_factor       : How much ratings change per match (40 = standard)
        
        defaultdict means: if a team is not in the dictionary yet,
        automatically give it the initial_rating value.
        This saves us from writing "if team not in dict" every time.
        """
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        
        # Main storage: team name -> current rating
        # defaultdict automatically creates entry with initial_rating for new teams
        self.ratings = defaultdict(lambda: float(initial_rating))
        
        # History storage: team name -> list of (date, rating) tuples
        # This lets us track how ratings changed over time
        self.rating_history = defaultdict(list)
        
        # Store pre-match ratings for feature engineering later
        # This is CRITICAL: we need to know what the rating WAS before the match
        # not what it becomes after
        self.pre_match_ratings = []
    
    # ================================================================
    # CORE ELO MATHEMATICS
    # ================================================================
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate the expected score (win probability) for Team A.
        
        This is the core Elo formula.
        Output is between 0 and 1 (a probability).
        
        Examples:
            equal teams (1500 vs 1500) -> 0.5 (50% chance each)
            strong vs weak (1800 vs 1200) -> ~0.91 (91% chance for strong team)
        
        Parameters:
            rating_a : Elo rating of Team A
            rating_b : Elo rating of Team B
        
        Returns:
            float: Probability that Team A wins (0.0 to 1.0)
        """
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
    
    def update_ratings(self, 
                       team_a: str, 
                       team_b: str, 
                       result: float, 
                       date=None) -> tuple:
        """
        Update both teams' ratings after a match.
        
        Parameters:
            team_a  : Name of Team A (home team in our case)
            team_b  : Name of Team B (away team)
            result  : Match result from Team A's perspective
                      1.0 = Team A wins
                      0.5 = Draw
                      0.0 = Team A loses
            date    : Match date (for tracking history)
        
        Returns:
            tuple: (new_rating_a, new_rating_b)
        """
        # Get CURRENT ratings before the match
        ra = self.ratings[team_a]
        rb = self.ratings[team_b]
        
        # Calculate expected scores
        expected_a = self.expected_score(ra, rb)
        expected_b = 1.0 - expected_a  # Their chances must add up to 1
        
        # Calculate actual scores from each team's perspective
        actual_a = result
        actual_b = 1.0 - result
        
        # Apply the Elo update formula
        # R_new = R_old + K * (Actual - Expected)
        new_ra = ra + self.k_factor * (actual_a - expected_a)
        new_rb = rb + self.k_factor * (actual_b - expected_b)
        
        # Save updated ratings
        self.ratings[team_a] = new_ra
        self.ratings[team_b] = new_rb
        
        # Record history if date provided
        if date is not None:
            self.rating_history[team_a].append({
                'date': date,
                'rating': round(new_ra, 2)
            })
            self.rating_history[team_b].append({
                'date': date,
                'rating': round(new_rb, 2)
            })
        
        return new_ra, new_rb
    
    # ================================================================
    # PROCESS ALL MATCHES
    # ================================================================
    
    def compute_ratings(self, df: pd.DataFrame) -> dict:
        """
        Process every match in chronological order to compute Elo ratings.
        
        WHY CHRONOLOGICAL ORDER MATTERS:
        Elo ratings are "living" numbers that update after every match.
        If we process matches out of order, the ratings won't make sense.
        Think of it like: you can't know Brazil's rating in 2020 without 
        processing all their matches from 2000-2019 first.
        
        Parameters:
            df : DataFrame with columns: date, home_team, away_team, 
                 home_score, away_score
        
        Returns:
            dict : {team_name: final_elo_rating}
        """
        # Sort matches from oldest to newest
        df = df.sort_values('date').reset_index(drop=True)
        
        total_matches = len(df)
        print(f"Processing {total_matches:,} matches to compute Elo ratings...")
        print("This may take 10-30 seconds...")
        
        # Process every match one by one
        for idx, row in df.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            date = row['date']
            
            # --------------------------------------------------------
            # RECORD PRE-MATCH RATINGS
            # This is important for feature engineering (Day 3)
            # We need to know ratings BEFORE the match updates them
            # --------------------------------------------------------
            home_elo_before = self.ratings[home_team]
            away_elo_before = self.ratings[away_team]
            elo_difference = home_elo_before - away_elo_before
            
            self.pre_match_ratings.append({
                'date': date,
                'home_team': home_team,
                'away_team': away_team,
                'home_elo_before': round(home_elo_before, 2),
                'away_elo_before': round(away_elo_before, 2),
                'elo_difference': round(elo_difference, 2)
            })
            
            # --------------------------------------------------------
            # DETERMINE RESULT FROM HOME TEAM'S PERSPECTIVE
            # --------------------------------------------------------
            if row['home_score'] > row['away_score']:
                result = 1.0   # Home team won
            elif row['home_score'] < row['away_score']:
                result = 0.0   # Home team lost (away team won)
            else:
                result = 0.5   # Draw
            
            # --------------------------------------------------------
            # UPDATE RATINGS
            # --------------------------------------------------------
            self.update_ratings(home_team, away_team, result, date)
            
            # Show progress every 5000 matches
            if (idx + 1) % 5000 == 0:
                print(f"  Processed {idx + 1:,} / {total_matches:,} matches...")
        
        final_ratings = dict(self.ratings)
        print(f"\nElo computation complete!")
        print(f"Total teams rated: {len(final_ratings)}")
        
        return final_ratings
    
    # ================================================================
    # GET RESULTS AS DATAFRAMES
    # ================================================================
    
    def get_current_ratings(self) -> pd.DataFrame:
        """
        Get current Elo ratings as a sorted DataFrame.
        
        Returns:
            pd.DataFrame with columns: rank, team, elo
        """
        ratings_list = [
            {'team': team, 'elo': round(rating, 2)}
            for team, rating in self.ratings.items()
        ]
        
        df = pd.DataFrame(ratings_list)
        df = df.sort_values('elo', ascending=False).reset_index(drop=True)
        df.index += 1  # Start ranking from 1
        df.index.name = 'rank'
        
        return df
    
    def get_pre_match_ratings(self) -> pd.DataFrame:
        """
        Get the pre-match Elo ratings recorded during computation.
        
        This is used in feature engineering (Day 3).
        For each match, it shows what the Elo ratings were BEFORE 
        the match was played.
        
        Returns:
            pd.DataFrame with columns: date, home_team, away_team,
                                       home_elo_before, away_elo_before,
                                       elo_difference
        """
        return pd.DataFrame(self.pre_match_ratings)
    
    def get_rating_history(self, team: str) -> pd.DataFrame:
        """
        Get the rating history for a specific team.
        
        Parameters:
            team : Team name (e.g., 'Brazil', 'France')
        
        Returns:
            pd.DataFrame with columns: date, rating
        """
        if team not in self.rating_history:
            print(f"No history found for: {team}")
            return pd.DataFrame()
        
        return pd.DataFrame(self.rating_history[team])
    
    def get_team_rating(self, team: str) -> float:
        """
        Get current Elo rating for a specific team.
        
        Parameters:
            team : Team name
        
        Returns:
            float : Current Elo rating (1500 if team not found)
        """
        return round(self.ratings[team], 2)
    
    # ================================================================
    # SAVE TO DISK
    # ================================================================
    
    def save_ratings(self, 
                     filepath: str = "data/processed/elo_ratings.csv") -> None:
        """
        Save current Elo ratings to a CSV file.
        
        Parameters:
            filepath : Where to save the file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        ratings_df = self.get_current_ratings()
        ratings_df.to_csv(filepath)
        print(f"Elo ratings saved to: {filepath}")
    
    def save_pre_match_ratings(self,
                                filepath: str = "data/processed/pre_match_elo.csv") -> None:
        """
        Save pre-match Elo data to CSV (used in feature engineering).
        
        Parameters:
            filepath : Where to save the file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        pre_df = self.get_pre_match_ratings()
        pre_df.to_csv(filepath, index=False)
        print(f"Pre-match Elo data saved to: {filepath}")
    
    # ================================================================
    # DISPLAY / ANALYSIS HELPERS
    # ================================================================
    
    def print_top_teams(self, n: int = 20) -> None:
        """
        Print the top N teams by Elo rating in a nice table.
        
        Parameters:
            n : Number of teams to show (default: 20)
        """
        ratings_df = self.get_current_ratings()
        top_n = ratings_df.head(n)
        
        print("\n" + "=" * 45)
        print(f"  TOP {n} TEAMS BY ELO RATING")
        print("=" * 45)
        print(f"{'Rank':<6} {'Team':<25} {'Elo Rating':<12}")
        print("-" * 45)
        
        for rank, row in top_n.iterrows():
            # Add medal emojis for top 3
            medal = ""
            if rank == 1:
                medal = " 🥇"
            elif rank == 2:
                medal = " 🥈"
            elif rank == 3:
                medal = " 🥉"
            
            print(f"{rank:<6} {row['team']:<25} {row['elo']:.2f}{medal}")
        
        print("=" * 45)
    
    def compare_teams(self, team_a: str, team_b: str) -> None:
        """
        Show head-to-head comparison and win probabilities for two teams.
        
        Parameters:
            team_a : First team name
            team_b : Second team name
        """
        ra = self.ratings[team_a]
        rb = self.ratings[team_b]
        
        prob_a = self.expected_score(ra, rb)
        prob_b = 1.0 - prob_a
        
        print("\n" + "=" * 50)
        print(f"  HEAD TO HEAD: {team_a} vs {team_b}")
        print("=" * 50)
        print(f"  {team_a:<20} Elo: {ra:.2f}")
        print(f"  {team_b:<20} Elo: {rb:.2f}")
        print(f"  Elo Difference: {abs(ra - rb):.2f}")
        print("-" * 50)
        print(f"  Win probability for {team_a}: {prob_a*100:.1f}%")
        print(f"  Win probability for {team_b}: {prob_b*100:.1f}%")
        print("=" * 50)


# ================================================================
# MAIN BLOCK: Runs when you execute this file directly
# Command: python -m src.elo
# ================================================================

if __name__ == "__main__":
    
    # Import the data loader we built on Day 1
    from src.data_loader import DataLoader
    
    print("=" * 60)
    print("  ELO RATING SYSTEM - DAY 2 TEST")
    print("=" * 60)
    
    # --------------------------------
    # Step 1: Load the data
    # --------------------------------
    print("\nStep 1: Loading match data...")
    loader = DataLoader()
    data = loader.load_raw_data()
    
    # --------------------------------
    # Step 2: Compute Elo ratings
    # --------------------------------
    print("\nStep 2: Computing Elo ratings...")
    elo = EloRatingSystem(initial_rating=1500, k_factor=40)
    elo.compute_ratings(data)
    
    # --------------------------------
    # Step 3: Show top 20 teams
    # --------------------------------
    elo.print_top_teams(20)
    
    # --------------------------------
    # Step 4: Show some match-ups
    # --------------------------------
    print("\nStep 4: Sample head-to-head predictions...")
    elo.compare_teams("Brazil", "France")
    elo.compare_teams("Argentina", "Germany")
    elo.compare_teams("England", "Spain")
    
    # --------------------------------
    # Step 5: Save ratings to disk
    # --------------------------------
    print("\nStep 5: Saving ratings to disk...")
    elo.save_ratings("data/processed/elo_ratings.csv")
    elo.save_pre_match_ratings("data/processed/pre_match_elo.csv")
    
    # --------------------------------
    # St