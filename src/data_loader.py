"""
data_loader.py
==============
PURPOSE: Load the raw football results CSV and do basic cleaning.

WHY THIS FILE EXISTS:
- Every project needs a clean way to load data
- We keep all data loading in one place so if the data changes, we only update one file
- This is called "separation of concerns" - each file does one job

WHAT IT DOES:
1. Reads results.csv from data/raw/
2. Filters to only keep matches from year 2000 onward
3. Removes rows with missing scores
4. Creates a 'target' column (what we want to predict):
   - 0 = Away team wins
   - 1 = Draw (tie)
   - 2 = Home team wins
"""

import pandas as pd
from pathlib import Path


class DataLoader:
    """
    A class to handle all data loading for this project.
    
    What is a class? Think of it as a blueprint.
    DataLoader is a blueprint that knows how to load our football data.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        __init__ runs automatically when you create a DataLoader object.
        It sets up the file paths we need.
        
        Parameters:
            data_dir: The folder where data is stored (default: "data")
        """
        self.data_dir = Path(data_dir)
        self.raw_path = self.data_dir / "raw" / "results.csv"
        self.processed_path = self.data_dir / "processed" / "features.csv"
        self.squad_path = self.data_dir / "squad_strength.csv"
        self.groups_path = self.data_dir / "wc2026_groups.csv"
    
    def load_raw_data(self, filter_year: int = 2000) -> pd.DataFrame:
        """
        Load and clean the raw football results.
        
        Parameters:
            filter_year: Only keep matches from this year onward (default: 2000)
        
        Returns:
            pd.DataFrame: A cleaned table of football matches
        """
        
        # Check if the file exists before trying to open it
        if not self.raw_path.exists():
            raise FileNotFoundError(
                f"\n\nERROR: Could not find the dataset at: {self.raw_path}\n"
                "Please download results.csv from Kaggle and place it in data/raw/\n"
                "Download link: https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017\n"
            )
        
        print(f"Loading data from: {self.raw_path}")
        
        # pd.read_csv reads a CSV file into a DataFrame (like a table/spreadsheet in Python)
        df = pd.read_csv(self.raw_path)
        
        print(f"Raw data shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")
        
        # Convert 'date' column from text string to actual date objects
        # This lets us do date math (like filter by year, sort chronologically)
        df['date'] = pd.to_datetime(df['date'])
        
        # Keep only matches from filter_year onward
        # More recent data is more relevant for predicting modern football
        df = df[df['date'].dt.year >= filter_year].copy()
        print(f"After filtering (year >= {filter_year}): {len(df):,} matches")
        
        # Remove rows where scores are missing (NaN = Not a Number = missing value)
        df = df.dropna(subset=['home_score', 'away_score']).reset_index(drop=True)
        print(f"After removing missing scores: {len(df):,} matches")
        
        # Make sure scores are integers (whole numbers), not decimals
        df['home_score'] = df['home_score'].astype(int)
        df['away_score'] = df['away_score'].astype(int)
        
        # ============================================================
        # CREATE THE TARGET VARIABLE
        # ============================================================
        # This is what our machine learning model will learn to predict
        # 0 = Away team wins
        # 1 = Draw
        # 2 = Home team wins
        
        def get_match_result(row):
            """Determine match result from home team perspective."""
            if row['home_score'] > row['away_score']:
                return 2  # Home win
            elif row['home_score'] < row['away_score']:
                return 0  # Away win
            else:
                return 1  # Draw
        
        # Apply this function to every row in the dataframe
        df['target'] = df.apply(get_match_result, axis=1)
        
        # Calculate goal difference (useful feature for models)
        df['goal_difference'] = df['home_score'] - df['away_score']
        
        # Sort by date (oldest first) - important for Elo calculations
        df = df.sort_values('date').reset_index(drop=True)
        
        # Show distribution of results
        result_counts = df['target'].value_counts().sort_index()
        print(f"\nResult distribution:")
        print(f"  Away wins (0): {result_counts.get(0, 0):,}")
        print(f"  Draws    (1): {result_counts.get(1, 0):,}")
        print(f"  Home wins(2): {result_counts.get(2, 0):,}")
        
        return df
    
    def load_squad_strength(self) -> pd.DataFrame:
        """
        Load team strength scores.
        
        Returns:
            pd.DataFrame with columns: team, strength
        """
        if not self.squad_path.exists():
            print(f"Warning: {self.squad_path} not found. Using default strengths.")
            return pd.DataFrame({'team': [], 'strength': []})
        
        return pd.read_csv(self.squad_path)
    
    def load_wc2026_groups(self) -> pd.DataFrame:
        """
        Load the 2026 World Cup group assignments.
        
        Returns:
            pd.DataFrame with columns: group, team
        """
        if not self.groups_path.exists():
            raise FileNotFoundError(f"Groups file not found: {self.groups_path}")
        
        return pd.read_csv(self.groups_path)
    
    def save_processed(self, df: pd.DataFrame) -> None:
        """Save the processed (feature-engineered) data to disk."""
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.processed_path, index=False)
        print(f"Saved processed data to: {self.processed_path}")
    
    def load_processed(self) -> pd.DataFrame:
        """Load previously saved processed data."""
        if not self.processed_path.exists():
            raise FileNotFoundError("No processed data found. Run feature engineering first.")
        return pd.read_csv(self.processed_path)


# ============================================================
# MAIN BLOCK - This runs when you execute this file directly
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("DATA LOADER TEST")
    print("=" * 60)
    
    # Create a DataLoader object
    loader = DataLoader()
    
    # Load the data
    data = loader.load_raw_data()
    
    print("\n" + "=" * 60)
    print("SAMPLE DATA (first 5 rows):")
    print("=" * 60)
    print(data[['date', 'home_team', 'away_team', 
                'home_score', 'away_score', 'target']].head())
    
    print("\n" + "=" * 60)
    print("DATE RANGE:")
    print("=" * 60)
    print(f"Earliest match: {data['date'].min().date()}")
    print(f"Latest match:   {data['date'].max().date()}")
    
    print("\n" + "=" * 60)
    print("TOP 10 TEAMS BY MATCHES PLAYED:")
    print("=" * 60)
    all_teams = pd.concat([data['home_team'], data['away_team']])
    print(all_teams.value_counts().head(10))
    
    print("\nData loader test PASSED!")