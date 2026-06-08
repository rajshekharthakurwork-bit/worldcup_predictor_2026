"""
train_model.py
==============
PURPOSE: Train machine learning models to predict football match outcomes.

WHY THIS FILE EXISTS:
- Takes the feature matrix built in Day 3
- Trains two models: Logistic Regression and Random Forest
- Compares their performance
- Saves the best model (Random Forest) to disk for use in simulation

WORKFLOW:
    1. Load features.csv
    2. Split into train (before 2022) and test (2022 onward)
    3. Train Logistic Regression → evaluate
    4. Train Random Forest → evaluate
    5. Compare results
    6. Save Random Forest model to models/rf_model.pkl

INPUTS:  data/processed/features.csv
OUTPUTS: models/rf_model.pkl
         outputs/model_evaluation.txt
"""

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler


class ModelTrainer:
    """
    Trains and evaluates ML models for match outcome prediction.
    
    HOW TO USE:
        trainer = ModelTrainer()
        trainer.load_features()
        trainer.split_data()
        trainer.train_logistic_regression()
        trainer.train_random_forest()
        trainer.compare_models()
        trainer.save_model()
    """

    def __init__(self,
                 features_path: str = "data/processed/features.csv",
                 model_dir: str = "models"):
        """
        Set up file paths and storage for models.

        Parameters:
            features_path : Path to the feature CSV built in Day 3
            model_dir     : Folder to save trained models
        """
        self.features_path = Path(features_path)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # These are the exact column names from feature_engineering.py
        # ONLY these columns are used as inputs to the model
        self.feature_columns = [
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

        # Target column: what we predict
        self.target_column = 'target'

        # Human-readable class labels for reports
        # 0=Away Win, 1=Draw, 2=Home Win
        self.class_names = ['Away Win', 'Draw', 'Home Win']

        # Storage for trained models and data splits
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train_df = None
        self.test_df = None

        # Scaler for Logistic Regression
        # (Random Forest doesn't need scaling)
        self.scaler = StandardScaler()

        # Trained model objects
        self.lr_model = None
        self.rf_model = None

        # Evaluation results storage
        self.results = {}

    # ================================================================
    # DATA LOADING AND SPLITTING
    # ================================================================

    def load_features(self) -> pd.DataFrame:
        """
        Load the feature matrix from CSV.

        Returns:
            pd.DataFrame : Feature matrix
        """
        if not self.features_path.exists():
            raise FileNotFoundError(
                f"\nFeatures file not found: {self.features_path}\n"
                "Please run Day 3 first: python -m src.feature_engineering"
            )

        print(f"Loading features from: {self.features_path}")
        self.df = pd.read_csv(self.features_path)
        self.df['date'] = pd.to_datetime(self.df['date'])

        print(f"  Total matches loaded: {len(self.df):,}")
        print(f"  Date range: {self.df['date'].min().date()} "
              f"to {self.df['date'].max().date()}")
        print(f"  Feature columns: {len(self.feature_columns)}")

        return self.df

    def split_data(self, split_year: int = 2022) -> tuple:
        """
        Split data into training and testing sets by year.

        WHY BY YEAR instead of random split?
        In time-series data like football matches, we must respect
        the order of time. Using a random split would let the model
        train on 2023 matches to predict 2021 matches — which is
        impossible in real life (you can't see the future).

        Training: matches before split_year  → model LEARNS from these
        Testing:  matches from split_year+   → model is EVALUATED on these

        Parameters:
            split_year : Year that separates train and test (default: 2022)

        Returns:
            tuple : (X_train, X_test, y_train, y_test)
        """
        if self.df is None:
            raise ValueError("Load features first using load_features()")

        # Split by year
        self.train_df = self.df[self.df['date'].dt.year < split_year].copy()
        self.test_df  = self.df[self.df['date'].dt.year >= split_year].copy()

        print(f"\nData Split (split year = {split_year}):")
        print(f"  Training set: {len(self.train_df):,} matches "
              f"({self.train_df['date'].min().date()} to "
              f"{self.train_df['date'].max().date()})")
        print(f"  Testing set:  {len(self.test_df):,} matches "
              f"({self.test_df['date'].min().date()} to "
              f"{self.test_df['date'].max().date()})")

        # Extract feature matrix X and target vector y
        self.X_train = self.train_df[self.feature_columns].values
        self.X_test  = self.test_df[self.feature_columns].values
        self.y_train = self.train_df[self.target_column].values
        self.y_test  = self.test_df[self.target_column].values

        # Show class distribution in training set
        print(f"\nTraining set class distribution:")
        unique, counts = np.unique(self.y_train, return_counts=True)
        for cls, cnt in zip(unique, counts):
            pct = cnt / len(self.y_train) * 100
            print(f"  {self.class_names[cls]}: {cnt:,} ({pct:.1f}%)")

        return self.X_train, self.X_test, self.y_train, self.y_test

    # ================================================================
    # MODEL 1: LOGISTIC REGRESSION
    # ================================================================

    def train_logistic_regression(self) -> dict:
        """
        Train a Logistic Regression model.

        WHAT IS LOGISTIC REGRESSION?
        Despite the name, it is a CLASSIFICATION algorithm (not regression).
        It learns a linear decision boundary between classes.

        Think of it as:
          "If elo_difference > X AND form_difference > Y, predict Home Win"

        It works by learning weights (coefficients) for each feature.
        Features with higher weights are more important.

        WHY SCALE FEATURES FOR LOGISTIC REGRESSION?
        Logistic Regression is sensitive to feature scale.
        If one feature ranges 0-2000 (Elo) and another 0-1 (form),
        the model unfairly weights the larger-scale feature.
        StandardScaler transforms each feature to have:
          - Mean = 0
          - Standard deviation = 1
        This puts all features on equal footing.

        Returns:
            dict : Evaluation metrics
        """
        print("\n" + "=" * 60)
        print("  MODEL 1: LOGISTIC REGRESSION")
        print("=" * 60)

        # Scale features (important for Logistic Regression)
        print("Scaling features with StandardScaler...")
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled  = self.scaler.transform(self.X_test)

        # Create and train the model
        # max_iter=1000: give it enough iterations to converge
        # random_state=42: reproducibility (same result every run)
        # C=1.0: regularization strength (prevents overfitting)
        print("Training Logistic Regression...")
        self.lr_model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=1.0,
            multi_class='multinomial'
        )
        self.lr_model.fit(X_train_scaled, self.y_train)

        # Make predictions on test set
        y_pred = self.lr_model.predict(X_test_scaled)
        y_prob = self.lr_model.predict_proba(X_test_scaled)

        # Calculate accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"\nLogistic Regression Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

        # Full classification report
        report = classification_report(
            self.y_test, y_pred,
            target_names=self.class_names,
            digits=4
        )
        print("\nClassification Report:")
        print(report)

        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print("Confusion Matrix:")
        print(self._format_confusion_matrix(cm))

        # Store results
        self.results['logistic_regression'] = {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_prob': y_prob
        }

        return self.results['logistic_regression']

    # ================================================================
    # MODEL 2: RANDOM FOREST
    # ================================================================

    def train_random_forest(self, 
                            n_estimators: int = 200,
                            max_depth: int = 10) -> dict:
        """
        Train a Random Forest classifier.

        WHAT IS RANDOM FOREST?
        A Random Forest builds many Decision Trees and combines
        their predictions by majority vote.

        DECISION TREE (one tree):
            Is elo_difference > 150?
             ├── YES: Is home_form > 0.6?
             │         ├── YES → Predict Home Win
             │         └── NO  → Predict Draw
             └── NO:  Is away_form > 0.7?
                       ├── YES → Predict Away Win
                       └── NO  → Predict Draw

        RANDOM FOREST (100-200 trees):
            Each tree sees a random subset of data and features.
            Final prediction = majority vote of all trees.
            This reduces overfitting and increases accuracy.

        WHY RANDOM FOREST FOR FOOTBALL?
        - Handles non-linear patterns (football is complex)
        - Works well without feature scaling
        - Provides feature importance scores
        - Naturally outputs probabilities via predict_proba()
        - Robust to noise and outliers

        Parameters:
            n_estimators : Number of trees to build (default: 200)
            max_depth    : Maximum depth of each tree (default: 10)

        Returns:
            dict : Evaluation metrics
        """
        print("\n" + "=" * 60)
        print("  MODEL 2: RANDOM FOREST")
        print("=" * 60)

        # Random Forest does NOT need feature scaling
        print(f"Training Random Forest ({n_estimators} trees, "
              f"max_depth={max_depth})...")
        print("This may take 30-60 seconds...")

        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1         # Use all CPU cores for speed
        )
        self.rf_model.fit(self.X_train, self.y_train)

        # Make predictions
        y_pred = self.rf_model.predict(self.X_test)
        y_prob = self.rf_model.predict_proba(self.X_test)

        # Calculate accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"\nRandom Forest Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

        # Classification report
        report = classification_report(
            self.y_test, y_pred,
            target_names=self.class_names,
            digits=4
        )
        print("\nClassification Report:")
        print(report)

        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print("Confusion Matrix:")
        print(self._format_confusion_matrix(cm))

        # Feature importance
        self._print_feature_importance()

        # Store results
        self.results['random_forest'] = {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_prob': y_prob
        }

        return self.results['random_forest']

    # ================================================================
    # MODEL COMPARISON
    # ================================================================

    def compare_models(self) -> None:
        """
        Print a side-by-side comparison of both models.
        """
        if not self.results:
            print("No models trained yet.")
            return

        print("\n" + "=" * 60)
        print("  MODEL COMPARISON")
        print("=" * 60)

        print(f"\n{'Model':<30} {'Accuracy':<15} {'Better?'}")
        print("-" * 55)

        scores = {}
        for name, result in self.results.items():
            scores[name] = result['accuracy']

        best_model = max(scores, key=scores.get)

        for name, acc in scores.items():
            is_best = "✅ WINNER" if name == best_model else ""
            display_name = name.replace('_', ' ').title()
            print(f"{display_name:<30} {acc:.4f} ({acc*100:.2f}%)   {is_best}")

        print("\n" + "=" * 60)
        print("  WHY FOOTBALL PREDICTION IS HARD")
        print("=" * 60)
        print("""
Even the best models achieve only 50-60% accuracy on football.
Here is why:

1. HIGH RANDOMNESS: One deflected shot can decide a match.
   The best team does not always win.

2. DRAWS: Football has 3 outcomes (Win/Draw/Loss).
   Random guessing = 33% accuracy. Models aim for 50-60%.

3. MISSING DATA: Injuries, weather, motivation, tactics
   are not in our dataset.

4. RARE EVENTS: Upsets happen. Qatar beat Germany at WC2022.
   No model could reliably predict that.

5. SOLUTION: Use PROBABILITIES not hard predictions.
   Instead of "Brazil wins", say "Brazil has 65% chance".
   This is what predict_proba() gives us.
   Monte Carlo simulation then uses these probabilities.
        """)

    # ================================================================
    # FEATURE IMPORTANCE
    # ================================================================

    def _print_feature_importance(self) -> None:
        """
        Print feature importance from Random Forest.

        WHAT IS FEATURE IMPORTANCE?
        Random Forest measures how much each feature
        helps reduce prediction error across all trees.
        Higher importance = more useful feature.

        This tells us which factors matter most in predicting
        football match outcomes.
        """
        if self.rf_model is None:
            return

        importances = self.rf_model.feature_importances_

        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print("\nFeature Importance (Random Forest):")
        print("-" * 45)
        print(f"{'Feature':<35} {'Importance'}")
        print("-" * 45)

        for _, row in importance_df.iterrows():
            bar = "█" * int(row['importance'] * 100)
            print(f"{row['feature']:<35} {row['importance']:.4f} {bar}")

    # ================================================================
    # CONFUSION MATRIX FORMATTER
    # ================================================================

    def _format_confusion_matrix(self, cm: np.ndarray) -> str:
        """
        Format confusion matrix as a readable table.

        WHAT IS A CONFUSION MATRIX?
        Shows how many predictions were correct vs wrong.

                    PREDICTED
                Away  Draw  Home
        ACTUAL Away [ 120    45    30 ]
               Draw [  55    80    65 ]
               Home [  25    40   200 ]

        Diagonal = correct predictions
        Off-diagonal = wrong predictions

        Parameters:
            cm : Confusion matrix array from sklearn

        Returns:
            str : Formatted string
        """
        lines = []
        lines.append(f"\n{'':>12} {'Predicted':^30}")
        lines.append(f"{'':>12} {'Away Win':^10} {'Draw':^10} {'Home Win':^10}")
        lines.append(f"{'':>12} {'-'*30}")

        row_labels = ['Away Win', 'Draw    ', 'Home Win']
        for i, (label, row) in enumerate(zip(row_labels, cm)):
            prefix = "Actual " if i == 1 else "       "
            lines.append(f"{prefix}{label} [ {row[0]:^8} {row[1]:^8} {row[2]:^8} ]")

        return '\n'.join(lines)

    # ================================================================
    # SAVE / LOAD MODEL
    # ================================================================

    def save_model(self, filename: str = "rf_model.pkl") -> None:
        """
        Save the trained Random Forest model to disk.

        WHAT IS A .pkl FILE?
        pkl = pickle. Python's way of saving any object to disk.
        We save the trained model so we don't need to retrain it
        every time we run the simulation.

        joblib is faster than pickle for large numpy arrays
        (which is what a Random Forest is internally).

        Parameters:
            filename : Name of the saved model file
        """
        if self.rf_model is None:
            raise ValueError("Random Forest not trained yet.")

        model_path = self.model_dir / filename
        scaler_path = self.model_dir / "scaler.pkl"

        # Save Random Forest model
        joblib.dump(self.rf_model, model_path)
        print(f"\nRandom Forest model saved to: {model_path}")

        # Save scaler too (needed if using Logistic Regression later)
        joblib.dump(self.scaler, scaler_path)
        print(f"Scaler saved to: {scaler_path}")

        # Save feature column list
        feature_path = self.model_dir / "feature_columns.txt"
        with open(feature_path, 'w') as f:
            f.write('\n'.join(self.feature_columns))
        print(f"Feature columns saved to: {feature_path}")

    def load_model(self, filename: str = "rf_model.pkl"):
        """
        Load a previously saved Random Forest model.

        Parameters:
            filename : Name of the model file to load

        Returns:
            RandomForestClassifier : The loaded model
        """
        model_path = self.model_dir / filename

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Please train the model first."
            )

        self.rf_model = joblib.load(model_path)
        print(f"Model loaded from: {model_path}")
        return self.rf_model

    # ================================================================
    # SAVE EVALUATION REPORT
    # ================================================================

    def save_evaluation_report(self,
                                filepath: str = "outputs/model_evaluation.txt") -> None:
        """
        Save a full evaluation report to a text file.

        Parameters:
            filepath : Where to save the report
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("  2026 FIFA WORLD CUP PREDICTOR - MODEL EVALUATION\n")
            f.write("=" * 60 + "\n\n")

            for model_name, result in self.results.items():
                f.write(f"Model: {model_name.replace('_', ' ').title()}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Accuracy: {result['accuracy']:.4f} "
                        f"({result['accuracy']*100:.2f}%)\n\n")
                f.write("Classification Report:\n")
                f.write(result['report'] + "\n")
                f.write("Confusion Matrix:\n")
                f.write(self._format_confusion_matrix(result['confusion_matrix']))
                f.write("\n\n")

        print(f"Evaluation report saved to: {filepath}")


# ================================================================
# PREDICT FUNCTION (used by simulator on Day 5)
# ================================================================

def predict_match(model,
                  home_features: dict,
                  feature_columns: list) -> np.ndarray:
    """
    Predict probabilities for a single match.

    This function is used by the simulator (Day 5) to get
    win/draw/loss probabilities for any two teams.

    Parameters:
        model         : Trained RandomForestClassifier
        home_features : Dict of feature values for the match
        feature_columns : List of feature column names in correct order

    Returns:
        np.ndarray : [away_win_prob, draw_prob, home_win_prob]
    """
    # Build feature vector in correct order
    feature_vector = np.array([
        home_features.get(col, 0.0) for col in feature_columns
    ]).reshape(1, -1)

    # Get probabilities
    probs = model.predict_proba(feature_vector)[0]
    return probs


# ================================================================
# MAIN BLOCK
# Command: python -m src.train_model
# ================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  MODEL TRAINING - DAY 4")
    print("=" * 60)

    # -----------------------------------------------
    # Step 1: Create trainer and load features
    # -----------------------------------------------
    trainer = ModelTrainer()

    print("\nStep 1: Loading feature data...")
    trainer.load_features()

    # -----------------------------------------------
    # Step 2: Split into train and test
    # -----------------------------------------------
    print("\nStep 2: Splitting data...")
    trainer.split_data(split_year=2022)

    # -----------------------------------------------
    # Step 3: Train Logistic Regression
    # -----------------------------------------------
    print("\nStep 3: Training Logistic Regression...")
    trainer.train_logistic_regression()

    # -----------------------------------------------
    # Step 4: Train Random Forest
    # -----------------------------------------------
    print("\nStep 4: Training Random Forest...")
    trainer.train_random_forest(n_estimators=200, max_depth=10)

    # -----------------------------------------------
    # Step 5: Compare models
    # -----------------------------------------------
    print("\nStep 5: Comparing models...")
    trainer.compare_models()

    # -----------------------------------------------
    # Step 6: Save model and report
    # -----------------------------------------------
    print("\nStep 6: Saving model to disk...")
    trainer.save_model()
    trainer.save_evaluation_report()

    # -----------------------------------------------
    # Step 7: Quick prediction demo
    # -----------------------------------------------
    print("\n" + "=" * 60)
    print("  QUICK PREDICTION DEMO")
    print("=" * 60)

    sample_features = {
        'home_elo': 1800.0,
        'away_elo': 1600.0,
        'elo_difference': 200.0,
        'home_form': 0.8,
        'away_form': 0.5,
        'form_difference': 0.3,
        'home_goals_scored_avg': 2.5,
        'home_goals_conceded_avg': 0.8,
        'away_goals_scored_avg': 1.5,
        'away_goals_conceded_avg': 1.2,
        'home_win_rate': 0.8,
        'away_win_rate': 0.4,
        'home_strength': 90.0,
        'away_strength': 75.0,
        'strength_difference': 15.0,
        'is_neutral': 0,
    }

    probs = predict_match(
        trainer.rf_model,
        sample_features,
        trainer.feature_columns
    )

    print("\nScenario: Strong Home Team (Elo 1800) vs Weaker Away Team (Elo 1600)")
    print(f"\n  Away Win probability : {probs[0]*100:.1f}%")
    print(f"  Draw probability     : {probs[1]*100:.1f}%")
    print(f"  Home Win probability : {probs[2]*100:.1f}%")
    print(f"\n  Most likely result   : {['Away Win','Draw','Home Win'][np.argmax(probs)]}")

    print("\n✅ Day 4 Complete!")
    print("\nFiles created:")
    print("  models/rf_model.pkl")
    print("  models/scaler.pkl")
    print("  models/feature_columns.txt")
    print("  outputs/model_evaluation.txt")