"""
run_pipeline.py
===============
PURPOSE: Run the entire 2026 FIFA World Cup Predictor pipeline
         in one single command.

HOW TO RUN:
    python run_pipeline.py

WHAT IT DOES (in order):
    1. Loads and validates raw data
    2. Computes Elo ratings for all teams
    3. Engineers ML features
    4. Trains Logistic Regression and Random Forest models
    5. Runs 10,000 Monte Carlo tournament simulations
    6. Generates all charts and visualizations
    7. Prints final summary

TOTAL TIME: approximately 10-20 minutes
"""

import time
import sys
from pathlib import Path

# ================================================================
# TIMING HELPER
# ================================================================

def print_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def print_step(step_num: int, description: str) -> None:
    """Print a step indicator."""
    print(f"\n{'─'*65}")
    print(f"  STEP {step_num}: {description}")
    print(f"{'─'*65}")


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"\n  ✅ {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"\n  ❌ ERROR: {message}")


def format_time(seconds: float) -> str:
    """Format seconds into mm:ss string."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m {secs}s"


# ================================================================
# PIPELINE
# ================================================================

def run_pipeline(n_simulations: int = 10000):
    """
    Execute the full pipeline from data loading to visualization.

    Parameters:
        n_simulations : Number of Monte Carlo simulations (default: 10000)
    """

    # Track total time
    pipeline_start = time.time()
    step_times = {}

    print_header("2026 FIFA WORLD CUP PREDICTOR — FULL PIPELINE")
    print(f"\n  Configuration:")
    print(f"    Monte Carlo simulations : {n_simulations:,}")
    print(f"    Feature window          : 5 matches")
    print(f"    Train/test split year   : 2022")
    print(f"    Random Forest trees     : 200")

    # ============================================================
    # PRE-FLIGHT CHECK
    # ============================================================
    print_step(0, "Pre-flight checks")

    data_path = Path("data/raw/results.csv")
    if not data_path.exists():
        print_error(
            f"Dataset not found at: {data_path}\n"
            "  Please download results.csv from Kaggle and place it in data/raw/\n"
            "  Download: https://www.kaggle.com/datasets/martj42/"
            "international-football-results-from-1872-to-2017"
        )
        sys.exit(1)

    groups_path = Path("data/wc2026_groups.csv")
    if not groups_path.exists():
        print_error(f"Groups file not found: {groups_path}")
        sys.exit(1)

    print_success("All required files found")
    print(f"  Dataset:     {data_path}")
    print(f"  Groups:      {groups_path}")

    # ============================================================
    # STEP 1: DATA LOADING
    # ============================================================
    print_step(1, "Loading and validating data")
    t0 = time.time()

    try:
        from src.data_loader import DataLoader
        loader = DataLoader()
        raw_data = loader.load_raw_data(filter_year=2000)
        squad_df = loader.load_squad_strength()
        groups_df = loader.load_wc2026_groups()

        step_times['data_loading'] = time.time() - t0
        print_success(
            f"Data loaded: {len(raw_data):,} matches | "
            f"{len(groups_df)} WC2026 teams | "
            f"{len(squad_df)} squad strength entries"
        )
        print(f"  Time: {format_time(step_times['data_loading'])}")

    except Exception as e:
        print_error(f"Data loading failed: {e}")
        sys.exit(1)

    # ============================================================
    # STEP 2: ELO RATINGS
    # ============================================================
    print_step(2, "Computing Elo ratings")
    t0 = time.time()

    try:
        from src.elo import EloRatingSystem
        elo_system = EloRatingSystem(initial_rating=1500, k_factor=40)
        elo_system.compute_ratings(raw_data)
        elo_ratings_df = elo_system.get_current_ratings()
        pre_match_elo = elo_system.get_pre_match_ratings()

        # Save Elo files
        elo_system.save_ratings("data/processed/elo_ratings.csv")
        elo_system.save_pre_match_ratings("data/processed/pre_match_elo.csv")

        step_times['elo'] = time.time() - t0
        print_success(
            f"Elo computed for {len(elo_ratings_df)} teams | "
            f"{len(pre_match_elo):,} pre-match records saved"
        )

        # Print top 10
        print("\n  Top 10 Teams by Elo Rating:")
        print(f"  {'Rank':<5} {'Team':<22} {'Elo':>8}")
        print(f"  {'─'*38}")
        for i, row in elo_ratings_df.head(10).iterrows():
            print(f"  {i:<5} {row['team']:<22} {row['elo']:>8.1f}")

        print(f"\n  Time: {format_time(step_times['elo'])}")

    except Exception as e:
        print_error(f"Elo computation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ============================================================
    # STEP 3: FEATURE ENGINEERING
    # ============================================================
    print_step(3, "Engineering features")
    t0 = time.time()

    try:
        from src.feature_engineering import FeatureEngineer
        fe = FeatureEngineer(window=5)
        features_df = fe.build_features(raw_data, pre_match_elo, squad_df)
        fe.save_features(features_df)

        step_times['features'] = time.time() - t0
        print_success(
            f"Features built: {len(features_df):,} matches × "
            f"{len(fe.get_feature_columns())} features"
        )

        # Show target distribution
        target_counts = features_df['target'].value_counts().sort_index()
        labels = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
        print("\n  Target Distribution:")
        for val, count in target_counts.items():
            pct = count / len(features_df) * 100
            bar = "█" * int(pct / 2)
            print(f"    {labels[val]:<12} {count:>5,}  ({pct:4.1f}%)  {bar}")

        print(f"\n  Time: {format_time(step_times['features'])}")

    except Exception as e:
        print_error(f"Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ============================================================
    # STEP 4: MODEL TRAINING
    # ============================================================
    print_step(4, "Training ML models")
    t0 = time.time()

    try:
        from src.train_model import ModelTrainer
        trainer = ModelTrainer()
        trainer.load_features()
        trainer.split_data(split_year=2022)

        print("\n  Training Logistic Regression...")
        lr_results = trainer.train_logistic_regression()

        print("\n  Training Random Forest (200 trees)...")
        rf_results = trainer.train_random_forest(
            n_estimators=200, max_depth=10
        )

        trainer.save_model()
        trainer.save_evaluation_report()

        step_times['training'] = time.time() - t0

        lr_acc = lr_results['accuracy']
        rf_acc = rf_results['accuracy']

        print_success("Models trained and saved")
        print(f"\n  Model Performance:")
        print(f"    Logistic Regression : {lr_acc:.4f} ({lr_acc*100:.2f}%)")
        print(f"    Random Forest       : {rf_acc:.4f} ({rf_acc*100:.2f}%)  ← Selected")
        print(f"\n  Time: {format_time(step_times['training'])}")

    except Exception as e:
        print_error(f"Model training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ============================================================
    # STEP 5: TOURNAMENT SIMULATION
    # ============================================================
    print_step(5, f"Running {n_simulations:,} Monte Carlo simulations")
    t0 = time.time()

    try:
        from src.predictor import MatchPredictor
        from src.simulator import WorldCupSimulator

        # Set up predictor
        predictor = MatchPredictor()
        predictor.load_model()
        predictor.load_team_data(elo_ratings_df, features_df, squad_df)

        # Set up simulator
        simulator = WorldCupSimulator(predictor, groups_df)

        # Run simulations
        print(f"\n  Running {n_simulations:,} simulations...")
        print("  (Progress updates every 1,000 simulations)")
        results = simulator.run_simulations(n=n_simulations)

        # Save results
        simulator.save_results(results)

        step_times['simulation'] = time.time() - t0
        print_success(f"Simulations complete in {format_time(step_times['simulation'])}")

        # Print top 15
        simulator.print_results(results, top_n=15)

    except Exception as e:
        print_error(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ============================================================
    # STEP 6: VISUALIZATION
    # ============================================================
    print_step(6, "Generating charts and visualizations")
    t0 = time.time()

    try:
        from src.visualize import Visualizer
        viz = Visualizer(output_dir="outputs/charts")
        viz.load_data()
        saved_charts = viz.create_all_charts()

        step_times['visualization'] = time.time() - t0
        charts_created = sum(
            1 for v in saved_charts.values()
            if v is not None and v != []
        )
        print_success(
            f"{charts_created} charts created in {format_time(step_times['visualization'])}"
        )

    except Exception as e:
        print_error(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        # Don't exit — visualization failure is not critical

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    total_time = time.time() - pipeline_start

    print_header("PIPELINE COMPLETE — FINAL SUMMARY")

    print(f"\n  {'Step':<30} {'Time':>10}")
    print(f"  {'─'*42}")
    for step, t in step_times.items():
        print(f"  {step.replace('_',' ').title():<30} {format_time(t):>10}")
    print(f"  {'─'*42}")
    print(f"  {'TOTAL':<30} {format_time(total_time):>10}")

    print(f"\n  Output Files Created:")
    output_files = [
        ("data/processed/elo_ratings.csv",         "Elo ratings for all teams"),
        ("data/processed/features.csv",            "ML feature matrix"),
        ("models/rf_model.pkl",                    "Trained Random Forest model"),
        ("outputs/model_evaluation.txt",           "Model performance report"),
        ("outputs/champion_probabilities.csv",     "Champion probabilities"),
        ("outputs/charts/champion_probabilities.png", "Main results chart"),
        ("outputs/charts/probability_heatmap.png",  "Stage probability heatmap"),
        ("outputs/charts/elo_ratings.png",          "Team Elo ratings chart"),
        ("outputs/charts/all_groups.png",           "Group draw tables"),
        ("outputs/charts/top_contenders.png",       "Top contenders detail"),
    ]

    for filepath, description in output_files:
        exists = Path(filepath).exists()
        status = "✅" if exists else "⚠️"
        print(f"  {status} {filepath:<45} {description}")

    print(f"\n  Top 5 Predicted Champions:")
    print(f"  {'─'*45}")
    medals = ["🥇", "🥈", "🥉", "4 ", "5 "]
    for i, (medal, (_, row)) in enumerate(
        zip(medals, results.head(5).iterrows())
    ):
        print(
            f"  {medal}  {row['team']:<22} "
            f"Champion: {row['champion_prob']:5.1f}%  |  "
            f"Final: {row['final_prob']:5.1f}%"
        )

    print(f"\n{'='*65}")
    print(f"  ✅ 2026 FIFA World Cup Predictor — Pipeline Complete!")
    print(f"  Total time: {format_time(total_time)}")
    print(f"{'='*65}\n")


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":

    # You can reduce simulations for quick testing:
    # run_pipeline(n_simulations=1000)

    # Full run (recommended):
    run_pipeline(n_simulations=1000)