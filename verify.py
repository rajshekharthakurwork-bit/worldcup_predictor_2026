"""
verify.py
=========
Checks that all project files and outputs exist correctly.
Run: python verify.py
"""

from pathlib import Path

print("=" * 60)
print("  PROJECT VERIFICATION CHECK")
print("=" * 60)

# All files that should exist
required_files = [
    # Source code
    ("src/__init__.py",                "Source package init"),
    ("src/data_loader.py",             "Data loader module"),
    ("src/elo.py",                     "Elo rating system"),
    ("src/feature_engineering.py",     "Feature engineering"),
    ("src/train_model.py",             "Model training"),
    ("src/predictor.py",               "Match predictor"),
    ("src/simulator.py",               "Tournament simulator"),
    ("src/visualize.py",               "Visualization"),

    # Config files
    ("requirements.txt",               "Python requirements"),
    ("README.md",                      "Project README"),
    ("run_pipeline.py",                "Full pipeline runner"),
    (".gitignore",                     "Git ignore file"),

    # Data files
    ("data/raw/results.csv",           "Raw Kaggle dataset"),
    ("data/wc2026_groups.csv",         "WC2026 group draw"),
    ("data/squad_strength.csv",        "Squad strength scores"),

    # Generated files (after running pipeline)
    ("data/processed/elo_ratings.csv", "Computed Elo ratings"),
    ("data/processed/features.csv",    "ML feature matrix"),
    ("models/rf_model.pkl",            "Trained RF model"),
    ("outputs/champion_probabilities.csv", "Simulation results"),
    ("outputs/model_evaluation.txt",   "Model evaluation report"),

    # Charts
    ("outputs/charts/champion_probabilities.png", "Champion chart"),
    ("outputs/charts/elo_ratings.png",            "Elo ratings chart"),
    ("outputs/charts/probability_heatmap.png",    "Heatmap chart"),
    ("outputs/charts/all_groups.png",             "Group tables"),
    ("outputs/charts/top_contenders.png",         "Contenders chart"),
    ("outputs/charts/dark_horses.png",            "Dark horses chart"),
]

passed = 0
failed = 0
missing = []

print(f"\n{'Status':<8} {'File':<48} {'Description'}")
print("─" * 80)

for filepath, description in required_files:
    exists = Path(filepath).exists()
    if exists:
        size = Path(filepath).stat().st_size
        if size > 0:
            status = "✅ OK"
            passed += 1
        else:
            status = "⚠️  EMPTY"
            failed += 1
            missing.append((filepath, "file is empty"))
    else:
        status = "❌ MISS"
        failed += 1
        missing.append((filepath, "file not found"))

    print(f"{status:<8} {filepath:<48} {description}")

# Summary
print("\n" + "=" * 60)
print(f"  RESULTS: {passed} passed | {failed} failed")
print("=" * 60)

if failed == 0:
    print("\n  🎉 All files present! Project is complete.")
    print("  Your project is ready for GitHub and portfolio.")
else:
    print(f"\n  ⚠️  {failed} file(s) missing or empty:")
    for filepath, reason in missing:
        print(f"    - {filepath} ({reason})")
    print("\n  Run: python run_pipeline.py to generate missing files.")

print()