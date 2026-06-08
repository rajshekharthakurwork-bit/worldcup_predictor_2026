<div align="center">

# 🌍 2026 FIFA World Cup Predictor

### Machine Learning meets Football Analytics

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.1-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![pandas](https://img.shields.io/badge/pandas-2.2.2-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![matplotlib](https://img.shields.io/badge/matplotlib-3.9.2-11557C?style=for-the-badge)](https://matplotlib.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<p align="center">
  <strong>An end-to-end ML pipeline that predicts the 2026 FIFA World Cup winner<br>
  using Elo ratings, feature engineering, Random Forest, and Monte Carlo simulation.</strong>
</p>

---

**[📊 View Results](#-results) · [🚀 Quick Start](#-quick-start) · [🧠 How It Works](#-how-it-works) · [📁 Project Structure](#-project-structure)**

</div>

---

## 📌 Project Overview

This project builds a complete **Data Science pipeline** that:

| Step | What happens |
|------|-------------|
| 📥 **Data Loading** | Loads 24+ years of international football results (47,000+ matches) |
| 📊 **Elo Ratings** | Computes dynamic strength ratings for every national team |
| ⚙️ **Feature Engineering** | Builds 16 ML features: form, goals, strength, venue |
| 🤖 **ML Modeling** | Trains Logistic Regression and Random Forest classifiers |
| 🏆 **Simulation** | Simulates the full 48-team tournament 10,000 times |
| 📈 **Visualization** | Generates charts, heatmaps, and an HTML dashboard |

> **Why this project?** Standard match prediction models predict single outcomes.
> This system predicts **probabilities** across an entire tournament using
> Monte Carlo simulation — far more realistic and portfolio-worthy.

---

## 🏆 Results

> *Based on 10,000 Monte Carlo simulations*

<div align="center">

| 🥇 Rank | Team | Champion % | Reach Final % | Reach Semi % |
|:---:|:---|:---:|:---:|:---:|
| 🥇 | **Brazil** | ~19% | ~34% | ~50% |
| 🥈 | **France** | ~16% | ~29% | ~45% |
| 🥉 | **Argentina** | ~13% | ~25% | ~40% |
| 4 | England | ~10% | ~21% | ~36% |
| 5 | Spain | ~9% | ~19% | ~33% |

*Exact values vary slightly between runs due to Monte Carlo randomness*

</div>

### 📊 Charts

<table>
  <tr>
    <td align="center">
      <img src="outputs/charts/champion_probabilities.png" width="420"/>
      <br><em>Champion Probabilities</em>
    </td>
    <td align="center">
      <img src="outputs/charts/probability_heatmap.png" width="420"/>
      <br><em>Stage Probability Heatmap</em>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="outputs/charts/elo_ratings.png" width="420"/>
      <br><em>Elo Ratings — Top 20</em>
    </td>
    <td align="center">
      <img src="outputs/charts/top10_stage_breakdown.png" width="420"/>
      <br><em>Stage Breakdown</em>
    </td>
  </tr>
</table>

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- ~500MB disk space

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/worldcup_predictor_2026.git
cd worldcup_predictor_2026
```

### 2. Create Virtual Environment

```bash
# Create
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Dataset

Download the **International Football Results** dataset from Kaggle:

👉 [https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)

Place the file at:
```
data/raw/results.csv
```

### 5. Run Full Pipeline

```bash
python run_pipeline.py
```

> ⏱️ Total runtime: approximately 10–20 minutes

### 6. View Results

```bash
# Interactive prediction tool
python predict.py

# HTML dashboard in browser
python dashboard.py
```

---

## 🧠 How It Works

### 1️⃣ Elo Rating System

Every team starts at **1500**. After every match ratings update based on result and surprise factor:

```
Expected:  E  =  1 / (1 + 10^((Rb - Ra) / 400))
Update:    R_new  =  R_old + K × (Actual - Expected)
K-factor = 40
```

**Key insight:** Beating a strong team = big gain. Losing to a weak team = big loss.

---

### 2️⃣ Feature Engineering

16 features computed per match — all using only data available **before** the match (no leakage):

```
Elo Features          → home_elo, away_elo, elo_difference
Form Features         → home_form, away_form (last 5 matches, 0–1 scale)
Goal Features         → avg goals scored/conceded (last 5 matches)
Win Rate              → home_win_rate, away_win_rate
Strength Features     → squad strength score, strength_difference
Venue                 → is_neutral (0 or 1)
```

---

### 3️⃣ Machine Learning

**Target variable:**
```
0 = Away Win    1 = Draw    2 = Home Win
```

**Train/Test Split:**
```
Training  →  Matches before 2022  (~13,000 matches)
Testing   →  Matches 2022 onward  (~2,000 matches)
```

**Why time-based split instead of random?**
Football is time-series data. Using future matches to predict past ones would be "data leakage" — unrealistically inflating accuracy.

| Model | Test Accuracy | Notes |
|-------|:---:|-------|
| Logistic Regression | ~52% | Simple baseline |
| **Random Forest** | **~54%** | ✅ Selected for simulation |

**Why only ~54%?** Football has 3 outcomes and high randomness. Even professional prediction systems achieve 52–58%. The model outputs **probabilities** — not hard predictions — which is the right approach.

---

### 4️⃣ Tournament Simulation

**2026 FIFA World Cup Format:**
```
48 Teams → 12 Groups × 4 Teams
  ↓
Top 2 per group     = 24 teams  qualified
Best 8 third-place  =  8 teams  qualified
  ↓
Round of 32  →  Round of 16  →  QF  →  SF  →  Final
```

**Monte Carlo method:**
```python
for simulation in range(10_000):
    simulate_group_stage()      # All 12 groups, 6 matches each
    qualify_32_teams()          # Top 2 + best 8 third-place
    simulate_knockout_rounds()  # R32 → R16 → QF → SF → Final
    record_champion()

champion_probability = wins / 10_000 × 100
```

Upsets happen in every simulation because results are **randomly sampled** from probabilities — not always picking the favourite.

---

## 📁 Project Structure

```
worldcup_predictor_2026/
│
├── 📂 data/
│   ├── raw/
│   │   └── results.csv              ← Download from Kaggle
│   ├── processed/                   ← Auto-generated by pipeline
│   ├── wc2026_groups.csv            ← 2026 World Cup draw (48 teams)
│   └── squad_strength.csv           ← Team strength scores
│
├── 📂 src/
│   ├── __init__.py
│   ├── data_loader.py               ← Load & clean dataset
│   ├── elo.py                       ← Elo rating computation
│   ├── feature_engineering.py       ← Build ML feature matrix
│   ├── train_model.py               ← Train & evaluate models
│   ├── predictor.py                 ← Match probability predictor
│   ├── simulator.py                 ← Monte Carlo tournament sim
│   └── visualize.py                 ← Generate all charts
│
├── 📂 models/
│   └── feature_columns.txt          ← Feature names list
│
├── 📂 outputs/
│   ├── champion_probabilities.csv   ← Simulation results
│   ├── model_evaluation.txt         ← Model performance report
│   └── charts/                      ← All PNG chart files
│
├── run_pipeline.py                  ← 🚀 Run everything
├── predict.py                       ← 🎯 Interactive predictor
├── dashboard.py                     ← 📊 HTML dashboard
├── verify.py                        ← ✅ Check all files exist
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📦 Requirements

```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
matplotlib==3.9.2
seaborn==0.13.2
joblib==1.4.2
jupyter==1.0.0
ipykernel==6.29.5
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🎯 Usage Examples

### Predict any match
```bash
python predict.py
# Choose option 2
# Enter: Brazil vs France
```

### Check a team's tournament chances
```bash
python predict.py
# Choose option 3
# Enter: Argentina
```

### Compare two teams
```bash
python predict.py
# Choose option 5
# Enter: Spain vs Germany
```

### Open interactive dashboard
```bash
python dashboard.py
# Opens browser automatically
```

---

## 🔮 Future Improvements

<details>
<summary>Click to expand</summary>

| Improvement | Description |
|-------------|-------------|
| 🌐 **Live Data API** | Replace CSV with football-data.org live API |
| 👤 **Player-level data** | Include individual FIFA/Sofascore ratings |
| 🏥 **Injury data** | Scrape squad availability before matches |
| 🧠 **XGBoost / Neural Net** | More powerful models for better accuracy |
| 🔧 **Hyperparameter tuning** | GridSearchCV / Optuna optimization |
| 📊 **MLflow tracking** | Experiment tracking and model versioning |
| 🐳 **Docker container** | Reproducible deployment environment |
| 🌐 **Streamlit dashboard** | Interactive web app version |
| ☁️ **Cloud deployment** | AWS/GCP model serving |
| 💰 **Betting odds features** | Market odds as additional signal |

</details>

---

## 🧩 Key Learnings

- **Elo ratings** effectively capture team strength from match history
- **Rolling features** capture current form better than all-time averages
- **Data leakage** must be carefully avoided in time-series ML
- **Monte Carlo simulation** gives reliable probability estimates
- **~54% accuracy** is realistic — football is inherently random
- **Feature importance** shows Elo difference is the strongest predictor

---

## 📄 License

This project is licensed under the **MIT License** — free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- Dataset: [Mart Jürisoo — International Football Results on Kaggle](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
- Elo system inspired by FiveThirtyEight's football ratings methodology
- Tournament format based on official [FIFA 2026 documentation](https://www.fifa.com/fifaplus/en/articles/fifa-world-cup-2026)

---

<div align="center">

**Built as a Data Science portfolio project**

⭐ Star this repo if you found it useful!

</div>