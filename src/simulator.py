"""
simulator.py
============
PURPOSE: Simulate the entire 2026 FIFA World Cup using Monte Carlo method.

WHY THIS FILE EXISTS:
- Uses the trained model's probabilities to simulate matches
- Runs the full tournament 10,000 times
- Tracks how often each team wins the tournament
- Produces champion probability estimates

2026 FORMAT:
  - 48 teams in 12 groups of 4
  - Top 2 from each group + best 8 third-place = 32 teams
  - Knockout: R32 → R16 → QF → SF → Final

INPUTS:  MatchPredictor + groups DataFrame
OUTPUTS: Champion probabilities CSV + stage probabilities CSV
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

from src.predictor import MatchPredictor


class WorldCupSimulator:
    """
    Simulates the 2026 FIFA World Cup using Monte Carlo simulation.

    HOW TO USE:
        sim = WorldCupSimulator(predictor, groups_df)
        results = sim.run_simulations(n=10000)
        sim.save_results(results)
    """

    def __init__(self,
                 predictor: MatchPredictor,
                 groups_df: pd.DataFrame):
        """
        Set up the simulator.

        Parameters:
            predictor : Trained MatchPredictor object
            groups_df : DataFrame with columns [group, team]
        """
        self.predictor = predictor
        self.groups_df = groups_df

        # Build group dictionary: {group_letter: [team1, team2, team3, team4]}
        self.groups = {}
        for group, group_data in groups_df.groupby('group'):
            self.groups[group] = group_data['team'].tolist()

        self.group_names = sorted(self.groups.keys())
        print(f"Tournament initialized with {len(self.groups)} groups")
        print(f"Total teams: {sum(len(t) for t in self.groups.values())}")

    # ================================================================
    # GROUP STAGE SIMULATION
    # ================================================================

    def _simulate_group(self, teams: list) -> pd.DataFrame:
        """
        Simulate all matches within one group.

        Each team plays the other 3 teams once (round-robin).
        Standings are calculated by: Points → Goal Difference → Goals Scored

        Points system:
            Win  = 3 points
            Draw = 1 point each
            Loss = 0 points

        Parameters:
            teams : List of 4 team names in this group

        Returns:
            pd.DataFrame : Group standings sorted by points
        """
        # Initialize standings for each team
        standings = {
            team: {
                'team': team,
                'played': 0,
                'won': 0,
                'drawn': 0,
                'lost': 0,
                'goals_for': 0,
                'goals_against': 0,
                'goal_diff': 0,
                'points': 0
            }
            for team in teams
        }

        # Play every combination: team_i vs team_j
        # With 4 teams: 6 matches total (4 choose 2 = 6)
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                team_a = teams[i]
                team_b = teams[j]

                # Simulate this match (draws allowed in group stage)
                result = self.predictor.predict_winner(
                    team_a, team_b,
                    neutral=True,
                    allow_draw=True
                )

                # Simulate a scoreline consistent with the result
                # This is needed for goal difference tiebreaker
                goals_a, goals_b = self._simulate_scoreline(
                    team_a, team_b, result
                )

                # Update standings
                standings[team_a]['played'] += 1
                standings[team_b]['played'] += 1
                standings[team_a]['goals_for'] += goals_a
                standings[team_a]['goals_against'] += goals_b
                standings[team_b]['goals_for'] += goals_b
                standings[team_b]['goals_against'] += goals_a
                standings[team_a]['goal_diff'] += goals_a - goals_b
                standings[team_b]['goal_diff'] += goals_b - goals_a

                if result == 'team_a':
                    standings[team_a]['won'] += 1
                    standings[team_a]['points'] += 3
                    standings[team_b]['lost'] += 1
                elif result == 'team_b':
                    standings[team_b]['won'] += 1
                    standings[team_b]['points'] += 3
                    standings[team_a]['lost'] += 1
                else:  # draw
                    standings[team_a]['drawn'] += 1
                    standings[team_a]['points'] += 1
                    standings[team_b]['drawn'] += 1
                    standings[team_b]['points'] += 1

        # Convert to DataFrame and sort by points, then goal diff, then goals for
        standings_df = pd.DataFrame(list(standings.values()))
        standings_df = standings_df.sort_values(
            ['points', 'goal_diff', 'goals_for'],
            ascending=[False, False, False]
        ).reset_index(drop=True)

        return standings_df

    def _simulate_scoreline(self,
                             team_a: str,
                             team_b: str,
                             result: str) -> tuple:
        """
        Simulate a realistic scoreline for a match.

        WHY: Group stage rankings use goal difference as a tiebreaker.
        We need actual scores, not just win/draw/loss.

        We use Poisson distribution to generate goals.
        Poisson is the standard statistical model for goal scoring
        in football — goals are rare, independent events.

        Parameters:
            team_a : First team
            team_b : Second team
            result : 'team_a', 'team_b', or 'draw'

        Returns:
            tuple : (goals_team_a, goals_team_b)
        """
        # Expected goals based on team strength
        avg_a = self.predictor.team_goals_scored.get(team_a, 1.3)
        avg_b = self.predictor.team_goals_scored.get(team_b, 1.3)

        # Clip to realistic range
        avg_a = np.clip(avg_a, 0.5, 3.5)
        avg_b = np.clip(avg_b, 0.5, 3.5)

        # Generate goals using Poisson distribution
        max_attempts = 20
        for _ in range(max_attempts):
            goals_a = np.random.poisson(avg_a)
            goals_b = np.random.poisson(avg_b)

            # Check if scoreline matches the result
            if result == 'team_a' and goals_a > goals_b:
                return goals_a, goals_b
            elif result == 'team_b' and goals_b > goals_a:
                return goals_a, goals_b
            elif result == 'draw' and goals_a == goals_b:
                return goals_a, goals_b

        # Fallback: force the correct result
        if result == 'team_a':
            return 1, 0
        elif result == 'team_b':
            return 0, 1
        else:
            return 1, 1

    # ================================================================
    # THIRD PLACE SELECTION
    # ================================================================

    def _select_best_third_place(self,
                                  third_place_teams: list) -> list:
        """
        Select the best 8 third-place teams from 12 groups.

        The 2026 format advances all 24 group winners and runners-up
        PLUS the best 8 of the 12 third-place teams.

        Ranking third-place teams: Points → Goal Difference → Goals Scored

        Parameters:
            third_place_teams : List of dicts with team standings

        Returns:
            list : 8 best third-place team names
        """
        if not third_place_teams:
            return []

        # Sort by points, then goal difference, then goals for
        sorted_third = sorted(
            third_place_teams,
            key=lambda x: (x['points'], x['goal_diff'], x['goals_for']),
            reverse=True
        )

        # Take the top 8
        best_8 = [t['team'] for t in sorted_third[:8]]
        return best_8

    # ================================================================
    # GROUP STAGE (ALL GROUPS)
    # ================================================================

    def _simulate_group_stage(self) -> dict:
        """
        Simulate all 12 groups and return qualified teams.

        Returns:
            dict with keys:
                'first_place'  : list of 12 group winners
                'second_place' : list of 12 group runners-up
                'third_place'  : list of 8 best third-place teams
                'all_qualified': list of all 32 qualified teams
                'group_standings': dict of group -> standings DataFrame
        """
        first_place = []
        second_place = []
        third_place_teams = []
        group_standings = {}

        for group_name in self.group_names:
            teams = self.groups[group_name]
            standings = self._simulate_group(teams)

            # Record standings
            group_standings[group_name] = standings

            # Extract positions
            first_place.append(standings.iloc[0]['team'])
            second_place.append(standings.iloc[1]['team'])

            # Third place team with their stats
            third = standings.iloc[2]
            third_place_teams.append({
                'team': third['team'],
                'group': group_name,
                'points': third['points'],
                'goal_diff': third['goal_diff'],
                'goals_for': third['goals_for']
            })

        # Select best 8 third-place teams
        best_third = self._select_best_third_place(third_place_teams)

        # All 32 qualified teams
        all_qualified = first_place + second_place + best_third

        return {
            'first_place': first_place,
            'second_place': second_place,
            'third_place': best_third,
            'all_qualified': all_qualified,
            'group_standings': group_standings
        }

    # ================================================================
    # KNOCKOUT STAGE
    # ================================================================

    def _simulate_knockout_match(self,
                                  team_a: str,
                                  team_b: str) -> str:
        """
        Simulate a single knockout match (no draws allowed).

        In knockout football, if teams are level after 90 minutes
        they go to extra time then penalties. We model this by
        removing the draw option and redistributing that probability.

        Parameters:
            team_a : First team
            team_b : Second team

        Returns:
            str : Winner ('team_a' or 'team_b')
        """
        result = self.predictor.predict_winner(
            team_a, team_b,
            neutral=True,
            allow_draw=False   # No draws in knockout
        )

        if result == 'team_a':
            return team_a
        else:
            return team_b

    def _simulate_knockout_round(self, teams: list) -> list:
        """
        Simulate one knockout round.

        Teams are paired sequentially: team[0] vs team[1],
        team[2] vs team[3], etc.

        Parameters:
            teams : List of teams (must be even number)

        Returns:
            list : Winners who advance to next round
        """
        winners = []

        # Pair teams and simulate each match
        for i in range(0, len(teams), 2):
            if i + 1 < len(teams):
                team_a = teams[i]
                team_b = teams[i + 1]
                winner = self._simulate_knockout_match(team_a, team_b)
                winners.append(winner)
            else:
                # Odd team out (shouldn't happen with correct format)
                winners.append(teams[i])

        return winners

    # ================================================================
    # FULL TOURNAMENT SIMULATION
    # ================================================================

    def _simulate_tournament(self) -> dict:
        """
        Simulate one complete World Cup tournament.

        Returns:
            dict : Results tracking which teams reached each stage
        """
        # Track which stage each team reached
        reached = defaultdict(str)

        # ============================================
        # GROUP STAGE
        # ============================================
        group_results = self._simulate_group_stage()
        qualified = group_results['all_qualified']

        # If we don't have exactly 32, pad or trim
        if len(qualified) < 32:
            # Fill with first place teams repeated if needed
            while len(qualified) < 32:
                qualified.append(group_results['first_place'][0])
        qualified = qualified[:32]

        # Mark all group stage participants
        for group_teams in self.groups.values():
            for team in group_teams:
                reached[team] = 'group_stage'

        # Mark qualified teams
        for team in qualified:
            reached[team] = 'round_of_32'

        # ============================================
        # ROUND OF 32 (32 → 16 teams)
        # ============================================
        r16_teams = self._simulate_knockout_round(qualified)

        for team in r16_teams:
            reached[team] = 'round_of_16'

        # ============================================
        # ROUND OF 16 (16 → 8 teams)
        # ============================================
        qf_teams = self._simulate_knockout_round(r16_teams)

        for team in qf_teams:
            reached[team] = 'quarterfinal'

        # ============================================
        # QUARTERFINALS (8 → 4 teams)
        # ============================================
        sf_teams = self._simulate_knockout_round(qf_teams)

        for team in sf_teams:
            reached[team] = 'semifinal'

        # ============================================
        # SEMIFINALS (4 → 2 teams)
        # ============================================
        finalists = self._simulate_knockout_round(sf_teams)

        for team in finalists:
            reached[team] = 'final'

        # ============================================
        # FINAL (2 → 1 CHAMPION)
        # ============================================
        champion = self._simulate_knockout_match(
            finalists[0], finalists[1]
        )
        reached[champion] = 'champion'

        return {
            'reached': dict(reached),
            'champion': champion,
            'finalists': finalists,
            'semifinalists': sf_teams,
            'quarterfinalists': qf_teams,
            'r16': r16_teams,
            'qualified': qualified
        }

    # ================================================================
    # MONTE CARLO: RUN N SIMULATIONS
    # ================================================================

    def run_simulations(self, n: int = 10000) -> dict:
        """
        Run N complete tournament simulations.

        This is the Monte Carlo simulation.
        We run the full tournament N times and count outcomes.

        Parameters:
            n : Number of simulations (default: 10000)

        Returns:
            dict : Probability estimates for each team at each stage
        """
        print(f"\n{'='*60}")
        print(f"  MONTE CARLO SIMULATION: {n:,} tournaments")
        print(f"{'='*60}")

        # Counters for each team at each stage
        all_teams = []
        for teams in self.groups.values():
            all_teams.extend(teams)
        all_teams = list(set(all_teams))

        stage_counts = {
            team: {
                'champion': 0,
                'final': 0,
                'semifinal': 0,
                'quarterfinal': 0,
                'round_of_16': 0,
                'round_of_32': 0,
                'group_stage': 0,
                'simulations': 0
            }
            for team in all_teams
        }

        # Run N simulations
        for sim_num in range(n):

            # Show progress every 1000 simulations
            if (sim_num + 1) % 1000 == 0:
                pct = (sim_num + 1) / n * 100
                print(f"  Simulation {sim_num+1:,}/{n:,} ({pct:.0f}%)...")

            # Simulate one full tournament
            result = self._simulate_tournament()
            reached = result['reached']

            # Update stage counters
            stage_order = [
                'group_stage', 'round_of_32', 'round_of_16',
                'quarterfinal', 'semifinal', 'final', 'champion'
            ]

            for team in all_teams:
                team_stage = reached.get(team, 'group_stage')
                stage_counts[team]['simulations'] += 1

                # Count all stages reached (cumulative)
                team_stage_idx = stage_order.index(team_stage) \
                    if team_stage in stage_order else 0

                for i, stage in enumerate(stage_order):
                    if i <= team_stage_idx:
                        stage_counts[team][stage] += 1

        print(f"\nAll {n:,} simulations complete!")

        # --------------------------------------------------------
        # Convert counts to probabilities
        # --------------------------------------------------------
        results = []
        for team, counts in stage_counts.items():
            sims = counts['simulations']
            if sims == 0:
                continue

            results.append({
                'team': team,
                'champion_prob': round(counts['champion'] / sims * 100, 2),
                'final_prob': round(counts['final'] / sims * 100, 2),
                'semifinal_prob': round(counts['semifinal'] / sims * 100, 2),
                'quarterfinal_prob': round(counts['quarterfinal'] / sims * 100, 2),
                'r16_prob': round(counts['round_of_16'] / sims * 100, 2),
                'qualified_prob': round(counts['round_of_32'] / sims * 100, 2),
            })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(
            'champion_prob', ascending=False
        ).reset_index(drop=True)

        return results_df

    # ================================================================
    # SAVE RESULTS
    # ================================================================

    def save_results(self, results_df: pd.DataFrame,
                     output_dir: str = "outputs") -> Path:
        """
        Save simulation results to CSV files.

        Parameters:
            results_df : Results DataFrame from run_simulations()
            output_dir : Output folder

        Returns:
            Path : Full path to saved champion probabilities CSV
        """
        # Anchor outputs to project root so the file lands in
        # worldcup_predictor_2026/outputs regardless of current cwd.
        project_root = Path(__file__).resolve().parent.parent
        output_path = project_root / output_dir
        output_path.mkdir(parents=True, exist_ok=True)

        # Save champion probabilities
        champ_path = output_path / "champion_probabilities.csv"
        results_df.to_csv(champ_path, index=False)
        print(f"Champion probabilities saved to: {champ_path}")
        return champ_path

    def print_results(self, results_df: pd.DataFrame,
                      top_n: int = 20) -> None:
        """
        Print simulation results in a nice table.

        Parameters:
            results_df : Results DataFrame
            top_n      : How many teams to show
        """
        print(f"\n{'='*75}")
        print(f"  2026 FIFA WORLD CUP - CHAMPION PROBABILITIES (Top {top_n})")
        print(f"{'='*75}")
        print(
            f"{'Rank':<5} {'Team':<22} {'Champion':>9} "
            f"{'Final':>8} {'Semi':>8} {'QF':>8} {'R16':>8}"
        )
        print("-" * 75)

        for idx, row in results_df.head(top_n).iterrows():
            rank = idx + 1
            medal = ""
            if rank == 1:
                medal = "🥇"
            elif rank == 2:
                medal = "🥈"
            elif rank == 3:
                medal = "🥉"

            print(
                f"{rank:<5} {row['team']:<22} "
                f"{row['champion_prob']:>8.1f}% "
                f"{row['final_prob']:>7.1f}% "
                f"{row['semifinal_prob']:>7.1f}% "
                f"{row['quarterfinal_prob']:>7.1f}% "
                f"{row['r16_prob']:>7.1f}% "
                f"{medal}"
            )

        print("=" * 75)
        print("All probabilities shown as percentages (%)")


# ================================================================
# MAIN BLOCK
# Command: python -m src.simulator
# ================================================================

if __name__ == "__main__":

    from src.data_loader import DataLoader
    from src.elo import EloRatingSystem
    from src.feature_engineering import FeatureEngineer
    from src.train_model import ModelTrainer

    print("=" * 60)
    print("  2026 FIFA WORLD CUP SIMULATOR - DAY 5")
    print("=" * 60)

    # -----------------------------------------------
    # Step 1: Load all data
    # -----------------------------------------------
    print("\nStep 1: Loading data...")
    loader = DataLoader()
    raw_data = loader.load_raw_data()
    squad_df = loader.load_squad_strength()
    groups_df = loader.load_wc2026_groups()

    # -----------------------------------------------
    # Step 2: Compute Elo ratings
    # -----------------------------------------------
    print("\nStep 2: Computing Elo ratings...")
    elo_system = EloRatingSystem()
    elo_system.compute_ratings(raw_data)
    elo_ratings_df = elo_system.get_current_ratings()
    pre_match_elo = elo_system.get_pre_match_ratings()

    # -----------------------------------------------
    # Step 3: Build features
    # -----------------------------------------------
    print("\nStep 3: Building features...")
    fe = FeatureEngineer(window=5)
    features_df = fe.build_features(raw_data, pre_match_elo, squad_df)

    # -----------------------------------------------
    # Step 4: Load trained model (or train if missing)
    # -----------------------------------------------
    print("\nStep 4: Loading trained model...")
    trainer = ModelTrainer()

    try:
        rf_model = trainer.load_model()
        print("Model loaded from disk.")
    except FileNotFoundError:
        print("Model not found. Training now...")
        trainer.load_features()
        trainer.split_data()
        trainer.train_random_forest()
        trainer.save_model()
        rf_model = trainer.rf_model

    # -----------------------------------------------
    # Step 5: Set up predictor
    # -----------------------------------------------
    print("\nStep 5: Setting up match predictor...")
    predictor = MatchPredictor()
    predictor.load_model()
    predictor.load_team_data(elo_ratings_df, features_df, squad_df)

    # -----------------------------------------------
    # Step 6: Quick prediction test
    # -----------------------------------------------
    print("\nStep 6: Quick prediction tests...")
    print("\n--- Sample Match Predictions ---")

    test_matches = [
        ("Brazil", "France"),
        ("Argentina", "England"),
        ("Spain", "Germany"),
        ("Portugal", "Netherlands"),
    ]

    for team_a, team_b in test_matches:
        probs = predictor.predict_match(team_a, team_b, neutral=True)
        print(f"\n  {team_a} vs {team_b}:")
        print(f"    {team_b} wins: {probs[0]*100:.1f}%")
        print(f"    Draw:       {probs[1]*100:.1f}%")
        print(f"    {team_a} wins: {probs[2]*100:.1f}%")

    # -----------------------------------------------
    # Step 7: Run Monte Carlo simulation
    # -----------------------------------------------
    print("\nStep 7: Running Monte Carlo simulation...")
    print("Running 10,000 tournament simulations...")
    print("This will take 3-8 minutes. Please wait...\n")

    simulator = WorldCupSimulator(predictor, groups_df)
    results = simulator.run_simulations(n=10000)

    # -----------------------------------------------
    # Step 8: Display and save results
    # -----------------------------------------------
    print("\nStep 8: Results!")
    simulator.print_results(results, top_n=20)
    champ_path = simulator.save_results(results)

    print("\n✅ Day 5 Complete!")
    print("\nFiles created:")
    print(f"  {champ_path}")