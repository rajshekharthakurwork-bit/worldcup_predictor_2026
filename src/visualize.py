"""
visualize.py
============
PURPOSE: Generate all charts and save them as PNG files to outputs/charts/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path


COLORS = {
    'primary':   '#1a472a',
    'secondary': '#2d6a4f',
    'accent':    '#52b788',
    'highlight': '#f4a261',
    'gold':      '#f7b731',
    'silver':    '#bdc3c7',
    'bronze':    '#cd7f32',
    'background':'#f8fff8',
    'white':     '#ffffff',
    'text':      '#2c3e50',
    'red':       '#e74c3c',
}


def set_style():
    plt.rcParams.update({
        'figure.facecolor':  COLORS['background'],
        'axes.facecolor':    COLORS['white'],
        'axes.edgecolor':    '#cccccc',
        'axes.labelcolor':   COLORS['text'],
        'axes.titlesize':    14,
        'axes.labelsize':    11,
        'xtick.labelsize':   10,
        'ytick.labelsize':   10,
        'text.color':        COLORS['text'],
        'font.family':       'DejaVu Sans',
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'figure.dpi':        120,
    })


class Visualizer:
    """Creates and saves all portfolio charts."""

    def __init__(self, output_dir: str = "outputs/charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.champion_df = None
        self.elo_df      = None
        self.groups_df   = None
        set_style()
        print(f"Visualizer ready. Saving charts to: {self.output_dir}")

    # ----------------------------------------------------------------
    # DATA LOADING
    # ----------------------------------------------------------------

    def load_data(self,
                  champion_path: str = "outputs/champion_probabilities.csv",
                  elo_path:      str = "data/processed/elo_ratings.csv",
                  groups_path:   str = "data/wc2026_groups.csv") -> None:

        print("Loading data for charts...")

        if Path(champion_path).exists():
            self.champion_df = pd.read_csv(champion_path)
            print(f"  Champion data : {len(self.champion_df)} teams")
        else:
            print(f"  WARNING: {champion_path} not found. Run simulator first.")

        if Path(elo_path).exists():
            self.elo_df = pd.read_csv(elo_path)
            print(f"  Elo data      : {len(self.elo_df)} teams")
        else:
            print(f"  WARNING: {elo_path} not found. Run elo module first.")

        if Path(groups_path).exists():
            self.groups_df = pd.read_csv(groups_path)
            print(f"  Groups data   : {len(self.groups_df)} teams")
        else:
            print(f"  WARNING: {groups_path} not found.")

        print("Data loaded.\n")

    # ----------------------------------------------------------------
    # CHART 1: CHAMPION PROBABILITY BAR CHART
    # ----------------------------------------------------------------

    def chart_champion_probabilities(self, top_n: int = 20) -> str:
        if self.champion_df is None:
            print("  Skipping champion chart — no data.")
            return None

        print("  Creating chart 1: Champion Probabilities...")

        top = self.champion_df.head(top_n).sort_values(
            'champion_prob', ascending=True
        )
        n = len(top)

        fig, ax = plt.subplots(figsize=(12, 10))
        fig.patch.set_facecolor(COLORS['background'])
        ax.set_facecolor(COLORS['background'])

        # Build color list
        def pick_color(i, n):
            r = i / max(n - 1, 1)
            if r > 0.8:   return COLORS['primary']
            elif r > 0.6: return COLORS['secondary']
            elif r > 0.4: return COLORS['accent']
            else:         return '#74c69d'

        bar_colors = [pick_color(i, n) for i in range(n)]

        bars = ax.barh(
            range(n),
            top['champion_prob'].values,
            color=bar_colors,
            height=0.68,
            edgecolor='white',
            linewidth=0.7
        )

        # Value labels
        for bar, val in zip(bars, top['champion_prob'].values):
            ax.text(
                val + 0.08,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%',
                va='center', ha='left',
                fontsize=9, fontweight='bold',
                color=COLORS['text']
            )

        # Y-axis labels with medals
        teams_list = top['team'].tolist()
        total      = len(self.champion_df)
        labels     = []
        for i, team in enumerate(teams_list):
            rank = total - i
            if rank == 1:   labels.append(f'🥇 {team}')
            elif rank == 2: labels.append(f'🥈 {team}')
            elif rank == 3: labels.append(f'🥉 {team}')
            else:           labels.append(f'    {team}')

        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=10.5)
        ax.xaxis.grid(True, alpha=0.25, linestyle='--')
        ax.set_axisbelow(True)
        ax.set_xlabel('Champion Probability (%)', fontsize=12, labelpad=8)
        ax.set_title(
            '2026 FIFA World Cup\nPredicted Champion Probabilities',
            fontsize=16, fontweight='bold',
            color=COLORS['primary'], pad=18
        )
        ax.axvline(x=0, color=COLORS['primary'], linewidth=2)

        fig.text(
            0.5, 0.01,
            'Based on 10,000 Monte Carlo simulations  |  '
            'Elo ratings + Random Forest ML model',
            ha='center', fontsize=8, color='gray', style='italic'
        )

        plt.tight_layout(rect=[0, 0.04, 1, 1])
        path = self.output_dir / "champion_probabilities.png"
        plt.savefig(path, dpi=150, bbox_inches='tight',
                    facecolor=COLORS['background'])
        plt.close()
        print(f"    Saved: {path}")
        return str(path)

    # ----------------------------------------------------------------
    # CHART 2: STAGE BREAKDOWN
    # ----------------------------------------------------------------

    def chart_stage_breakdown(self, top_n: int = 12) -> str:
        if self.champion_df is None:
            print("  Skipping stage breakdown — no data.")
            return None

        print("  Creating chart 2: Stage Breakdown...")

        top = self.champion_df.head(top_n).copy()

        stages = [
            ('qualified_prob',    'Qualify',    '#a8dadc'),
            ('r16_prob',          'Round 16',   '#457b9d'),
            ('quarterfinal_prob', 'Quarter F',  '#1d3557'),
            ('semifinal_prob',    'Semi Final', '#e9c46a'),
            ('final_prob',        'Final',      '#f4a261'),
            ('champion_prob',     'Champion',   '#e76f51'),
        ]
        available = [(c, l, col) for c, l, col in stages
                     if c in top.columns]

        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor(COLORS['background'])
        ax.set_facecolor(COLORS['background'])

        teams   = top['team'].tolist()
        x       = np.arange(len(teams))
        n_st    = len(available)
        width   = 0.72

        for i, (col, label, color) in enumerate(available):
            offset = (i - n_st / 2) * (width / n_st)
            ax.bar(
                x + offset,
                top[col].values,
                width=width / n_st,
                label=label,
                color=color,
                edgecolor='white',
                linewidth=0.4
            )

        ax.set_xticks(x)
        ax.set_xticklabels(teams, rotation=28, ha='right', fontsize=9.5)
        ax.set_ylabel('Probability (%)', fontsize=11)
        ax.set_title(
            f'Top {top_n} Teams — Stage-by-Stage Tournament Probabilities',
            fontsize=14, fontweight='bold',
            color=COLORS['primary'], pad=14
        )
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax.yaxis.grid(True, alpha=0.25, linestyle='--')
        ax.set_axisbelow(True)

        plt.tight_layout()
        path = self.output_dir / "top10_stage_breakdown.png"
        plt.savefig(path, dpi=150, bbox_inches='tight',
                    facecolor=COLORS['background'])
        plt.close()
        print(f"    Saved: {path}")
        return str(path)

    # ----------------------------------------------------------------
    # CHART 3: ELO RATINGS
    # ----------------------------------------------------------------

    def chart_elo_ratings(self, top_n: int = 20) -> str:
        if self.elo_df is None:
            print("  Skipping Elo chart — no data.")
            return None

        print("  Creating chart 3: Elo Ratings...")

        top = self.elo_df.head(top_n).sort_values('elo', ascending=True)

        fig, ax = plt.subplots(figsize=(12, 9))
        fig.patch.set_facecolor(COLORS['background'])
        ax.set_facecolor(COLORS['background'])

        bar_colors = []
        for elo in top['elo']:
            if elo >= 1750:   bar_colors.append(COLORS['primary'])
            elif elo >= 1650: bar_colors.append(COLORS['secondary'])
            elif elo >= 1600: bar_colors.append(COLORS['accent'])
            else:             bar_colors.append('#74c69d')

        bars = ax.barh(
            range(len(top)),
            top['elo'].values,
            color=bar_colors,
            height=0.65,
            edgecolor='white',
            linewidth=0.5
        )

        for bar, val in zip(bars, top['elo'].values):
            ax.text(
                val + 1,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.0f}',
                va='center', ha='left',
                fontsize=9, fontweight='bold',
                color=COLORS['text']
            )

        ax.axvline(x=1500, color=COLORS['red'],
                   linewidth=1.5, linestyle='--',
                   alpha=0.65, label='Average (1500)')
        ax.legend(fontsize=9)

        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top['team'].tolist(), fontsize=10)
        ax.set_xlabel('Elo Rating', fontsize=12)
        ax.set_title(
            f'Top {top_n} National Teams — Elo Ratings\n'
            'Computed from matches 2000–2024',
            fontsize=15, fontweight='bold',
            color=COLORS['primary'], pad=15
        )

        min_val = top['elo'].min()
        ax.set_xlim(min_val - 60, top['elo'].max() + 90)
        ax.xaxis.grid(True, alpha=0.25, linestyle='--')
        ax.set_axisbelow(True)

        plt.tight_layout()
        path = self.output_dir / "elo_ratings.png"
        plt.savefig(path, dpi=150, bbox_inches='tight',
                    facecolor=COLORS['background'])
        plt.close()
        print(f"    Saved: {path}")
        return str(path)

    # ----------------------------------------------------------------
    # CHART 4: PROBABILITY HEATMAP
    # ----------------------------------------------------------------

    def chart_probability_heatmap(self, top_n: int = 20) -> str:
        if self.champion_df is None:
            print("  Skipping heatmap — no data.")
            return None

        print("  Creating chart 4: Probability Heatmap...")

        top = self.champion_df.head(top_n).copy()

        stage_cols = [
            'qualified_prob', 'r16_prob', 'quarterfinal_prob',
            'semifinal_prob', 'final_prob', 'champion_prob'
        ]
        col_labels = [
            'Qualify', 'Round 16', 'Quarter F',
            'Semi Final', 'Final', 'Champion'
        ]

        available      = [c for c in stage_cols if c in top.columns]
        avail_labels   = [col_labels[stage_cols.index(c)] for c in available]
        matrix         = top[available].values
        teams          = top['team'].tolist()

        fig, ax = plt.subplots(figsize=(12, 9))
        fig.patch.set_facecolor(COLORS['background'])

        cmap = LinearSegmentedColormap.from_list(
            'wc_green',
            ['#f0f7f0', '#b7e4c7', '#52b788', '#2d6a4f', '#1b4332']
        )

        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=100)

        for i in range(len(teams)):
            for j in range(len(available)):
                val   = matrix[i, j]
                color = 'white' if val > 45 else COLORS['text']
                ax.text(j, i, f'{val:.1f}%',
                        ha='center', va='center',
                        fontsize=8.5, fontweight='bold', color=color)

        ax.set_xticks(range(len(available)))
        ax.set_xticklabels(avail_labels, fontsize=10.5, fontweight='bold')
        ax.set_yticks(range(len(teams)))
        ax.set_yticklabels(teams, fontsize=10)

        cbar = plt.colorbar(im, ax=ax, shrink=0.78, pad=0.02)
        cbar.set_label('Probability (%)', fontsize=10)

        ax.set_title(
            f'Stage Probability Heatmap — Top {top_n} Teams',
            fontsize=15, fontweight='bold',
            color=COLORS['primary'], pad=15
        )

        plt.tight_layout()
        path = self.output_dir / "probability_heatmap.png"
        plt.savefig(path, dpi=150, bbox_inches='tight',
                    facecolor=COLORS['background'])
        plt.close()
        print(f"    Saved: {path}")
        return str(path)

    # ----------------------------------------------------------------
    # CHART 5: GROUP TABLES
    # ----------------------------------------------------------------

    def chart_group_tables(self) -> list:
        if self.groups_df is None or self.champion_df is None:
            print("  Skipping group tables — missing data.")
            return []

        print("  Creating chart 5: Group Tables...")

        groups      = self.groups_df.groupby('group')
        group_names = sorted(groups.groups.keys())
        n_groups    = len(group_names)

        cols_per_row = 4
        n_rows       = (n_groups + cols_per_row - 1) // cols_per_row

        fig = plt.figure(figsize=(20, n_rows * 4 + 1))
        fig.patch.set_facecolor(COLORS['background'])
        fig.suptitle(
            '2026 FIFA World Cup — Group Stage Draw & Predictions',
            fontsize=17, fontweight='bold',
            color=COLORS['primary'], y=0.98
        )

        for idx, group_name in enumerate(group_names):
            group_teams = groups.get_group(group_name)['team'].tolist()
            ax = fig.add_subplot(n_rows, cols_per_row, idx + 1)
            ax.axis('off')

            table_data = []
            for team in group_teams:
                row_data = self.champion_df[
                    self.champion_df['team'] == team
                ]
                champ = row_data.iloc[0]['champion_prob'] \
                    if len(row_data) > 0 else 0.0
                qual  = row_data.iloc[0].get('qualified_prob', 0.0) \
                    if len(row_data) > 0 else 0.0
                table_data.append([team, f'{qual:.1f}%', f'{champ:.1f}%'])

            table_data.sort(
                key=lambda x: float(x[2].replace('%', '')),
                reverse=True
            )

            tbl = ax.table(
                cellText=table_data,
                colLabels=['Team', 'Qualify%', 'Win%'],
                cellLoc='center',
                loc='center'
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            tbl.scale(1, 1.6)

            # Header
            for j in range(3):
                cell = tbl[0, j]
                cell.set_facecolor(COLORS['primary'])
                cell.set_text_props(color='white', fontweight='bold')
                cell.set_edgecolor('white')

            # Data rows
            for i in range(1, len(table_data) + 1):
                for j in range(3):
                    cell = tbl[i, j]
                    cell.set_facecolor(
                        '#d8f3dc' if i <= 2 else
                        ('#f0f7f0' if i % 2 == 0 else '#ffffff')
                    )
                    cell.set_edgecolor('#dddddd')

            ax.set_title(
                f'Group {group_name}',
                fontsize=12, fontweight='bold',
                color=COLORS['primary'], pad=8
            )

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        path = self.output_dir / "all_groups.png"
        plt.savefig(path, dpi=130, bbox_inches='tight',
                    facecolor=COLORS['background'])
        plt.close()
        print(f"    Saved: {path}")
        return [str(path)]

    # ----------------------------------------------------------------
    # CHART 6: TOP CONTENDERS
    # ----------------------------------------------------------------

    def chart_top_contenders(self, top_n: int = 8) -> str:
        if self.champion_df is None:
            print("  Skipping top contenders — no data.")
            return None

        print("  Creating chart 6: Top Contenders...")

        top = self.champion_df.head(top_n).copy()

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.patch.set_facecolor(COLORS['background'])
        fig.suptitle(
            '2026 FIFA World Cup — Top Contenders Detail',
            fontsize=16, fontweight='bold',
            color=COLORS['primary'], y=1.01
        )

        axes_flat   = axes.flatten()
        medal_colors = {
            0: '#fff3cd', 1: '#e9ecef', 2: '#f5e6d3'
        }

        stage_cols   = [
            'qualified_prob', 'r16_prob', 'quarterfinal_prob',
            'semifinal_prob', 'final_prob', 'champion_prob'
        ]
        stage_labels = ['Qualify', 'R16', 'QF', 'SF', 'Final', 'Champion']
        bar_colors   = [
            '#a8dadc', '#457b9d', '#1d3557',
            '#e9c46a', '#f4a261', '#e76f51'
        ]

        for idx, (_, row) in enumerate(top.iterrows()):
            if idx >= len(axes_flat):
                break

            ax = axes_flat[idx]
            ax.set_facecolor(medal_colors.get(idx, COLORS['background']))

            vals = [row.get(c, 0.0) for c in stage_cols]

            ax.barh(
                range(len(stage_labels)),
                vals,
                color=bar_colors,
                height=0.6,
                edgecolor='white'
            )

            for i, val in enumerate(vals):
                ax.text(
                    val + 0.3, i,
                    f'{val:.1f}%',
                    va='center', ha='left',
                    fontsize=7, fontweight='bold'
                )

            ax.set_yticks(range(len(stage_labels)))
            ax.set_yticklabels(stage_labels, fontsize=8)
            ax.set_xlim(0, 115)
            ax.set_xlabel('Probability (%)', fontsize=7)

            rank_symbols = ['🥇','🥈','🥉','4️⃣','5️⃣','6️⃣','7️⃣','8️⃣']
            ax.set_title(
                f'{rank_symbols[idx]} {row["team"]}\n'
                f'Champion: {row["champion_prob"]:.1f}%',
                fontsize=10, fontweight='bold',
                color=COLORS['primary'], pad=6
            )
            ax.xaxis.grid(True, alpha=0.2)
            ax.set_axisbelow(True)

        plt.tight_layout()
        path = self.output_dir / "top_contenders.png"
        plt.savefig(path, dpi=150, bbox_inches='tight',
                    facecolor=COLORS['background'])
        plt.close()
        print(f"    Saved: {path}")
        return str(path)

    # ----------------------------------------------------------------
    # CHART 7: DARK HORSES
    # ----------------------------------------------------------------

    def chart_dark_horses(self) -> str:
        if self.champion_df is None:
            print("  Skipping dark horses — no data.")
            return None

        print("  Creating chart 7: Dark Horses...")

        df = self.champion_df.copy().reset_index(drop=True)

        if 'quarterfinal_prob' not in df.columns:
            print("  Skipping dark horses — quarterfinal_prob column missing.")
            return None

        df['dh_score'] = (
            df['quarterfinal_prob'] / (df['champion_prob'] + 0.5)
        )
        dark = df[df.index >= 4].nlargest(8, 'dh_score')

        if len(dark) == 0:
            print("  No dark horses found.")
            return None

        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor(COLORS['background'])
        ax.set_facecolor(COLORS['background'])

        teams = dark['team'].tolist()
        x     = np.arange(len(teams))
        w     = 0.32

        b1 = ax.bar(x - w / 2, dark['champion_prob'].values,
                    w, label='Champion %',
                    color=COLORS['primary'], edgecolor='white')
        b2 = ax.bar(x + w / 2, dark['quarterfinal_prob'].values,
                    w, label='Quarterfinal %',
                    color=COLORS['highlight'], edgecolor='white')

        for bar in list(b1) + list(b2):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.15,
                f'{h:.1f}%',
                ha='center', va='bottom',
                fontsize=8, fontweight='bold'
            )

        ax.set_xticks(x)
        ax.set_xticklabels(teams, rotation=22, ha='right', fontsize=10)
        ax.set_ylabel('Probability (%)', fontsize=11)
        ax.set_title(
            '⚡ Dark Horse Teams\n'
            'High Quarterfinal % vs Low Champion % — Could Cause Upsets',
            fontsize=13, fontweight='bold',
            color=COLORS['primary'], pad=12
        )
        ax.legend(fontsize=10)
        ax.yaxis.grid(True, alpha=0.25, linestyle='--')
        ax.set_axisbelow(True)

        plt.tight_layout()
        path = self.output_dir / "dark_horses.png"
        plt.savefig(path, dpi=150, bbox_inches='tight',
                    facecolor=COLORS['background'])
        plt.close()
        print(f"    Saved: {path}")
        return str(path)

    # ----------------------------------------------------------------
    # RUN ALL CHARTS
    # ----------------------------------------------------------------

    def create_all_charts(self) -> dict:
        print("=" * 55)
        print("  GENERATING ALL CHARTS")
        print("=" * 55 + "\n")

        saved = {}
        saved['champion_probabilities'] = self.chart_champion_probabilities()
        saved['stage_breakdown']        = self.chart_stage_breakdown()
        saved['elo_ratings']            = self.chart_elo_ratings()
        saved['probability_heatmap']    = self.chart_probability_heatmap()
        saved['group_tables']           = self.chart_group_tables()
        saved['top_contenders']         = self.chart_top_contenders()
        saved['dark_horses']            = self.chart_dark_horses()

        print("\n" + "=" * 55)
        print("  CHARTS COMPLETE")
        print("=" * 55)
        for name, path in saved.items():
            if path and path != []:
                print(f"  ✅ {name}")
            else:
                print(f"  ⚠️  {name} — skipped")

        return saved

    def get_feature_columns(self) -> list:
        return [
            'home_elo', 'away_elo', 'elo_difference',
            'home_form', 'away_form', 'form_difference',
            'home_goals_scored_avg', 'home_goals_conceded_avg',
            'away_goals_scored_avg', 'away_goals_conceded_avg',
            'home_win_rate', 'away_win_rate',
            'home_strength', 'away_strength',
            'strength_difference', 'is_neutral',
        ]


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  VISUALIZE — GENERATING ALL CHARTS")
    print("=" * 55)

    viz = Visualizer(output_dir="outputs/charts")
    viz.load_data()
    viz.create_all_charts()

    print("\n✅ All charts saved to outputs/charts/")
    print("Open the folder in VS Code Explorer to preview PNG files.")