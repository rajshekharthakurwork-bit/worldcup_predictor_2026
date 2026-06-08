"""
dashboard.py
============
PURPOSE: Generate a beautiful HTML dashboard that opens in your browser
         and shows all charts, tables, and predictions in one place.

HOW TO RUN:
    python dashboard.py

WHAT IT PRODUCES:
    outputs/dashboard.html  ← Opens automatically in your browser

WHAT YOU SEE:
    - Champion probability bar chart
    - Stage probability heatmap table
    - Group standings tables
    - Elo ratings chart
    - Top contenders cards
    - Dark horse teams
    - Match predictor output
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import base64
import io
import webbrowser
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path


# ================================================================
# COLOR THEME
# ================================================================

COLORS = {
    'primary':   '#1a472a',
    'secondary': '#2d6a4f',
    'accent':    '#52b788',
    'highlight': '#f4a261',
    'gold':      '#f7b731',
    'silver':    '#bdc3c7',
    'bronze':    '#cd7f32',
    'bg':        '#f0f4f0',
    'white':     '#ffffff',
    'text':      '#2c3e50',
    'red':       '#e74c3c',
    'light':     '#d8f3dc',
}


# ================================================================
# HELPER: Convert matplotlib figure to base64 image
# ================================================================

def fig_to_base64(fig) -> str:
    """
    Convert a matplotlib figure to a base64 string.
    This lets us embed charts directly inside the HTML file
    without needing separate image files.

    Parameters:
        fig : matplotlib Figure object

    Returns:
        str : base64 encoded PNG string
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130,
                bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


# ================================================================
# CHART BUILDERS
# ================================================================

def build_champion_chart(df: pd.DataFrame, top_n: int = 20) -> str:
    """
    Build champion probability horizontal bar chart.

    Parameters:
        df    : Champion probabilities DataFrame
        top_n : Number of teams to show

    Returns:
        str : base64 encoded chart image
    """
    top = df.head(top_n).sort_values('champion_prob', ascending=True)

    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor('#f8fff8')
    ax.set_facecolor('#f8fff8')

    n = len(top)

    # Color gradient
    def get_color(i, n):
        ratio = i / max(n - 1, 1)
        if ratio > 0.85:   return COLORS['primary']
        elif ratio > 0.65: return COLORS['secondary']
        elif ratio > 0.45: return COLORS['accent']
        elif ratio > 0.25: return '#74c69d'
        else:              return '#b7e4c7'

    bar_colors = [get_color(i, n) for i in range(n)]

    bars = ax.barh(
        range(n),
        top['champion_prob'].values,
        color=bar_colors,
        height=0.68,
        edgecolor='white',
        linewidth=0.8
    )

    # Value labels on bars
    for bar, val in zip(bars, top['champion_prob'].values):
        ax.text(
            val + 0.08,
            bar.get_y() + bar.get_height() / 2,
            f'{val:.1f}%',
            va='center', ha='left',
            fontsize=9, fontweight='bold',
            color=COLORS['text']
        )

    # Team labels with medals
    teams_list = top['team'].tolist()
    total = len(df)
    labels = []
    for i, team in enumerate(teams_list):
        rank = total - i
        if rank == 1:   labels.append(f'🥇 {team}')
        elif rank == 2: labels.append(f'🥈 {team}')
        elif rank == 3: labels.append(f'🥉 {team}')
        else:           labels.append(f'     {team}')

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.xaxis.grid(True, alpha=0.25, linestyle='--')
    ax.set_axisbelow(True)
    ax.set_xlabel('Champion Probability (%)', fontsize=11, labelpad=8)
    ax.set_title(
        '2026 FIFA World Cup\nPredicted Champion Probabilities',
        fontsize=15, fontweight='bold',
        color=COLORS['primary'], pad=16
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.text(
        0.5, 0.01,
        'Based on 10,000 Monte Carlo simulations using Elo ratings & ML model',
        ha='center', fontsize=8, color='gray', style='italic'
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    return fig_to_base64(fig)


def build_elo_chart(elo_df: pd.DataFrame, top_n: int = 20) -> str:
    """
    Build Elo ratings horizontal bar chart.

    Parameters:
        elo_df : DataFrame with team and elo columns
        top_n  : Number of teams to show

    Returns:
        str : base64 encoded chart image
    """
    top = elo_df.head(top_n).sort_values('elo', ascending=True)

    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor('#f8fff8')
    ax.set_facecolor('#f8fff8')

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
        linewidth=0.6
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

    ax.axvline(x=1500, color=COLORS['red'], linewidth=1.5,
               linestyle='--', alpha=0.6, label='Average (1500)')
    ax.legend(fontsize=9)

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top['team'].tolist(), fontsize=10)
    ax.set_xlabel('Elo Rating', fontsize=11)
    ax.set_title(
        f'Top {top_n} National Teams — Elo Ratings\n'
        'Computed from matches 2000–2024',
        fontsize=14, fontweight='bold',
        color=COLORS['primary'], pad=14
    )

    min_val = top['elo'].min()
    ax.set_xlim(min_val - 60, top['elo'].max() + 90)
    ax.xaxis.grid(True, alpha=0.25, linestyle='--')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig_to_base64(fig)


def build_stage_breakdown_chart(df: pd.DataFrame, top_n: int = 12) -> str:
    """
    Build grouped bar chart showing probability at each tournament stage.

    Parameters:
        df    : Champion probabilities DataFrame
        top_n : Number of teams to show

    Returns:
        str : base64 encoded chart image
    """
    top = df.head(top_n).copy()

    stages = [
        ('qualified_prob',    'Qualify',    '#a8dadc'),
        ('r16_prob',          'Round 16',   '#457b9d'),
        ('quarterfinal_prob', 'Quarter F',  '#1d3557'),
        ('semifinal_prob',    'Semi Final', '#e9c46a'),
        ('final_prob',        'Final',      '#f4a261'),
        ('champion_prob',     'Champion',   '#e76f51'),
    ]

    available = [(c, l, col) for c, l, col in stages if c in top.columns]

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('#f8fff8')
    ax.set_facecolor('#f8fff8')

    teams = top['team'].tolist()
    x = np.arange(len(teams))
    n_stages = len(available)
    width = 0.7

    for i, (col, label, color) in enumerate(available):
        offset = (i - n_stages / 2) * (width / n_stages)
        ax.bar(
            x + offset,
            top[col].values,
            width=width / n_stages,
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
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig_to_base64(fig)


def build_heatmap_chart(df: pd.DataFrame, top_n: int = 20) -> str:
    """
    Build heatmap showing all teams × all stages probability matrix.

    Parameters:
        df    : Champion probabilities DataFrame
        top_n : Number of teams to show

    Returns:
        str : base64 encoded chart image
    """
    top = df.head(top_n).copy()

    stage_cols = [
        'qualified_prob', 'r16_prob', 'quarterfinal_prob',
        'semifinal_prob', 'final_prob', 'champion_prob'
    ]
    col_labels = [
        'Qualify', 'Round 16', 'Quarter F',
        'Semi Final', 'Final', 'Champion'
    ]

    available_cols = [c for c in stage_cols if c in top.columns]
    available_labels = [
        col_labels[stage_cols.index(c)] for c in available_cols
    ]

    matrix = top[available_cols].values
    teams = top['team'].tolist()

    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor('#f8fff8')

    cmap = LinearSegmentedColormap.from_list(
        'wc_green',
        ['#f0f7f0', '#b7e4c7', '#52b788', '#2d6a4f', '#1b4332']
    )

    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=100)

    for i in range(len(teams)):
        for j in range(len(available_cols)):
            val = matrix[i, j]
            color = 'white' if val > 45 else COLORS['text']
            ax.text(j, i, f'{val:.1f}%',
                    ha='center', va='center',
                    fontsize=8.5, fontweight='bold', color=color)

    ax.set_xticks(range(len(available_cols)))
    ax.set_xticklabels(available_labels, fontsize=10.5, fontweight='bold')
    ax.set_yticks(range(len(teams)))
    ax.set_yticklabels(teams, fontsize=10)

    cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label('Probability (%)', fontsize=10)

    ax.set_title(
        f'Stage Probability Heatmap — Top {top_n} Teams',
        fontsize=14, fontweight='bold',
        color=COLORS['primary'], pad=14
    )

    plt.tight_layout()
    return fig_to_base64(fig)


def build_dark_horse_chart(df: pd.DataFrame) -> str:
    """
    Build dark horse teams chart (high QF% but low champion%).

    Parameters:
        df : Champion probabilities DataFrame

    Returns:
        str : base64 encoded chart image
    """
    data = df.copy().reset_index(drop=True)

    if 'quarterfinal_prob' not in data.columns:
        return None

    data['dark_horse_score'] = (
        data['quarterfinal_prob'] / (data['champion_prob'] + 0.5)
    )
    dark_horses = data[data.index >= 4].nlargest(8, 'dark_horse_score')

    if len(dark_horses) == 0:
        return None

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#f8fff8')
    ax.set_facecolor('#f8fff8')

    teams = dark_horses['team'].tolist()
    x = np.arange(len(teams))
    w = 0.32

    b1 = ax.bar(x - w / 2, dark_horses['champion_prob'].values,
                w, label='Champion %', color=COLORS['primary'],
                edgecolor='white')
    b2 = ax.bar(x + w / 2, dark_horses['quarterfinal_prob'].values,
                w, label='Quarterfinal %', color=COLORS['highlight'],
                edgecolor='white')

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.15,
                f'{h:.1f}%', ha='center', va='bottom',
                fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(teams, rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('Probability (%)', fontsize=11)
    ax.set_title(
        '⚡ Dark Horse Teams — High QF% vs Low Champion%\n'
        'Teams that could cause upsets',
        fontsize=13, fontweight='bold',
        color=COLORS['primary'], pad=12
    )
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, alpha=0.25, linestyle='--')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig_to_base64(fig)


# ================================================================
# HTML TABLE BUILDERS
# ================================================================

def build_top20_html_table(df: pd.DataFrame) -> str:
    """
    Build HTML table for top 20 champion probabilities.

    Parameters:
        df : Champion probabilities DataFrame

    Returns:
        str : HTML string for the table
    """
    top = df.head(20)
    medals = {1: '🥇', 2: '🥈', 3: '🥉'}

    rows = ""
    for idx, (_, row) in enumerate(top.iterrows()):
        rank = idx + 1
        medal = medals.get(rank, str(rank))

        # Color rows alternately
        bg = '#f0f7f0' if idx % 2 == 0 else '#ffffff'

        # Progress bar for champion prob
        bar_width = int(row['champion_prob'] / df['champion_prob'].max() * 120)
        bar_html = (
            f'<div style="display:flex;align-items:center;gap:6px;">'
            f'<div style="width:{bar_width}px;height:14px;'
            f'background:{COLORS["primary"]};border-radius:3px;"></div>'
            f'<span style="font-weight:bold">{row["champion_prob"]:.1f}%</span>'
            f'</div>'
        )

        rows += f"""
        <tr style="background:{bg};">
            <td style="text-align:center;font-size:1.1em;">{medal}</td>
            <td style="font-weight:600;color:{COLORS['primary']};">
                {row['team']}
            </td>
            <td>{bar_html}</td>
            <td style="text-align:center;">{row['final_prob']:.1f}%</td>
            <td style="text-align:center;">{row['semifinal_prob']:.1f}%</td>
            <td style="text-align:center;">{row['quarterfinal_prob']:.1f}%</td>
            <td style="text-align:center;">{row.get('r16_prob', 0):.1f}%</td>
        </tr>"""

    return f"""
    <table style="width:100%;border-collapse:collapse;font-size:0.95em;">
        <thead>
            <tr style="background:{COLORS['primary']};color:white;">
                <th style="padding:10px;">Rank</th>
                <th style="padding:10px;text-align:left;">Team</th>
                <th style="padding:10px;text-align:left;">Champion %</th>
                <th style="padding:10px;">Final %</th>
                <th style="padding:10px;">Semi %</th>
                <th style="padding:10px;">QF %</th>
                <th style="padding:10px;">R16 %</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>"""


def build_groups_html(groups_df: pd.DataFrame,
                       champion_df: pd.DataFrame,
                       elo_df: pd.DataFrame) -> str:
    """
    Build HTML group stage tables for all groups.

    Parameters:
        groups_df   : Group assignments DataFrame
        champion_df : Champion probabilities DataFrame
        elo_df      : Elo ratings DataFrame

    Returns:
        str : HTML string containing all group tables
    """
    elo_map = dict(zip(elo_df['team'], elo_df['elo']))

    def get_champ_prob(team):
        row = champion_df[champion_df['team'] == team]
        return row.iloc[0]['champion_prob'] if len(row) > 0 else 0.0

    def get_qual_prob(team):
        row = champion_df[champion_df['team'] == team]
        if len(row) > 0 and 'qualified_prob' in champion_df.columns:
            return row.iloc[0]['qualified_prob']
        return 0.0

    all_groups_html = ""
    groups = groups_df.groupby('group')

    for group_name in sorted(groups.groups.keys()):
        group_teams = groups.get_group(group_name)['team'].tolist()

        team_data = []
        for team in group_teams:
            team_data.append({
                'team': team,
                'elo': elo_map.get(team, 1500),
                'qualify': get_qual_prob(team),
                'champion': get_champ_prob(team),
            })

        team_data.sort(key=lambda x: x['elo'], reverse=True)

        rows_html = ""
        for i, td in enumerate(team_data):
            bg    = '#d8f3dc' if i < 2 else '#ffffff'
            badge = '✅ Q' if i < 2 else '   '
            rows_html += f"""
            <tr style="background:{bg};">
                <td style="padding:7px 10px;font-weight:{'700' if i<2 else '400'};
                    color:{COLORS['primary'] if i<2 else COLORS['text']};">
                    {badge} {td['team']}
                </td>
                <td style="text-align:center;padding:7px;">{td['elo']:.0f}</td>
                <td style="text-align:center;padding:7px;">{td['qualify']:.1f}%</td>
                <td style="text-align:center;padding:7px;font-weight:bold;">
                    {td['champion']:.1f}%
                </td>
            </tr>"""

        all_groups_html += f"""
        <div style="break-inside:avoid;margin-bottom:18px;">
            <table style="width:100%;border-collapse:collapse;
                          font-size:0.88em;border-radius:8px;
                          overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.1);">
                <thead>
                    <tr style="background:{COLORS['primary']};color:white;">
                        <th colspan="4" style="padding:9px;text-align:left;
                            font-size:1.05em;">
                            ⚽ Group {group_name}
                        </th>
                    </tr>
                    <tr style="background:{COLORS['secondary']};color:white;">
                        <th style="padding:7px 10px;text-align:left;">Team</th>
                        <th style="padding:7px;text-align:center;">Elo</th>
                        <th style="padding:7px;text-align:center;">Qualify%</th>
                        <th style="padding:7px;text-align:center;">WC Win%</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
            <div style="font-size:0.75em;color:gray;margin-top:3px;
                        padding-left:4px;">
                ✅ Q = Predicted to qualify
            </div>
        </div>"""

    return all_groups_html


# ================================================================
# FULL HTML PAGE BUILDER
# ================================================================

def build_html_dashboard(
        champion_df: pd.DataFrame,
        elo_df: pd.DataFrame,
        groups_df: pd.DataFrame) -> str:
    """
    Build the complete HTML dashboard page.

    Parameters:
        champion_df : Champion probabilities DataFrame
        elo_df      : Elo ratings DataFrame
        groups_df   : Group assignments DataFrame

    Returns:
        str : Complete HTML page as string
    """

    print("Building charts...")

    print("  Building champion probability chart...")
    champ_chart = build_champion_chart(champion_df)

    print("  Building Elo ratings chart...")
    elo_chart = build_elo_chart(elo_df)

    print("  Building stage breakdown chart...")
    stage_chart = build_stage_breakdown_chart(champion_df)

    print("  Building heatmap chart...")
    heatmap_chart = build_heatmap_chart(champion_df)

    print("  Building dark horse chart...")
    dark_horse = build_dark_horse_chart(champion_df)

    print("  Building HTML tables...")
    top20_table = build_top20_html_table(champion_df)
    groups_html = build_groups_html(groups_df, champion_df, elo_df)

    # Top winner info
    winner = champion_df.iloc[0]
    second = champion_df.iloc[1]
    third  = champion_df.iloc[2]

    # Dark horse chart section
    dark_horse_section = ""
    if dark_horse:
        dark_horse_section = f"""
        <div class="section">
            <h2>⚡ Dark Horse Teams</h2>
            <p class="subtitle">
                Teams that could cause major upsets — high quarterfinal
                probability relative to their champion probability.
            </p>
            <div class="chart-container">
                <img src="data:image/png;base64,{dark_horse}"
                     alt="Dark Horses" class="chart-img">
            </div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>2026 FIFA World Cup Predictor — Dashboard</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #edf2ee;
            color: {COLORS['text']};
            line-height: 1.6;
        }}

        /* ── HEADER ── */
        .header {{
            background: linear-gradient(135deg, {COLORS['primary']} 0%,
                                        {COLORS['secondary']} 100%);
            color: white;
            text-align: center;
            padding: 48px 20px 36px;
        }}
        .header h1 {{
            font-size: 2.4em;
            font-weight: 800;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }}
        .header p {{
            font-size: 1.05em;
            opacity: 0.85;
            max-width: 680px;
            margin: 0 auto;
        }}
        .header .badge {{
            display: inline-block;
            background: rgba(255,255,255,0.2);
            border-radius: 20px;
            padding: 4px 16px;
            margin-top: 14px;
            font-size: 0.88em;
        }}

        /* ── WINNER BANNER ── */
        .winner-banner {{
            background: linear-gradient(135deg, #f7b731, #f4a261);
            padding: 28px 20px;
            text-align: center;
        }}
        .winner-banner h2 {{
            font-size: 1.3em;
            color: #7d4f00;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 2px;
        }}
        .podium {{
            display: flex;
            justify-content: center;
            align-items: flex-end;
            gap: 16px;
            flex-wrap: wrap;
        }}
        .podium-card {{
            background: white;
            border-radius: 12px;
            padding: 20px 28px;
            text-align: center;
            box-shadow: 0 4px 16px rgba(0,0,0,0.12);
            min-width: 160px;
        }}
        .podium-card.gold   {{ border-top: 5px solid #f7b731; }}
        .podium-card.silver {{ border-top: 5px solid #bdc3c7; }}
        .podium-card.bronze {{ border-top: 5px solid #cd7f32; }}
        .podium-card .medal {{ font-size: 2.4em; }}
        .podium-card .team-name {{
            font-size: 1.25em;
            font-weight: 800;
            color: {COLORS['primary']};
            margin: 6px 0 4px;
        }}
        .podium-card .prob {{
            font-size: 1.7em;
            font-weight: 900;
            color: {COLORS['text']};
        }}
        .podium-card .label {{
            font-size: 0.78em;
            color: gray;
            margin-top: 2px;
        }}

        /* ── MAIN CONTENT ── */
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 28px 20px;
        }}

        /* ── SECTION ── */
        .section {{
            background: white;
            border-radius: 14px;
            padding: 30px;
            margin-bottom: 28px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        }}
        .section h2 {{
            font-size: 1.5em;
            font-weight: 700;
            color: {COLORS['primary']};
            margin-bottom: 6px;
            padding-bottom: 10px;
            border-bottom: 3px solid {COLORS['accent']};
        }}
        .subtitle {{
            color: gray;
            font-size: 0.92em;
            margin-bottom: 20px;
            margin-top: 6px;
        }}

        /* ── CHARTS ── */
        .chart-container {{
            text-align: center;
            margin-top: 16px;
        }}
        .chart-img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }}
        .chart-row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-top: 16px;
        }}
        @media (max-width: 768px) {{
            .chart-row {{ grid-template-columns: 1fr; }}
        }}

        /* ── TABLE ── */
        table {{ border-radius: 8px; overflow: hidden; }}
        th, td {{ padding: 8px 12px; }}
        thead tr {{ position: sticky; top: 0; }}

        /* ── GROUPS GRID ── */
        .groups-grid {{
            column-count: 3;
            column-gap: 20px;
            margin-top: 16px;
        }}
        @media (max-width: 900px) {{
            .groups-grid {{ column-count: 2; }}
        }}
        @media (max-width: 600px) {{
            .groups-grid {{ column-count: 1; }}
        }}

        /* ── STATS CARDS ── */
        .stat-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 28px;
        }}
        .stat-card {{
            background: white;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.07);
            border-left: 4px solid {COLORS['accent']};
        }}
        .stat-card .stat-value {{
            font-size: 2em;
            font-weight: 900;
            color: {COLORS['primary']};
        }}
        .stat-card .stat-label {{
            font-size: 0.85em;
            color: gray;
            margin-top: 4px;
        }}

        /* ── FOOTER ── */
        .footer {{
            background: {COLORS['primary']};
            color: rgba(255,255,255,0.75);
            text-align: center;
            padding: 24px;
            font-size: 0.88em;
            margin-top: 16px;
        }}

        /* ── NAV TABS ── */
        .nav {{
            background: white;
            padding: 0 20px;
            display: flex;
            gap: 0;
            overflow-x: auto;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            position: sticky;
            top: 0;
            z-index: 100;
        }}
        .nav a {{
            padding: 14px 20px;
            text-decoration: none;
            color: {COLORS['text']};
            font-weight: 600;
            font-size: 0.9em;
            border-bottom: 3px solid transparent;
            white-space: nowrap;
            transition: all 0.2s;
        }}
        .nav a:hover {{
            color: {COLORS['primary']};
            border-bottom-color: {COLORS['accent']};
        }}
    </style>
</head>

<body>

<!-- ── HEADER ── -->
<div class="header">
    <h1>🌍 2026 FIFA World Cup Predictor</h1>
    <p>
        Machine Learning predictions using Elo ratings,
        feature engineering, Random Forest classification,
        and 10,000 Monte Carlo simulations.
    </p>
    <div class="badge">
        📊 10,000 Simulations · 48 Teams · 6 Tournament Stages
    </div>
</div>

<!-- ── NAV ── -->
<nav class="nav">
    <a href="#winner">🏆 Winner</a>
    <a href="#probabilities">📊 Probabilities</a>
    <a href="#charts">📈 Charts</a>
    <a href="#heatmap">🔥 Heatmap</a>
    <a href="#groups">⚽ Groups</a>
    <a href="#darkhorses">⚡ Dark Horses</a>
    <a href="#elo">🎯 Elo Ratings</a>
</nav>

<!-- ── WINNER PODIUM ── -->
<div class="winner-banner" id="winner">
    <h2>🏆 Predicted World Cup Podium</h2>
    <div class="podium">
        <div class="podium-card silver">
            <div class="medal">🥈</div>
            <div class="team-name">{second['team']}</div>
            <div class="prob">{second['champion_prob']:.1f}%</div>
            <div class="label">Champion Probability</div>
        </div>
        <div class="podium-card gold">
            <div class="medal">🥇</div>
            <div class="team-name">{winner['team']}</div>
            <div class="prob">{winner['champion_prob']:.1f}%</div>
            <div class="label">Champion Probability</div>
        </div>
        <div class="podium-card bronze">
            <div class="medal">🥉</div>
            <div class="team-name">{third['team']}</div>
            <div class="prob">{third['champion_prob']:.1f}%</div>
            <div class="label">Champion Probability</div>
        </div>
    </div>
</div>

<!-- ── STATS CARDS ── -->
<div class="container">
    <div class="stat-cards">
        <div class="stat-card">
            <div class="stat-value">48</div>
            <div class="stat-label">Teams in Tournament</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">10K</div>
            <div class="stat-label">Monte Carlo Simulations</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">200</div>
            <div class="stat-label">Random Forest Trees</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">16</div>
            <div class="stat-label">ML Features Used</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">
                {champion_df['champion_prob'].max():.1f}%
            </div>
            <div class="stat-label">Highest Win Probability</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">~54%</div>
            <div class="stat-label">Model Test Accuracy</div>
        </div>
    </div>

    <!-- ── CHAMPION PROBABILITIES TABLE ── -->
    <div class="section" id="probabilities">
        <h2>📊 Top 20 Champion Probabilities</h2>
        <p class="subtitle">
            Ranked by champion probability from 10,000 tournament simulations.
            Green rows = predicted to qualify from group stage.
        </p>
        {top20_table}
    </div>

    <!-- ── CHAMPION CHART ── -->
    <div class="section" id="charts">
        <h2>📈 Champion Probability Chart</h2>
        <p class="subtitle">
            Horizontal bar chart of top 20 teams by predicted champion probability.
        </p>
        <div class="chart-container">
            <img src="data:image/png;base64,{champ_chart}"
                 alt="Champion Probabilities" class="chart-img">
        </div>
    </div>

    <!-- ── STAGE BREAKDOWN ── -->
    <div class="section">
        <h2>📊 Stage-by-Stage Breakdown</h2>
        <p class="subtitle">
            How far is each top team predicted to go?
            Each colour represents a different tournament stage.
        </p>
        <div class="chart-container">
            <img src="data:image/png;base64,{stage_chart}"
                 alt="Stage Breakdown" class="chart-img">
        </div>
    </div>

    <!-- ── HEATMAP ── -->
    <div class="section" id="heatmap">
        <h2>🔥 Probability Heatmap</h2>
        <p class="subtitle">
            Darker green = higher probability. Read across each row
            to see how a team's chances drop at each stage.
        </p>
        <div class="chart-container">
            <img src="data:image/png;base64,{heatmap_chart}"
                 alt="Heatmap" class="chart-img">
        </div>
    </div>

    <!-- ── GROUP STAGE ── -->
    <div class="section" id="groups">
        <h2>⚽ Group Stage Draw & Predictions</h2>
        <p class="subtitle">
            Teams sorted by Elo rating within each group.
            ✅ Q = Predicted to qualify for Round of 32.
            Green rows = predicted qualifiers.
        </p>
        <div class="groups-grid">
            {groups_html}
        </div>
    </div>

    <!-- ── DARK HORSES ── -->
    {dark_horse_section}

    <!-- ── ELO RATINGS ── -->
    <div class="section" id="elo">
        <h2>🎯 Elo Ratings — Top 20 Teams</h2>
        <p class="subtitle">
            Elo ratings computed from all international matches
            from 2000 to 2024. The red dashed line marks the
            global average (1500).
        </p>
        <div class="chart-container">
            <img src="data:image/png;base64,{elo_chart}"
                 alt="Elo Ratings" class="chart-img">
        </div>
    </div>

    <!-- ── METHODOLOGY ── -->
    <div class="section">
        <h2>🧠 How This Works</h2>
        <p class="subtitle">Technical methodology overview</p>
        <div style="display:grid;grid-template-columns:1fr 1fr;
                    gap:20px;margin-top:10px;">
            <div>
                <h3 style="color:{COLORS['primary']};margin-bottom:8px;">
                    1. Elo Rating System
                </h3>
                <p style="font-size:0.92em;line-height:1.7;">
                    Every team starts at 1500. After each match, ratings update
                    based on the result and how surprising it was. Beating a
                    stronger team earns more points.
                    <br><br>
                    <code style="background:#f0f0f0;padding:2px 6px;
                                 border-radius:4px;">
                        E = 1 / (1 + 10^((Rb-Ra)/400))
                    </code>
                </p>
            </div>
            <div>
                <h3 style="color:{COLORS['primary']};margin-bottom:8px;">
                    2. Feature Engineering
                </h3>
                <p style="font-size:0.92em;line-height:1.7;">
                    16 features per match: Elo ratings, recent form (last 5),
                    goals scored/conceded averages, win rates,
                    squad strength, and neutral venue flag.
                </p>
            </div>
            <div>
                <h3 style="color:{COLORS['primary']};margin-bottom:8px;">
                    3. Random Forest Model
                </h3>
                <p style="font-size:0.92em;line-height:1.7;">
                    200 decision trees trained on matches before 2022,
                    tested on 2022+ matches. Outputs probabilities for
                    Home Win / Draw / Away Win.
                    Achieves ~54% accuracy (normal for football).
                </p>
            </div>
            <div>
                <h3 style="color:{COLORS['primary']};margin-bottom:8px;">
                    4. Monte Carlo Simulation
                </h3>
                <p style="font-size:0.92em;line-height:1.7;">
                    The full 2026 World Cup (48 teams, 12 groups, 6 knockout
                    rounds) is simulated 10,000 times. Results are randomly
                    sampled from model probabilities, allowing upsets.
                    Champion probability = wins / 10,000.
                </p>
            </div>
        </div>
    </div>

</div>

<!-- ── FOOTER ── -->
<div class="footer">
    <p>
        🌍 2026 FIFA World Cup Predictor &nbsp;·&nbsp;
        Built with Python, scikit-learn, matplotlib &nbsp;·&nbsp;
        Portfolio Project
    </p>
    <p style="margin-top:6px;font-size:0.85em;">
        Data: Kaggle International Football Results 1872–2024 &nbsp;·&nbsp;
        10,000 Monte Carlo simulations &nbsp;·&nbsp;
        Results are probabilistic estimates, not certainties.
    </p>
</div>

</body>
</html>"""

    return html


# ================================================================
# MAIN
# ================================================================

def main():
    """Generate and open the HTML dashboard."""

    print("=" * 60)
    print("  2026 FIFA WORLD CUP — HTML DASHBOARD GENERATOR")
    print("=" * 60)

    # ── Check required files ──
    required = {
        "outputs/champion_probabilities.csv": "Run python run_pipeline.py first",
        "data/processed/elo_ratings.csv":     "Run python -m src.elo first",
        "data/wc2026_groups.csv":             "Create data/wc2026_groups.csv first",
    }

    all_ok = True
    for path, fix in required.items():
        if not Path(path).exists():
            print(f"\n  ❌ Missing: {path}")
            print(f"     Fix: {fix}")
            all_ok = False

    if not all_ok:
        return

    # ── Load data ──
    print("\nLoading data...")
    champion_df = pd.read_csv("outputs/champion_probabilities.csv")
    elo_df      = pd.read_csv("data/processed/elo_ratings.csv")
    groups_df   = pd.read_csv("data/wc2026_groups.csv")

    print(f"  Champion data : {len(champion_df)} teams")
    print(f"  Elo data      : {len(elo_df)} teams")
    print(f"  Groups        : {len(groups_df)} teams in "
          f"{groups_df['group'].nunique()} groups")

    # ── Build dashboard ──
    print("\nBuilding dashboard (takes ~30 seconds)...")
    html = build_html_dashboard(champion_df, elo_df, groups_df)

    # ── Save HTML file ──
    output_path = Path("outputs/dashboard.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding='utf-8')

    print(f"\n  ✅ Dashboard saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.0f} KB")

    # ── Open in browser ──
    abs_path = output_path.resolve()
    browser_url = f"file:///{abs_path}".replace("\\", "/")
    print(f"\n  Opening in browser: {browser_url}")
    webbrowser.open(browser_url)

    print("\n  If browser did not open automatically:")
    print(f"  1. Open your browser")
    print(f"  2. Press Ctrl+O (or File → Open)")
    print(f"  3. Navigate to: {abs_path}")

    print("\n✅ Dashboard complete!")


if __name__ == "__main__":
    main()