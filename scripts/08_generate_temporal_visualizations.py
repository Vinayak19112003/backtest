"""
═══════════════════════════════════════════════════════════════════════
08_generate_temporal_visualizations.py
Temporal Optimization Visualization Suite — 7 Professional Charts
Strategy Code: ALPHA-BTC-15M-v2.0
═══════════════════════════════════════════════════════════════════════
Generates heatmaps and bar charts for the temporal optimization data.
All outputs go to visualizations/temporal_optimization/.
═══════════════════════════════════════════════════════════════════════
"""
import sys, os, time, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "reports", "temporal_optimization")
VIZ_DIR = os.path.join(BASE_DIR, "visualizations", "temporal_optimization")
os.makedirs(VIZ_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# COLOR PALETTE
# ═══════════════════════════════════════════════════════════════
NAVY = '#1B2A4A'
DARK_NAVY = '#0D1B2A'
STEEL = '#415A77'
LIGHT_STEEL = '#778DA9'
GOLD = '#D4A843'
WHITE = '#E0E1DD'
RED = '#C1292E'
GREEN = '#2A9D8F'
BG_COLOR = '#0D1B2A'

def style_ax(ax, title, xlabel='', ylabel=''):
    ax.set_facecolor(BG_COLOR)
    ax.set_title(title, color=WHITE, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, color=LIGHT_STEEL, fontsize=10)
    ax.set_ylabel(ylabel, color=LIGHT_STEEL, fontsize=10)
    ax.tick_params(colors=LIGHT_STEEL, which='both')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color(STEEL)
    ax.grid(True, alpha=0.15, color=LIGHT_STEEL)

def add_watermark(fig):
    fig.text(0.5, 0.5, 'CONFIDENTIAL', fontsize=60, color='white', alpha=0.03,
             ha='center', va='center', rotation=30, fontweight='bold')

def save_chart(fig, name):
    fig.patch.set_facecolor(BG_COLOR)
    add_watermark(fig)
    path = os.path.join(VIZ_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG_COLOR, edgecolor='none')
    plt.close(fig)

def get_color(wr):
    if wr < 50: return RED
    elif wr > 55: return GREEN
    else: return STEEL

# Helper custom colormap for heatmaps
from matplotlib.colors import LinearSegmentedColormap
cmap_red_green = LinearSegmentedColormap.from_list('rg', [RED, BG_COLOR, GREEN], N=256)

print("═"*70)
print("  TEMPORAL OPTIMIZATION VISUALIZATION SUITE — 7 Charts")
print("═"*70)

t0 = time.time()

# Ensure reports exist before plotting
if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
    print("Error: Temporal optimization reports not found. Run script 07 first.")
    sys.exit(1)

print("\n[1/7] Generating Hourly Performance Bar Chart...")
try:
    df_h = pd.read_csv(os.path.join(DATA_DIR, 'hourly_performance.csv'))
    df_h = df_h.sort_values('hour')
    fig, ax = plt.subplots(figsize=(14, 6))
    style_ax(ax, 'Win Rate by Hour of Day (UTC)', 'Hour', 'Win Rate (%)')
    
    colors = [get_color(wr) for wr in df_h['win_rate']]
    bars = ax.bar(df_h['hour'], df_h['win_rate'], color=colors, edgecolor=DARK_NAVY)
    ax.axhline(y=50, color='white', linestyle='--', alpha=0.3)
    ax.set_xticks(range(24))
    ax.set_xticklabels([f'{int(h):02d}' for h in df_h['hour']])
    ax.set_ylim(40, max(df_h['win_rate'])+2)
    
    for bar, wr in zip(bars, df_h['win_rate']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{wr:.1f}%', ha='center', color=WHITE, fontsize=8, rotation=90)
    
    save_chart(fig, '01_hourly_performance_heatmap')
except Exception as e:
    print(f"Error generating chart 1: {e}")

print("[2/7] Generating Daily Performance Bar Chart...")
try:
    df_d = pd.read_csv(os.path.join(DATA_DIR, 'daily_performance.csv'))
    df_d = df_d.sort_values('day')
    fig, ax = plt.subplots(figsize=(10, 6))
    style_ax(ax, 'Win Rate by Day of Week', '', 'Win Rate (%)')
    
    colors = [get_color(wr) for wr in df_d['win_rate']]
    bars = ax.bar(df_d['day_name'], df_d['win_rate'], color=colors, edgecolor=DARK_NAVY)
    ax.axhline(y=50, color='white', linestyle='--', alpha=0.3)
    ax.set_ylim(45, max(df_d['win_rate'])+2)
    
    for bar, wr in zip(bars, df_d['win_rate']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{wr:.1f}%', ha='center', color=WHITE, fontsize=10)
    
    save_chart(fig, '02_daily_performance_heatmap')
except Exception as e:
    print(f"Error generating chart 2: {e}")

print("[3/7] Generating Quarter Comparison Chart...")
try:
    df_q = pd.read_csv(os.path.join(DATA_DIR, 'quarter_performance.csv'))
    df_q = df_q.sort_values('quarter')
    fig, ax = plt.subplots(figsize=(8, 6))
    style_ax(ax, 'Win Rate by Quarter of Hour', '', 'Win Rate (%)')
    
    colors = [get_color(wr) for wr in df_q['win_rate']]
    bars = ax.bar(df_q['quarter'], df_q['win_rate'], color=colors, edgecolor=DARK_NAVY, width=0.6)
    ax.axhline(y=50, color='white', linestyle='--', alpha=0.3)
    ax.set_ylim(45, max(df_q['win_rate'])+2)
    
    for bar, wr in zip(bars, df_q['win_rate']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{wr:.2f}%', ha='center', color=WHITE, fontsize=11, fontweight='bold')
        
    save_chart(fig, '03_quarter_performance_comparison')
except Exception as e:
    print(f"Error generating chart 3: {e}")

print("[4/7] Generating Hour x Day Heatmap...")
try:
    df_hd = pd.read_csv(os.path.join(DATA_DIR, 'hour_day_matrix.csv'))
    # Filter out statistically insignificant combos for heatmap clarity
    df_hd = df_hd[df_hd['trades'] >= 20]
    
    pivot_hd = df_hd.pivot(index='day_name', columns='hour', values='win_rate')
    # Reorder to real week
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_hd = pivot_hd.reindex(days_order)
    
    fig, ax = plt.subplots(figsize=(20, 8))
    sns.heatmap(pivot_hd, cmap=cmap_red_green, annot=True, fmt=".1f", center=53, 
                vmin=40, vmax=65, cbar_kws={'label': 'Win Rate (%)'}, ax=ax, linewidths=0.5, linecolor=DARK_NAVY)
    style_ax(ax, 'Hour × Day Win Rate Map (Minimum 20 Trades)', 'Hour of Day (UTC)', '')
    
    # Customise y-ticks for readability
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    save_chart(fig, '04_hour_day_heatmap')
except Exception as e:
    print(f"Error generating chart 4: {e}")

print("[5/7] Generating Hour x Quarter Heatmap...")
try:
    df_hq = pd.read_csv(os.path.join(DATA_DIR, 'hour_quarter_matrix.csv'))
    df_hq = df_hq[df_hq['trades'] >= 20]
    pivot_hq = df_hq.pivot(index='quarter', columns='hour', values='win_rate')
    pivot_hq = pivot_hq.reindex(['Q1', 'Q2', 'Q3', 'Q4'])
    
    fig, ax = plt.subplots(figsize=(20, 5))
    sns.heatmap(pivot_hq, cmap=cmap_red_green, annot=True, fmt=".1f", center=53, 
                vmin=40, vmax=65, cbar_kws={'label': 'Win Rate (%)'}, ax=ax, linewidths=0.5, linecolor=DARK_NAVY)
    style_ax(ax, 'Hour × Quarter Win Rate Map', 'Hour of Day (UTC)', 'Quarter')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    save_chart(fig, '05_hour_quarter_heatmap')
except Exception as e:
    print(f"Error generating chart 5: {e}")

print("[6/7] Generating Day x Quarter Heatmap...")
try:
    df_dq = pd.read_csv(os.path.join(DATA_DIR, 'day_quarter_matrix.csv'))
    df_dq = df_dq[df_dq['trades'] >= 20]
    pivot_dq = df_dq.pivot(index='day_name', columns='quarter', values='win_rate')
    pivot_dq = pivot_dq.reindex(days_order)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot_dq, cmap=cmap_red_green, annot=True, fmt=".1f", center=53, 
                vmin=45, vmax=60, cbar_kws={'label': 'Win Rate (%)'}, ax=ax, linewidths=0.5, linecolor=DARK_NAVY)
    style_ax(ax, 'Day × Quarter Win Rate Map', 'Quarter of Hour', '')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    save_chart(fig, '06_day_quarter_heatmap')
except Exception as e:
    print(f"Error generating chart 6: {e}")

print("[7/7] Generating Best/Worst Summary Comparison...")
try:
    # We will compute pseudo aggregates from the hour_day matrix
    # Best 20% vs Worst 20% vs Baseline
    df_hd = pd.read_csv(os.path.join(DATA_DIR, 'hour_day_matrix.csv'))
    df_hd = df_hd[df_hd['trades'] >= 50] # High confidence only
    
    total_slots = len(df_hd)
    top_n = max(1, int(total_slots * 0.2))
    bottom_n = max(1, int(total_slots * 0.2))
    
    df_hd_sorted = df_hd.sort_values('win_rate', ascending=False)
    
    top_slots = df_hd_sorted.head(top_n)
    bottom_slots = df_hd_sorted.tail(bottom_n)
    keep_slots = df_hd_sorted.head(total_slots - bottom_n) # Remove worst 20%
    
    def aggr(sub_df):
        tr = sub_df['trades'].sum()
        pnl = sub_df['pnl'].sum()
        w = (sub_df['win_rate']/100 * sub_df['trades']).sum()
        return tr, (w/tr)*100 if tr>0 else 0, pnl
        
    base_tr, base_wr, base_pnl = aggr(df_hd)
    top_tr, top_wr, top_pnl = aggr(top_slots)
    rem_tr, rem_wr, rem_pnl = aggr(keep_slots)
    
    categories = ['Baseline', 'Trade Top 20% Slots', 'Remove Worst 20% Slots']
    wrs = [base_wr, top_wr, rem_wr]
    pnls = [base_pnl, top_pnl, rem_pnl]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Estimated Theoretical Impact of Temporal Filtering', color=WHITE, fontsize=16, fontweight='bold', y=1.02)
    fig.patch.set_facecolor(BG_COLOR)
    
    # Left: Win Rate
    style_ax(ax1, 'Composite Win Rate Analysis', '', 'Win Rate (%)')
    colors = [STEEL, GOLD, GREEN]
    bars1 = ax1.bar(categories, wrs, color=colors, edgecolor=DARK_NAVY, width=0.5)
    ax1.set_ylim(min(wrs)-5, max(wrs)+5)
    ax1.axhline(y=50, color='red', linestyle='--', alpha=0.3)
    ax1.axhline(y=base_wr, color='white', linestyle=':', alpha=0.5)
    for bar, wr in zip(bars1, wrs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{wr:.2f}%', ha='center', color=WHITE, fontsize=12, fontweight='bold')
                 
    # Right: P&L
    style_ax(ax2, 'Total Expected P&L Comparison', '', 'Total P&L ($)')
    bars2 = ax2.bar(categories, pnls, color=colors, edgecolor=DARK_NAVY, width=0.5)
    ax2.axhline(y=base_pnl, color='white', linestyle=':', alpha=0.5)
    for bar, pnl in zip(bars2, pnls):
        ax2.text(bar.get_x() + bar.get_width()/2, pnl + max(pnls)*0.02 if pnl>0 else pnl*1.05,
                 f'${pnl:,.0f}', ha='center', color=WHITE, fontsize=12, fontweight='bold')
                 
    save_chart(fig, '07_best_worst_comparison')

except Exception as e:
    print(f"Error generating chart 7: {e}")

elapsed = time.time() - t0
print(f"\n{'═'*70}")
print(f"  COMPLETE: 7 charts generated")
print(f"  Output: {VIZ_DIR}")
print(f"  Duration: {elapsed:.1f}s")
print(f"{'═'*70}")
