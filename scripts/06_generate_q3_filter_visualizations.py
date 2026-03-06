"""
═══════════════════════════════════════════════════════════════════════
06_generate_q3_filter_visualizations.py
Q3 Filter Visualization Suite — 6 Professional Charts
Strategy Code: ALPHA-BTC-15M-v2.0
═══════════════════════════════════════════════════════════════════════
Generates charts comparing baseline vs Q3-filtered strategy.
All outputs go to visualizations/q3_filter/.
═══════════════════════════════════════════════════════════════════════
"""
import sys, os, time, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
warnings.filterwarnings('ignore')

try:
    import seaborn as sns
    sns.set_theme(style='darkgrid')
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIZ_DIR = os.path.join(BASE_DIR, "visualizations", "q3_filter")
os.makedirs(VIZ_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# COLOR PALETTE (same institutional Navy/Gray/Gold as 04_)
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
GRID_COLOR = '#1B2A4A'
CYAN = '#4CC9F0'

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
    return path

# ═══════════════════════════════════════════════════════════════
# DATA LOADING & INDICATORS (reused from existing scripts)
# ═══════════════════════════════════════════════════════════════
def load_data(filepath):
    df = pd.read_csv(filepath)
    df.columns = [col.lower() for col in df.columns]
    if 'open_time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['open_time'])
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        df['timestamp'] = pd.to_datetime(df.iloc[:, 0])
    df.set_index('timestamp', inplace=True)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def calculate_rsi_wilder(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    return 100 - (100 / (1 + gain / loss))

def compute_indicators(df):
    df['rsi'] = calculate_rsi_wilder(df['close'], 14)
    tr = pd.concat([df['high']-df['low'], (df['high']-df['close'].shift(1)).abs(),
                     (df['low']-df['close'].shift(1)).abs()], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean()
    up_m, dn_m = df['high'].diff(), -df['low'].diff()
    plus_dm = np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0)
    minus_dm = np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0)
    atr_s = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100*(pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean()/atr_s)
    minus_di = 100*(pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean()/atr_s)
    dx = (abs(plus_di-minus_di)/(plus_di+minus_di).replace(0,np.nan))*100
    df['adx_14'] = dx.ewm(alpha=1/14, adjust=False).mean()
    atr_14_arr = atr_14.values
    n = len(df)
    atr_pct = np.zeros(n)
    for i in range(28, n):
        lb = min(96, i)
        rec = atr_14_arr[i-lb:i+1]
        if pd.isna(rec).all(): continue
        valid = rec[~np.isnan(rec)]
        if len(valid)<5: continue
        atr_pct[i] = (valid<atr_14_arr[i]).sum()/len(valid)*100
    df['atr_pct'] = atr_pct
    df['regime_vol'] = np.where(df['atr_pct']>80,'High',np.where(df['atr_pct']>40,'Medium','Low'))
    df['regime_trend'] = np.where(df['adx_14']>25,'Strong',np.where(df['adx_14']>15,'Medium','Weak'))
    df = df.dropna(subset=['rsi','adx_14']).copy()
    return df

def is_q3(timestamp):
    minute = pd.Timestamp(timestamp).minute
    return 30 <= minute <= 44

def generate_trades(df, apply_q3_filter=False):
    rsi_arr=df['rsi'].values; adx_arr=df['adx_14'].values; atr_pct_arr=df['atr_pct'].values
    c_arr=df['close'].values; ts_arr=df.index.values
    vol_arr=df['regime_vol'].values; trend_arr=df['regime_trend'].values
    trades=[]
    for i in range(1, len(df)-1):
        rsi=rsi_arr[i]
        buy_yes=rsi<43; buy_no=rsi>57
        if not(buy_yes or buy_no): continue
        if adx_arr[i]>25 and atr_pct_arr[i]>80: continue
        if apply_q3_filter and is_q3(ts_arr[i]):
            continue
        signal='YES' if buy_yes else 'NO'
        settle_c=c_arr[i+1]
        won=(signal=='YES' and settle_c>c_arr[i]) or (signal=='NO' and settle_c<c_arr[i])
        pnl=0.98 if won else -1.02
        trades.append({'timestamp':ts_arr[i],'signal':signal,'win':won,'pnl':pnl,
                       'regime_vol':vol_arr[i],'regime_trend':trend_arr[i]})
    tdf=pd.DataFrame(trades)
    tdf['timestamp']=pd.to_datetime(tdf['timestamp'])
    tdf.set_index('timestamp',inplace=True)
    return tdf

# ═══════════════════════════════════════════════════════════════
print("═"*70)
print("  Q3 FILTER VISUALIZATION SUITE — 6 Professional Charts")
print("═"*70)

t0 = time.time()
print("\n[1/3] Loading data and generating trades...")
df = load_data(os.path.join(BASE_DIR, 'data', 'BTCUSDT_15m_3_years.csv'))
df = compute_indicators(df)

baseline_tdf = generate_trades(df, apply_q3_filter=False)
q3_tdf = generate_trades(df, apply_q3_filter=True)

INITIAL_CAPITAL = 100.0

# Baseline series
b_daily_pnl = baseline_tdf['pnl'].resample('1D').sum().fillna(0)
b_daily_cap = INITIAL_CAPITAL + b_daily_pnl.cumsum()
b_win_rate = baseline_tdf['win'].mean()

# Q3 series
q_daily_pnl = q3_tdf['pnl'].resample('1D').sum().fillna(0)
q_daily_cap = INITIAL_CAPITAL + q_daily_pnl.cumsum()
q_win_rate = q3_tdf['win'].mean()

# Drawdowns
q_roll_max = q_daily_cap.cummax()
q_dd_pct = ((q_daily_cap - q_roll_max) / q_roll_max * 100)
b_roll_max = b_daily_cap.cummax()
b_dd_pct = ((b_daily_cap - b_roll_max) / b_roll_max * 100)

print(f"  Baseline: {len(baseline_tdf):,} trades | Q3-Filtered: {len(q3_tdf):,} trades")
chart_count = 0

# ═══════════════════════════════════════════════════════════════
# CHART 1: Q3 Equity Curve (Baseline vs Q3 overlay)
# ═══════════════════════════════════════════════════════════════
print("\n[2/3] Generating charts...")

fig, ax = plt.subplots(figsize=(16, 7))
style_ax(ax, 'Equity Curve — Baseline vs Q3-Filtered', 'Date', 'Balance ($)')
ax.plot(b_daily_cap.index, b_daily_cap.values, color=LIGHT_STEEL, linewidth=1.2, alpha=0.7, label='Baseline')
ax.plot(q_daily_cap.index, q_daily_cap.values, color=GOLD, linewidth=1.5, label='Q3-Filtered')
ax.axhline(y=INITIAL_CAPITAL, color=RED, linestyle='--', alpha=0.3, label='Starting Capital')
ax.legend(facecolor=DARK_NAVY, edgecolor=STEEL, labelcolor=WHITE, fontsize=11)
# Add summary text box
textstr = f'Baseline: ${b_daily_cap.iloc[-1]:,.0f} | Q3: ${q_daily_cap.iloc[-1]:,.0f}'
props = dict(boxstyle='round', facecolor=NAVY, alpha=0.8, edgecolor=STEEL)
ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top',
        color=WHITE, bbox=props)
save_chart(fig, '01_q3_equity_curve')
chart_count += 1

# ═══════════════════════════════════════════════════════════════
# CHART 2: Q3 Drawdown
# ═══════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

style_ax(ax1, 'Q3-Filtered Equity with Drawdown Periods', '', 'Balance ($)')
ax1.plot(q_daily_cap.index, q_daily_cap.values, color=GOLD, linewidth=1.2)
ax1.fill_between(q_daily_cap.index, q_daily_cap.values, q_roll_max.values, color=RED, alpha=0.3, label='Drawdown')
ax1.plot(q_roll_max.index, q_roll_max.values, color=GREEN, linewidth=0.8, alpha=0.5, label='Peak')
ax1.legend(facecolor=DARK_NAVY, edgecolor=STEEL, labelcolor=WHITE)

style_ax(ax2, '', 'Date', 'Drawdown (%)')
ax2.fill_between(q_dd_pct.index, q_dd_pct.values, 0, color=RED, alpha=0.5)
ax2.axhline(y=0, color=LIGHT_STEEL, linewidth=0.5)
max_dd_val = q_dd_pct.min()
max_dd_date = q_dd_pct.idxmin()
ax2.annotate(f'Max DD: {max_dd_val:.1f}%', xy=(max_dd_date, max_dd_val),
             xytext=(max_dd_date + pd.Timedelta(days=60), max_dd_val + 2),
             arrowprops=dict(arrowstyle='->', color=GOLD), color=GOLD, fontsize=11, fontweight='bold')
save_chart(fig, '02_q3_drawdown')
chart_count += 1

# ═══════════════════════════════════════════════════════════════
# CHART 3: Q3 Monthly Heatmap
# ═══════════════════════════════════════════════════════════════
monthly_pnl = q3_tdf['pnl'].resample('ME').sum()
pivot_data = {}
for dt, val in monthly_pnl.items():
    yr = dt.year
    mo = dt.month
    if yr not in pivot_data: pivot_data[yr] = {}
    pivot_data[yr][mo] = val
years = sorted(pivot_data.keys())
months = range(1, 13)
heat_arr = np.full((len(years), 12), np.nan)
for i, yr in enumerate(years):
    for mo in months:
        if mo in pivot_data[yr]:
            heat_arr[i, mo-1] = pivot_data[yr][mo]

fig, ax = plt.subplots(figsize=(16, 6))
style_ax(ax, 'Q3-Filtered Monthly Returns Heatmap ($)', '', '')
cmap = plt.cm.RdYlGn
im = ax.imshow(heat_arr, cmap=cmap, aspect='auto', vmin=-150, vmax=200)
ax.set_xticks(range(12))
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], color=LIGHT_STEEL)
ax.set_yticks(range(len(years)))
ax.set_yticklabels(years, color=LIGHT_STEEL)
for i in range(len(years)):
    for j in range(12):
        val = heat_arr[i, j]
        if not np.isnan(val):
            ax.text(j, i, f'${val:.0f}', ha='center', va='center',
                    color='black' if abs(val) < 100 else 'white', fontsize=8, fontweight='bold')
plt.colorbar(im, ax=ax, label='P&L ($)')
save_chart(fig, '03_q3_monthly_heatmap')
chart_count += 1

# ═══════════════════════════════════════════════════════════════
# CHART 4: Hourly Performance (showing Q3 exclusion zone)
# ═══════════════════════════════════════════════════════════════
# Baseline hourly
b_hourly = baseline_tdf.groupby(baseline_tdf.index.hour).agg(
    trades=('win', 'count'), wr=('win', 'mean'))
q_hourly = q3_tdf.groupby(q3_tdf.index.hour).agg(
    trades=('win', 'count'), wr=('win', 'mean'))

fig, ax1 = plt.subplots(figsize=(16, 7))
style_ax(ax1, 'Hourly Performance — Baseline vs Q3-Filtered (UTC)', 'Hour', 'Number of Trades')

# Bar chart: baseline trades (muted) + Q3 trades (bright)
x = np.arange(24)
width = 0.35
bars1 = ax1.bar(x - width/2, b_hourly.reindex(range(24), fill_value=0)['trades'],
                width, color=STEEL, alpha=0.5, label='Baseline Trades', edgecolor=DARK_NAVY)
bars2 = ax1.bar(x + width/2, q_hourly.reindex(range(24), fill_value=0)['trades'],
                width, color=GOLD, alpha=0.8, label='Q3-Filtered Trades', edgecolor=DARK_NAVY)

ax2 = ax1.twinx()
# Win rate lines
b_wr_vals = b_hourly.reindex(range(24))['wr'] * 100
q_wr_vals = q_hourly.reindex(range(24))['wr'] * 100
ax2.plot(x, b_wr_vals, color=LIGHT_STEEL, linewidth=1.5, marker='s', markersize=4, alpha=0.7, label='Baseline WR')
ax2.plot(x, q_wr_vals, color=CYAN, linewidth=2, marker='o', markersize=5, label='Q3 WR')
ax2.axhline(y=q_win_rate * 100, color=GREEN, linestyle='--', alpha=0.5, label=f'Q3 Avg: {q_win_rate*100:.1f}%')
ax2.set_ylabel('Win Rate (%)', color=LIGHT_STEEL)
ax2.tick_params(colors=LIGHT_STEEL)

# Note about Q3 in subtitle
ax1.text(0.5, -0.12, 'Note: Q3 filter removes candles at minute :30 (30-44 block) from every hour',
         transform=ax1.transAxes, ha='center', color=LIGHT_STEEL, fontsize=9, style='italic')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, facecolor=DARK_NAVY, edgecolor=STEEL,
           labelcolor=WHITE, fontsize=9, loc='upper left')
ax1.set_xticks(x)
ax1.set_xticklabels([f'{h:02d}' for h in range(24)])
save_chart(fig, '04_q3_hourly_performance')
chart_count += 1

# ═══════════════════════════════════════════════════════════════
# CHART 5: Baseline vs Q3 — Key Metrics Comparison
# ═══════════════════════════════════════════════════════════════
b_daily = b_daily_pnl
q_daily = q_daily_pnl
b_sharpe = (b_daily.mean() / b_daily.std()) * np.sqrt(365) if b_daily.std() > 0 else 0
q_sharpe = (q_daily.mean() / q_daily.std()) * np.sqrt(365) if q_daily.std() > 0 else 0

b_neg_std = b_daily[b_daily < 0].std()
q_neg_std = q_daily[q_daily < 0].std()
b_sortino = (b_daily.mean() / b_neg_std) * np.sqrt(365) if b_neg_std > 0 else 0
q_sortino = (q_daily.mean() / q_neg_std) * np.sqrt(365) if q_neg_std > 0 else 0

b_gross_prof = baseline_tdf[baseline_tdf['pnl'] > 0]['pnl'].sum()
b_gross_loss = abs(baseline_tdf[baseline_tdf['pnl'] < 0]['pnl'].sum())
q_gross_prof = q3_tdf[q3_tdf['pnl'] > 0]['pnl'].sum()
q_gross_loss = abs(q3_tdf[q3_tdf['pnl'] < 0]['pnl'].sum())
b_pf = b_gross_prof / b_gross_loss if b_gross_loss > 0 else 0
q_pf = q_gross_prof / q_gross_loss if q_gross_loss > 0 else 0

metrics_names = ['Win Rate (%)', 'Sharpe Ratio', 'Sortino Ratio', 'Profit Factor',
                 'Max DD (%)', 'Total Trades']
baseline_vals = [b_win_rate * 100, b_sharpe, b_sortino, b_pf,
                 abs(b_dd_pct.min()), len(baseline_tdf)]
q3_vals = [q_win_rate * 100, q_sharpe, q_sortino, q_pf,
           abs(q_dd_pct.min()), len(q3_tdf)]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Baseline vs Q3-Filtered — Key Metrics Comparison', color=WHITE, fontsize=16, fontweight='bold', y=0.98)

for idx, (name, bval, qval) in enumerate(zip(metrics_names, baseline_vals, q3_vals)):
    ax = axes[idx // 3][idx % 3]
    style_ax(ax, name, '', '')
    bars = ax.bar(['Baseline', 'Q3-Filter'], [bval, qval],
                  color=[STEEL, GOLD], alpha=0.8, edgecolor=DARK_NAVY, width=0.5)
    for bar, val in zip(bars, [bval, qval]):
        if name == 'Total Trades':
            label = f'{val:,.0f}'
        elif 'Ratio' in name or 'Factor' in name:
            label = f'{val:.2f}'
        else:
            label = f'{val:.1f}'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(bval, qval)*0.02,
                label, ha='center', color=WHITE, fontweight='bold', fontsize=11)

    # Delta annotation
    if name == 'Max DD (%)' or name == 'Total Trades':
        diff = qval - bval
        better = diff < 0 if name == 'Max DD (%)' else True
    else:
        diff = qval - bval
        better = diff > 0
    arrow_color = GREEN if better else RED
    ax.text(0.5, 0.02, f'Δ = {diff:+.2f}', transform=ax.transAxes, ha='center',
            color=arrow_color, fontsize=10, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])
save_chart(fig, '05_q3_vs_baseline_comparison')
chart_count += 1

# ═══════════════════════════════════════════════════════════════
# CHART 6: Trade P&L Distribution Comparison
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 7))
style_ax(ax, 'Trade P&L Distribution — Baseline vs Q3-Filtered', 'P&L per Trade ($)', 'Frequency')

# Both distributions are binary (0.98 or -1.02), so show as grouped bar
b_outcomes = baseline_tdf['pnl'].value_counts().sort_index()
q_outcomes = q3_tdf['pnl'].value_counts().sort_index()

all_values = sorted(set(b_outcomes.index) | set(q_outcomes.index))
x_pos = np.arange(len(all_values))
width = 0.35

b_counts = [b_outcomes.get(v, 0) for v in all_values]
q_counts = [q_outcomes.get(v, 0) for v in all_values]

bars1 = ax.bar(x_pos - width/2, b_counts, width, color=STEEL, alpha=0.7,
               label=f'Baseline ({len(baseline_tdf):,})', edgecolor=DARK_NAVY)
bars2 = ax.bar(x_pos + width/2, q_counts, width, color=GOLD, alpha=0.8,
               label=f'Q3-Filtered ({len(q3_tdf):,})', edgecolor=DARK_NAVY)

ax.set_xticks(x_pos)
ax.set_xticklabels([f'${v:.2f}' for v in all_values], color=LIGHT_STEEL)

# Add counts on bars
for bar, count in zip(bars1, b_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(max(b_counts), max(q_counts))*0.01,
            f'{count:,}', ha='center', color=LIGHT_STEEL, fontsize=9)
for bar, count in zip(bars2, q_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(max(b_counts), max(q_counts))*0.01,
            f'{count:,}', ha='center', color=GOLD, fontsize=9)

# Add win rate annotation
props = dict(boxstyle='round', facecolor=NAVY, alpha=0.8, edgecolor=STEEL)
textstr = f'Baseline WR: {b_win_rate*100:.2f}%\nQ3-Filtered WR: {q_win_rate*100:.2f}%\nΔ WR: {(q_win_rate-b_win_rate)*100:+.2f}%'
ax.text(0.98, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top',
        ha='right', color=WHITE, bbox=props)

ax.legend(facecolor=DARK_NAVY, edgecolor=STEEL, labelcolor=WHITE, fontsize=11)
save_chart(fig, '06_q3_trade_distribution')
chart_count += 1

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
elapsed = time.time() - t0

summary = f"""Q3 FILTER VISUALIZATION SUITE — GENERATION SUMMARY
{'='*50}
Generated: {pd.Timestamp.now().strftime('%B %d, %Y %H:%M')}
Total Charts: {chart_count}
Output Directory: {VIZ_DIR}
Duration: {elapsed:.1f} seconds

CHART INDEX:
01. Equity Curve — Baseline vs Q3-Filtered
02. Q3-Filtered Drawdown (equity + underwater)
03. Q3-Filtered Monthly Returns Heatmap
04. Hourly Performance — Baseline vs Q3
05. Key Metrics Comparison (6 panels)
06. Trade P&L Distribution Comparison

Classification: CONFIDENTIAL
"""

with open(os.path.join(VIZ_DIR, "q3_visualization_summary.txt"), "w", encoding='utf-8') as f:
    f.write(summary)

print(f"\n{'═'*70}")
print(f"  COMPLETE: {chart_count} charts generated")
print(f"  Output: {VIZ_DIR}")
print(f"  Duration: {elapsed:.1f}s")
print(f"{'═'*70}")
