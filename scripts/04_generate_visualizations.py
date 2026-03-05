"""
═══════════════════════════════════════════════════════════════════════
04_generate_visualizations.py
Professional Visualization Suite — 26 Charts
Strategy Code: ALPHA-BTC-15M-v2.0
═══════════════════════════════════════════════════════════════════════
"""
import sys, os, time, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
warnings.filterwarnings('ignore')

try:
    import seaborn as sns
    sns.set_theme(style='darkgrid')
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIZ_DIR = os.path.join(BASE_DIR, "visualizations")
os.makedirs(VIZ_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# COLOR PALETTE (Navy / Gray / Gold — Institutional)
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
# DATA LOADING (same as other scripts)
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
    tr = pd.concat([df['high']-df['low'], (df['high']-df['close'].shift(1)).abs(), (df['low']-df['close'].shift(1)).abs()], axis=1).max(axis=1)
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

def generate_trades(df):
    rsi_arr=df['rsi'].values; adx_arr=df['adx_14'].values; atr_pct_arr=df['atr_pct'].values
    c_arr=df['close'].values; ts_arr=df.index.values
    vol_arr=df['regime_vol'].values; trend_arr=df['regime_trend'].values
    trades=[]
    for i in range(1, len(df)-1):
        rsi=rsi_arr[i]
        buy_yes=rsi<43; buy_no=rsi>57
        if not(buy_yes or buy_no): continue
        if adx_arr[i]>25 and atr_pct_arr[i]>80: continue
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
print("  VISUALIZATION SUITE — 26 Professional Charts")
print("═"*70)

t0 = time.time()
print("\n[1/5] Loading data and generating trades...")
df = load_data(os.path.join(BASE_DIR, 'data', 'BTCUSDT_15m_3_years.csv'))
df = compute_indicators(df)
tdf = generate_trades(df)
INITIAL_CAPITAL = 100.0
total_trades = len(tdf)
win_rate = tdf['win'].mean()
daily_pnl = tdf['pnl'].resample('1D').sum().fillna(0)
daily_cap = INITIAL_CAPITAL + daily_pnl.cumsum()
running_max = daily_cap.cummax()
dd_pct = ((daily_cap - running_max) / running_max * 100)

print(f"  {total_trades:,} trades | WR: {win_rate*100:.2f}%")
chart_count = 0

# ═══════════════════════════════════════════════════════════════
# EQUITY & PERFORMANCE CHARTS (1-5)
# ═══════════════════════════════════════════════════════════════
print("\n[2/5] Generating Equity & Performance charts...")

# Chart 1: Cumulative P&L
fig, ax = plt.subplots(figsize=(16,7))
style_ax(ax, 'Cumulative P&L — Equity Curve', 'Date', 'Cumulative P&L ($)')
ax.plot(daily_cap.index, daily_cap.values, color=GOLD, linewidth=1.5, label='Equity')
ax.fill_between(daily_cap.index, INITIAL_CAPITAL, daily_cap.values,
                where=daily_cap.values>=INITIAL_CAPITAL, color=GREEN, alpha=0.15)
ax.fill_between(daily_cap.index, INITIAL_CAPITAL, daily_cap.values,
                where=daily_cap.values<INITIAL_CAPITAL, color=RED, alpha=0.15)
ax.axhline(y=INITIAL_CAPITAL, color=LIGHT_STEEL, linestyle='--', alpha=0.5, label='Baseline ($100)')
ax.legend(facecolor=DARK_NAVY, edgecolor=STEEL, labelcolor=WHITE)
save_chart(fig, '01_cumulative_pnl')
chart_count += 1

# Chart 2: Daily P&L Distribution
fig, ax = plt.subplots(figsize=(14,7))
style_ax(ax, 'Daily P&L Distribution', 'Daily P&L ($)', 'Frequency')
ax.hist(daily_pnl.values, bins=60, color=STEEL, edgecolor=NAVY, alpha=0.8)
ax.axvline(x=daily_pnl.mean(), color=GOLD, linestyle='--', linewidth=2, label=f'Mean: ${daily_pnl.mean():.2f}')
ax.axvline(x=0, color=RED, linestyle='-', linewidth=1, alpha=0.5)
# Normal overlay
x_range = np.linspace(daily_pnl.min(), daily_pnl.max(), 200)
from scipy.stats import norm
normal_pdf = norm.pdf(x_range, daily_pnl.mean(), daily_pnl.std())
ax2 = ax.twinx()
ax2.plot(x_range, normal_pdf, color=GOLD, alpha=0.6, linewidth=2, label='Normal Fit')
ax2.set_yticks([])
ax.legend(facecolor=DARK_NAVY, edgecolor=STEEL, labelcolor=WHITE)
save_chart(fig, '02_daily_pnl_distribution')
chart_count += 1

# Chart 3: Equity with Underwater Periods
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16,10), gridspec_kw={'height_ratios':[3,1]}, sharex=True)
style_ax(ax1, 'Equity Curve with Drawdown Periods', '', 'Balance ($)')
ax1.plot(daily_cap.index, daily_cap.values, color=GOLD, linewidth=1.2)
ax1.fill_between(daily_cap.index, daily_cap.values, running_max.values, color=RED, alpha=0.3, label='Drawdown')
ax1.plot(running_max.index, running_max.values, color=GREEN, linewidth=0.8, alpha=0.5, label='Peak')
ax1.legend(facecolor=DARK_NAVY, edgecolor=STEEL, labelcolor=WHITE)
style_ax(ax2, '', 'Date', 'Drawdown (%)')
ax2.fill_between(dd_pct.index, dd_pct.values, 0, color=RED, alpha=0.5)
ax2.axhline(y=0, color=LIGHT_STEEL, linewidth=0.5)
save_chart(fig, '03_equity_underwater')
chart_count += 1

# Chart 4: Rolling 30-Day Sharpe
rolling_30d_mean = daily_pnl.rolling(30).mean()
rolling_30d_std = daily_pnl.rolling(30).std()
rolling_sharpe = (rolling_30d_mean / rolling_30d_std * np.sqrt(365)).dropna()
fig, ax = plt.subplots(figsize=(16,7))
style_ax(ax, 'Rolling 30-Day Sharpe Ratio (Annualized)', 'Date', 'Sharpe Ratio')
ax.plot(rolling_sharpe.index, rolling_sharpe.values, color=GOLD, linewidth=1)
overall_sharpe = (daily_pnl.mean()/daily_pnl.std())*np.sqrt(365)
ax.axhline(y=overall_sharpe, color=GREEN, linestyle='--', alpha=0.7, label=f'Overall: {overall_sharpe:.1f}')
ax.axhline(y=0, color=RED, linestyle='-', alpha=0.3)
mean_roll = rolling_sharpe.mean()
std_roll = rolling_sharpe.std()
ax.fill_between(rolling_sharpe.index, mean_roll-std_roll, mean_roll+std_roll, color=STEEL, alpha=0.15, label='±1σ Band')
ax.legend(facecolor=DARK_NAVY, edgecolor=STEEL, labelcolor=WHITE)
save_chart(fig, '04_rolling_sharpe')
chart_count += 1

# Chart 5: Monthly Returns Heatmap
monthly_pnl = tdf['pnl'].resample('ME').sum()
pivot_data = {}
for dt, val in monthly_pnl.items():
    yr = dt.year
    mo = dt.month
    if yr not in pivot_data: pivot_data[yr] = {}
    pivot_data[yr][mo] = val
years = sorted(pivot_data.keys())
months = range(1,13)
heat_arr = np.full((len(years), 12), np.nan)
for i, yr in enumerate(years):
    for mo in months:
        if mo in pivot_data[yr]:
            heat_arr[i, mo-1] = pivot_data[yr][mo]
fig, ax = plt.subplots(figsize=(16,6))
style_ax(ax, 'Monthly Returns Heatmap ($)', '', '')
cmap = plt.cm.RdYlGn
im = ax.imshow(heat_arr, cmap=cmap, aspect='auto', vmin=-150, vmax=200)
ax.set_xticks(range(12))
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], color=LIGHT_STEEL)
ax.set_yticks(range(len(years)))
ax.set_yticklabels(years, color=LIGHT_STEEL)
for i in range(len(years)):
    for j in range(12):
        val = heat_arr[i,j]
        if not np.isnan(val):
            ax.text(j, i, f'${val:.0f}', ha='center', va='center', color='black' if abs(val)<100 else 'white', fontsize=8, fontweight='bold')
plt.colorbar(im, ax=ax, label='P&L ($)')
save_chart(fig, '05_monthly_heatmap')
chart_count += 1

# ═══════════════════════════════════════════════════════════════
# DRAWDOWN CHARTS (6-9)
# ═══════════════════════════════════════════════════════════════
print("[3/5] Generating Drawdown & Distribution charts...")

# Chart 6: Drawdown Curve
fig, ax = plt.subplots(figsize=(16,6))
style_ax(ax, 'Drawdown from Peak (%)', 'Date', 'Drawdown (%)')
ax.fill_between(dd_pct.index, dd_pct.values, 0, color=RED, alpha=0.6)
ax.axhline(y=0, color=LIGHT_STEEL, linewidth=0.5)
ax.annotate(f'Max DD: {dd_pct.min():.1f}%', xy=(dd_pct.idxmin(), dd_pct.min()),
            xytext=(dd_pct.idxmin()+pd.Timedelta(days=60), dd_pct.min()+2),
            arrowprops=dict(arrowstyle='->', color=GOLD), color=GOLD, fontsize=11, fontweight='bold')
save_chart(fig, '06_drawdown_curve')
chart_count += 1

# Chart 7: Win/Loss Streak Distribution
win_streaks = []
loss_streaks = []
current_streak = 0
current_type = None
for w in tdf['win'].values:
    if current_type is None:
        current_type = w; current_streak = 1
    elif w == current_type:
        current_streak += 1
    else:
        if current_type: win_streaks.append(current_streak)
        else: loss_streaks.append(current_streak)
        current_type = w; current_streak = 1
if current_type is not None:
    if current_type: win_streaks.append(current_streak)
    else: loss_streaks.append(current_streak)

fig, ax = plt.subplots(figsize=(14,7))
style_ax(ax, 'Win/Loss Streak Distribution', 'Streak Length', 'Frequency')
max_s = max(max(win_streaks), max(loss_streaks))
bins = range(1, min(max_s+2, 25))
ax.hist(win_streaks, bins=bins, color=GREEN, alpha=0.7, label='Win Streaks', edgecolor=DARK_NAVY)
ax.hist(loss_streaks, bins=bins, color=RED, alpha=0.7, label='Loss Streaks', edgecolor=DARK_NAVY)
ax.set_yscale('log')
ax.legend(facecolor=DARK_NAVY, edgecolor=STEEL, labelcolor=WHITE)
save_chart(fig, '07_streak_distribution')
chart_count += 1

# Chart 8: Win Rate by Hour
hourly = tdf.groupby(tdf.index.hour).agg(trades=('win','count'), wr=('win','mean'))
fig, ax1 = plt.subplots(figsize=(14,7))
style_ax(ax1, 'Performance by Hour (UTC)', 'Hour', 'Number of Trades')
bars = ax1.bar(hourly.index, hourly['trades'], color=STEEL, alpha=0.7, label='Trades')
ax2 = ax1.twinx()
ax2.plot(hourly.index, hourly['wr']*100, color=GOLD, linewidth=2.5, marker='o', markersize=5, label='Win Rate')
ax2.axhline(y=win_rate*100, color=GREEN, linestyle='--', alpha=0.5, label=f'Avg WR: {win_rate*100:.1f}%')
ax2.set_ylabel('Win Rate (%)', color=LIGHT_STEEL)
ax2.tick_params(colors=LIGHT_STEEL)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, facecolor=DARK_NAVY, edgecolor=STEEL, labelcolor=WHITE)
save_chart(fig, '08_hourly_performance')
chart_count += 1

# Chart 9: Win Rate by Day of Week
days_map = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
dow = tdf.groupby(tdf.index.dayofweek).agg(trades=('win','count'), wr=('win','mean'))
fig, ax1 = plt.subplots(figsize=(12,7))
style_ax(ax1, 'Performance by Day of Week', 'Day', 'Number of Trades')
ax1.bar([days_map[i] for i in dow.index], dow['trades'], color=STEEL, alpha=0.7)
ax2 = ax1.twinx()
ax2.plot([days_map[i] for i in dow.index], dow['wr']*100, color=GOLD, linewidth=2.5, marker='o', markersize=8)
ax2.axhline(y=win_rate*100, color=GREEN, linestyle='--', alpha=0.5)
ax2.set_ylabel('Win Rate (%)', color=LIGHT_STEEL)
ax2.tick_params(colors=LIGHT_STEEL)
save_chart(fig, '09_day_of_week')
chart_count += 1

# Chart 10: Profit Factor by Month
monthly_wins = tdf[tdf['pnl']>0].resample('ME')['pnl'].sum()
monthly_losses = tdf[tdf['pnl']<0].resample('ME')['pnl'].sum().abs()
monthly_pf = (monthly_wins / monthly_losses).fillna(0).replace([np.inf], 5)
fig, ax = plt.subplots(figsize=(16,6))
style_ax(ax, 'Monthly Profit Factor', 'Date', 'Profit Factor')
colors_pf = [GREEN if v > 1 else RED for v in monthly_pf.values]
ax.bar(monthly_pf.index, monthly_pf.values, width=20, color=colors_pf, alpha=0.8)
ax.axhline(y=1.0, color=GOLD, linestyle='--', linewidth=2, label='Breakeven (PF=1.0)')
ax.legend(facecolor=DARK_NAVY, edgecolor=STEEL, labelcolor=WHITE)
save_chart(fig, '10_monthly_profit_factor')
chart_count += 1

# ═══════════════════════════════════════════════════════════════
# REGIME & RISK CHARTS (11-18)
# ═══════════════════════════════════════════════════════════════
print("[4/5] Generating Regime & Risk charts...")

# Chart 11: Regime Performance (Vol)  
vol_regimes = tdf.groupby('regime_vol').agg(trades=('win','count'), wr=('win','mean'), pnl=('pnl','sum'))
fig, ax = plt.subplots(figsize=(10,7))
style_ax(ax, 'Win Rate by Volatility Regime', 'Regime', 'Win Rate (%)')
colors_r = [GREEN, GOLD, RED]
for i, (regime, row) in enumerate(vol_regimes.iterrows()):
    ax.bar(regime, row['wr']*100, color=colors_r[i%3], alpha=0.8, edgecolor=DARK_NAVY)
    ax.text(i, row['wr']*100+0.3, f"{row['wr']*100:.1f}%\n({int(row['trades']):,})", ha='center', color=WHITE, fontsize=10)
ax.axhline(y=win_rate*100, color=LIGHT_STEEL, linestyle='--', alpha=0.7)
save_chart(fig, '11_vol_regime')
chart_count += 1

# Chart 12: Regime Performance (Trend)
trend_regimes = tdf.groupby('regime_trend').agg(trades=('win','count'), wr=('win','mean'), pnl=('pnl','sum'))
fig, ax = plt.subplots(figsize=(10,7))
style_ax(ax, 'Win Rate by Trend Regime', 'Regime', 'Win Rate (%)')
for i, (regime, row) in enumerate(trend_regimes.iterrows()):
    ax.bar(regime, row['wr']*100, color=colors_r[i%3], alpha=0.8, edgecolor=DARK_NAVY)
    ax.text(i, row['wr']*100+0.3, f"{row['wr']*100:.1f}%\n({int(row['trades']):,})", ha='center', color=WHITE, fontsize=10)
ax.axhline(y=win_rate*100, color=LIGHT_STEEL, linestyle='--', alpha=0.7)
save_chart(fig, '12_trend_regime')
chart_count += 1

# Chart 13: Rolling 500-Trade Win Rate
rolling_wr = tdf['win'].rolling(500).mean() * 100
fig, ax = plt.subplots(figsize=(16,6))
style_ax(ax, 'Rolling 500-Trade Win Rate', 'Trade Number', 'Win Rate (%)')
ax.plot(range(len(rolling_wr)), rolling_wr.values, color=GOLD, linewidth=1)
ax.axhline(y=win_rate*100, color=GREEN, linestyle='--', alpha=0.7, label=f'Overall: {win_rate*100:.1f}%')
ax.axhline(y=50.5, color=RED, linestyle='--', alpha=0.5, label='Breakeven')
mean_wr_r = rolling_wr.mean()
std_wr_r = rolling_wr.std()
ax.fill_between(range(len(rolling_wr)), mean_wr_r-std_wr_r, mean_wr_r+std_wr_r, color=STEEL, alpha=0.15, label='±1σ')
ax.legend(facecolor=DARK_NAVY, edgecolor=STEEL, labelcolor=WHITE)
save_chart(fig, '13_rolling_win_rate')
chart_count += 1

# Chart 14: Rolling Volatility
rolling_vol = daily_pnl.rolling(30).std()
fig, ax = plt.subplots(figsize=(16,6))
style_ax(ax, 'Rolling 30-Day P&L Volatility', 'Date', 'Std Dev ($)')
ax.plot(rolling_vol.index, rolling_vol.values, color=GOLD, linewidth=1)
ax.axhline(y=daily_pnl.std(), color=GREEN, linestyle='--', alpha=0.7, label=f'Overall: ${daily_pnl.std():.2f}')
ax.legend(facecolor=DARK_NAVY, edgecolor=STEEL, labelcolor=WHITE)
save_chart(fig, '14_rolling_volatility')
chart_count += 1

# Chart 15: Rolling Max DD
rolling_dd = pd.Series(index=daily_cap.index, dtype=float)
for i in range(30, len(daily_cap)):
    window = daily_cap.iloc[i-30:i+1]
    peak = window.cummax()
    dd = ((window - peak) / peak * 100).min()
    rolling_dd.iloc[i] = dd
fig, ax = plt.subplots(figsize=(16,6))
style_ax(ax, 'Rolling 30-Day Maximum Drawdown', 'Date', 'Max Drawdown (%)')
ax.fill_between(rolling_dd.dropna().index, rolling_dd.dropna().values, 0, color=RED, alpha=0.5)
ax.axhline(y=0, color=LIGHT_STEEL, linewidth=0.5)
save_chart(fig, '15_rolling_max_dd')
chart_count += 1

# Chart 16: VaR Cone
var_95 = np.percentile(daily_pnl.values, 5)
var_99 = np.percentile(daily_pnl.values, 1)
fig, ax = plt.subplots(figsize=(14,7))
style_ax(ax, 'Value-at-Risk Analysis', 'Daily P&L ($)', 'Frequency')
ax.hist(daily_pnl.values, bins=60, color=STEEL, alpha=0.7, edgecolor=NAVY)
ax.axvline(x=var_95, color=GOLD, linestyle='--', linewidth=2, label=f'VaR 95%: ${var_95:.2f}')
ax.axvline(x=var_99, color=RED, linestyle='--', linewidth=2, label=f'VaR 99%: ${var_99:.2f}')
ax.axvline(x=0, color=LIGHT_STEEL, linestyle='-', alpha=0.3)
ax.legend(facecolor=DARK_NAVY, edgecolor=STEEL, labelcolor=WHITE)
save_chart(fig, '16_var_analysis')
chart_count += 1

# ═══════════════════════════════════════════════════════════════
# STATISTICAL VALIDATION CHARTS (17-22)
# ═══════════════════════════════════════════════════════════════
print("[5/5] Generating Statistical Validation charts...")

# Chart 17: Monte Carlo P&L Distribution (1,000 quick sims)
np.random.seed(42)
MC_QUICK = 1000
mc_pnls = np.zeros(MC_QUICK)
for run in range(MC_QUICK):
    outcomes = np.where(np.random.random(total_trades) < win_rate, 0.98, -1.02)
    mc_pnls[run] = outcomes.sum()
fig, ax = plt.subplots(figsize=(14,7))
style_ax(ax, f'Monte Carlo P&L Distribution ({MC_QUICK:,} simulations)', 'Final P&L ($)', 'Frequency')
ax.hist(mc_pnls, bins=50, color=STEEL, alpha=0.7, edgecolor=NAVY)
ax.axvline(x=tdf['pnl'].sum(), color=GOLD, linewidth=2.5, linestyle='--', label=f'Actual: ${tdf["pnl"].sum():,.0f}')
ax.axvline(x=np.mean(mc_pnls), color=GREEN, linewidth=1.5, linestyle=':', label=f'MC Mean: ${np.mean(mc_pnls):,.0f}')
ci95 = (np.percentile(mc_pnls, 2.5), np.percentile(mc_pnls, 97.5))
ax.axvspan(ci95[0], ci95[1], alpha=0.1, color=GOLD, label='95% CI')
ax.legend(facecolor=DARK_NAVY, edgecolor=STEEL, labelcolor=WHITE)
save_chart(fig, '17_monte_carlo_pnl')
chart_count += 1

# Chart 18: Bootstrap Win Rate Distribution (1,000 quick)
BOOT_QUICK = 1000
boot_wrs = np.zeros(BOOT_QUICK)
win_flags = tdf['win'].values.astype(int)
for run in range(BOOT_QUICK):
    idx = np.random.randint(0, total_trades, size=total_trades)
    boot_wrs[run] = win_flags[idx].mean() * 100
fig, ax = plt.subplots(figsize=(14,7))
style_ax(ax, f'Bootstrap Win Rate Distribution ({BOOT_QUICK:,} samples)', 'Win Rate (%)', 'Frequency')
ax.hist(boot_wrs, bins=50, color=STEEL, alpha=0.7, edgecolor=NAVY)
ax.axvline(x=win_rate*100, color=GOLD, linewidth=2.5, linestyle='--', label=f'Actual: {win_rate*100:.2f}%')
ax.axvline(x=50.5, color=RED, linewidth=1.5, linestyle=':', label='Breakeven: 50.5%')
ax.legend(facecolor=DARK_NAVY, edgecolor=STEEL, labelcolor=WHITE)
save_chart(fig, '18_bootstrap_win_rate')
chart_count += 1

# Chart 19: Walk-Forward Consistency
wf_window = 6
periods_data = []
start = tdf.index[0]
end_date = tdf.index[-1]
wf_n = 1
while start < end_date:
    window_end = start + pd.DateOffset(months=wf_window)
    if window_end > end_date: window_end = end_date + pd.Timedelta(days=1)
    mask = (tdf.index >= start) & (tdf.index < window_end)
    wdf = tdf[mask]
    if len(wdf) > 0:
        periods_data.append({'label': f'WF{wf_n}', 'wr': wdf['win'].mean()*100, 'pnl': wdf['pnl'].sum()})
    start = window_end; wf_n += 1

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,7))
style_ax(ax1, 'Walk-Forward Win Rate', 'Period', 'Win Rate (%)')
labels = [p['label'] for p in periods_data]
wrs = [p['wr'] for p in periods_data]
colors_wf = [GREEN if w > 50.5 else RED for w in wrs]
ax1.bar(labels, wrs, color=colors_wf, alpha=0.8, edgecolor=DARK_NAVY)
ax1.axhline(y=win_rate*100, color=GOLD, linestyle='--', linewidth=2, label=f'Overall: {win_rate*100:.1f}%')
ax1.axhline(y=50.5, color=RED, linestyle=':', alpha=0.5, label='Breakeven')
ax1.legend(facecolor=DARK_NAVY, edgecolor=STEEL, labelcolor=WHITE, fontsize=9)

style_ax(ax2, 'Walk-Forward P&L', 'Period', 'P&L ($)')
pnls = [p['pnl'] for p in periods_data]
colors_p = [GREEN if v > 0 else RED for v in pnls]
ax2.bar(labels, pnls, color=colors_p, alpha=0.8, edgecolor=DARK_NAVY)
ax2.axhline(y=0, color=LIGHT_STEEL, linewidth=0.5)
save_chart(fig, '19_walk_forward')
chart_count += 1

# Chart 20: Q-Q Plot
from scipy import stats as sp_stats
fig, ax = plt.subplots(figsize=(10,10))
style_ax(ax, 'Q-Q Plot (Daily Returns vs Normal)', 'Theoretical Quantiles', 'Sample Quantiles')
sorted_returns = np.sort(daily_pnl.values)
theoretical = sp_stats.norm.ppf(np.linspace(0.001, 0.999, len(sorted_returns)))
ax.scatter(theoretical, sorted_returns, color=GOLD, s=3, alpha=0.6)
min_v = min(theoretical.min(), sorted_returns.min())
max_v = max(theoretical.max(), sorted_returns.max())
ax.plot([min_v, max_v], [min_v, max_v], color=RED, linestyle='--', alpha=0.7, label='Normal Line')
ax.legend(facecolor=DARK_NAVY, edgecolor=STEEL, labelcolor=WHITE)
save_chart(fig, '20_qq_plot')
chart_count += 1

# Chart 21: Regime Heatmap (Vol x Trend)
regime_pivot = tdf.groupby(['regime_vol', 'regime_trend'])['win'].mean() * 100
vol_order = ['Low', 'Medium', 'High']
trend_order = ['Weak', 'Medium', 'Strong']
heat_data = np.full((3,3), np.nan)
for i, v in enumerate(vol_order):
    for j, t in enumerate(trend_order):
        try: heat_data[i,j] = regime_pivot.loc[(v,t)]
        except: pass
fig, ax = plt.subplots(figsize=(10,8))
style_ax(ax, 'Win Rate by Regime (Vol × Trend)', '', '')
cmap2 = plt.cm.RdYlGn
im = ax.imshow(heat_data, cmap=cmap2, aspect='auto', vmin=48, vmax=60)
ax.set_xticks(range(3)); ax.set_xticklabels(trend_order, color=LIGHT_STEEL, fontsize=12)
ax.set_yticks(range(3)); ax.set_yticklabels(vol_order, color=LIGHT_STEEL, fontsize=12)
ax.set_xlabel('Trend Strength', color=LIGHT_STEEL)
ax.set_ylabel('Volatility', color=LIGHT_STEEL)
for i in range(3):
    for j in range(3):
        if not np.isnan(heat_data[i,j]):
            count = len(tdf[(tdf['regime_vol']==vol_order[i])&(tdf['regime_trend']==trend_order[j])])
            ax.text(j, i, f'{heat_data[i,j]:.1f}%\n({count:,})', ha='center', va='center',
                    color='black' if heat_data[i,j]>54 else 'white', fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax, label='Win Rate (%)')
save_chart(fig, '21_regime_heatmap')
chart_count += 1

# Chart 22-26: Quick additional charts
# Chart 22: Cumulative Wins vs Losses
cum_wins = tdf['win'].cumsum()
cum_losses = (~tdf['win']).cumsum()
fig, ax = plt.subplots(figsize=(16,7))
style_ax(ax, 'Cumulative Wins vs Losses', 'Trade Number', 'Cumulative Count')
ax.plot(range(len(cum_wins)), cum_wins.values, color=GREEN, linewidth=1.2, label='Wins')
ax.plot(range(len(cum_losses)), cum_losses.values, color=RED, linewidth=1.2, label='Losses')
ax.legend(facecolor=DARK_NAVY, edgecolor=STEEL, labelcolor=WHITE)
save_chart(fig, '22_cumulative_wins_losses')
chart_count += 1

# Chart 23: Monthly Trade Count
monthly_trades = tdf.resample('ME').size()
fig, ax = plt.subplots(figsize=(16,6))
style_ax(ax, 'Monthly Trade Count', 'Date', 'Trades')
ax.bar(monthly_trades.index, monthly_trades.values, width=20, color=STEEL, alpha=0.8, edgecolor=NAVY)
ax.axhline(y=monthly_trades.mean(), color=GOLD, linestyle='--', label=f'Avg: {monthly_trades.mean():.0f}')
ax.legend(facecolor=DARK_NAVY, edgecolor=STEEL, labelcolor=WHITE)
save_chart(fig, '23_monthly_trade_count')
chart_count += 1

# Chart 24: Signal Distribution
signal_counts = tdf['signal'].value_counts()
fig, ax = plt.subplots(figsize=(8,8))
style_ax(ax, 'Signal Distribution', '', '')
ax.pie(signal_counts.values, labels=signal_counts.index, colors=[GREEN, RED],
       autopct='%1.1f%%', textprops={'color': WHITE, 'fontsize': 14},
       wedgeprops={'edgecolor': DARK_NAVY, 'linewidth': 2})
save_chart(fig, '24_signal_distribution')
chart_count += 1

# Chart 25: Risk-Adjusted Returns Comparison
sortino = (daily_pnl.mean()/daily_pnl[daily_pnl<0].std())*np.sqrt(365) if daily_pnl[daily_pnl<0].std()>0 else 0
calmar = ((daily_cap.iloc[-1]/INITIAL_CAPITAL)**(365/len(daily_pnl))-1)*100 / abs(dd_pct.min()) if dd_pct.min()<0 else 0
fig, ax = plt.subplots(figsize=(10,7))
style_ax(ax, 'Risk-Adjusted Return Ratios', '', 'Ratio Value')
ratios = {'Sharpe': overall_sharpe, 'Sortino': sortino, 'Calmar': calmar}
bars = ax.bar(ratios.keys(), ratios.values(), color=[GOLD, GREEN, STEEL], alpha=0.8, edgecolor=DARK_NAVY)
for bar, val in zip(bars, ratios.values()):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f'{val:.2f}', ha='center', color=WHITE, fontweight='bold')
save_chart(fig, '25_risk_ratios')
chart_count += 1

# Chart 26: Monte Carlo Equity Paths (100 quick paths)
fig, ax = plt.subplots(figsize=(16,8))
style_ax(ax, 'Monte Carlo Equity Paths (100 simulations)', 'Trade Number', 'Cumulative P&L ($)')
sample_points = np.linspace(0, total_trades-1, 500).astype(int)
for _ in range(100):
    outcomes = np.where(np.random.random(total_trades) < win_rate, 0.98, -1.02)
    cum = INITIAL_CAPITAL + np.cumsum(outcomes)
    ax.plot(sample_points, cum[sample_points], color=LIGHT_STEEL, alpha=0.08, linewidth=0.5)
actual_cum = INITIAL_CAPITAL + tdf['pnl'].cumsum().values
ax.plot(sample_points, actual_cum[sample_points], color=GOLD, linewidth=2, label='Actual Path')
ax.legend(facecolor=DARK_NAVY, edgecolor=STEEL, labelcolor=WHITE)
save_chart(fig, '26_monte_carlo_paths')
chart_count += 1

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
elapsed = time.time() - t0

# Write summary
summary = f"""VISUALIZATION SUITE — GENERATION SUMMARY
{'='*50}
Generated: {pd.Timestamp.now().strftime('%B %d, %Y %H:%M')}
Total Charts: {chart_count}
Output Directory: {VIZ_DIR}
Duration: {elapsed:.1f} seconds

CHART INDEX:
01. Cumulative P&L Equity Curve
02. Daily P&L Distribution (with Normal overlay)
03. Equity Curve with Underwater Periods
04. Rolling 30-Day Sharpe Ratio
05. Monthly Returns Heatmap
06. Drawdown from Peak
07. Win/Loss Streak Distribution
08. Hourly Performance (trades + WR)
09. Day of Week Performance
10. Monthly Profit Factor
11. Volatility Regime Win Rates
12. Trend Regime Win Rates
13. Rolling 500-Trade Win Rate
14. Rolling 30-Day Volatility
15. Rolling 30-Day Max Drawdown
16. Value-at-Risk Analysis
17. Monte Carlo P&L Distribution
18. Bootstrap Win Rate Distribution
19. Walk-Forward Consistency
20. Q-Q Plot (Returns vs Normal)
21. Regime Heatmap (Vol × Trend)
22. Cumulative Wins vs Losses
23. Monthly Trade Count
24. Signal Distribution
25. Risk-Adjusted Return Ratios
26. Monte Carlo Equity Paths

Classification: CONFIDENTIAL
"""

with open(os.path.join(VIZ_DIR, "visualization_summary.txt"), "w", encoding='utf-8') as f:
    f.write(summary)

print(f"\n{'═'*70}")
print(f"  COMPLETE: {chart_count} charts generated")
print(f"  Output: {VIZ_DIR}")
print(f"  Duration: {elapsed:.1f}s")
print(f"{'═'*70}")
