"""
═══════════════════════════════════════════════════════════════════════
05_generate_q3_filter_analysis.py
Q3 Time Filter Analysis — Standalone Analysis Track
Strategy Code: ALPHA-BTC-15M-v2.0
═══════════════════════════════════════════════════════════════════════
Evaluates the baseline strategy with a Q3 time filter applied.
Q3 = trades whose candle starts in the 30–44 minute block of each hour.
These trades are EXCLUDED (skipped) from the Q3-filtered strategy.

Outputs (all in reports/q3_filter/):
  - Q3_FILTER_SUMMARY.txt         Full institutional report
  - q3_filter_trade_log.csv       All Q3-filtered trades
  - q3_filter_daily_metrics.csv
  - q3_filter_monthly_metrics.csv
  - q3_filter_hourly_metrics.csv
  - q3_filter_drawdown_analysis.csv
  - q3_filter_regime_analysis.csv
  - q3_filter_risk_metrics.json
═══════════════════════════════════════════════════════════════════════
"""
import sys, os, time, json
import pandas as pd
import numpy as np
import scipy.stats as stats
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "reports", "q3_filter")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEE_RATE = 0.01
INITIAL_CAPITAL = 100.0
SIM_ENTRY_PRICE = 0.50
FIXED_RISK = 1.02

# ═══════════════════════════════════════════════════════════════
# DATA LOADING & INDICATORS (same pattern as existing scripts)
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
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean()
    up_m, dn_m = df['high'].diff(), -df['low'].diff()
    plus_dm = np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0)
    minus_dm = np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0)
    atr_s = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr_s)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr_s)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    df['adx_14'] = dx.ewm(alpha=1/14, adjust=False).mean()
    atr_14_arr = atr_14.values
    n = len(df)
    atr_pct = np.zeros(n)
    for i in range(28, n):
        lb = min(96, i)
        rec = atr_14_arr[i-lb:i+1]
        if pd.isna(rec).all(): continue
        valid = rec[~np.isnan(rec)]
        if len(valid) < 5: continue
        atr_pct[i] = (valid < atr_14_arr[i]).sum() / len(valid) * 100
    df['atr_pct'] = atr_pct
    df['regime_vol'] = np.where(df['atr_pct'] > 80, 'High', np.where(df['atr_pct'] > 40, 'Medium', 'Low'))
    df['regime_trend'] = np.where(df['adx_14'] > 25, 'Strong', np.where(df['adx_14'] > 15, 'Medium', 'Weak'))
    df = df.dropna(subset=['rsi', 'adx_14']).copy()
    return df

# ═══════════════════════════════════════════════════════════════
# SIMULATION ENGINE
# ═══════════════════════════════════════════════════════════════
def is_q3(timestamp):
    """Return True if timestamp falls in Q3 block (minute 30–44)."""
    minute = pd.Timestamp(timestamp).minute
    return 30 <= minute <= 44

def run_simulation(df, apply_q3_filter=False):
    """Run the strategy simulation. If apply_q3_filter=True, skip Q3 candles."""
    rsi_arr = df['rsi'].values
    adx_arr = df['adx_14'].values
    atr_pct_arr = df['atr_pct'].values
    c_arr = df['close'].values
    ts_arr = df.index.values
    vol_arr = df['regime_vol'].values
    trend_arr = df['regime_trend'].values

    trades = []
    capital = INITIAL_CAPITAL
    for i in range(1, len(df) - 1):
        ts = ts_arr[i]
        rsi = rsi_arr[i]

        buy_yes = rsi < 43
        buy_no = rsi > 57
        if not (buy_yes or buy_no): continue

        adx = adx_arr[i]
        atr_pct = atr_pct_arr[i]
        if adx > 25 and atr_pct > 80: continue

        # Q3 filter: skip trades in the 30-44 minute block
        if apply_q3_filter and is_q3(ts):
            continue

        signal = 'YES' if buy_yes else 'NO'
        target_shares = int(1.0 / SIM_ENTRY_PRICE)
        shares = max(1, target_shares)
        bet_amount = shares * SIM_ENTRY_PRICE
        if bet_amount > capital: continue

        settle_c = c_arr[i + 1]
        won = (signal == 'YES' and settle_c > c_arr[i]) or (signal == 'NO' and settle_c < c_arr[i])
        fees = bet_amount * FEE_RATE * 2
        pnl = (shares * 1.0) - bet_amount - fees if won else -bet_amount - fees
        capital += pnl

        trades.append({
            'timestamp': ts,
            'signal': signal,
            'win': won,
            'pnl': pnl,
            'capital': capital,
            'regime_vol': vol_arr[i],
            'regime_trend': trend_arr[i]
        })

    tdf = pd.DataFrame(trades)
    if len(tdf) > 0:
        tdf['timestamp'] = pd.to_datetime(tdf['timestamp'])
        tdf.set_index('timestamp', inplace=True)
    return tdf

# ═══════════════════════════════════════════════════════════════
# METRICS COMPUTATION
# ═══════════════════════════════════════════════════════════════
def compute_full_metrics(tdf, label="Strategy"):
    """Compute a complete metrics dictionary for a trade DataFrame."""
    m = {}
    total = len(tdf)
    if total == 0:
        return {'label': label, 'total_trades': 0}

    m['label'] = label
    m['total_trades'] = total
    wins = tdf['win'].sum()
    losses = total - wins
    m['wins'] = int(wins)
    m['losses'] = int(losses)
    m['win_rate'] = wins / total * 100
    m['total_pnl'] = tdf['pnl'].sum()
    m['expectancy'] = m['total_pnl'] / total

    gross_prof = tdf[tdf['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(tdf[tdf['pnl'] < 0]['pnl'].sum())
    m['gross_profit'] = gross_prof
    m['gross_loss'] = gross_loss
    m['profit_factor'] = gross_prof / gross_loss if gross_loss > 0 else 999

    m['avg_win'] = tdf[tdf['win'] == True]['pnl'].mean() if wins > 0 else 0
    m['avg_loss'] = abs(tdf[tdf['win'] == False]['pnl'].mean()) if losses > 0 else 0
    m['largest_win'] = tdf['pnl'].max()
    m['largest_loss'] = tdf['pnl'].min()

    # Daily P&L
    daily_pnl = tdf['pnl'].resample('1D').sum().fillna(0)
    daily_cap = INITIAL_CAPITAL + daily_pnl.cumsum()
    m['final_capital'] = daily_cap.iloc[-1]
    m['total_return_pct'] = (m['final_capital'] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    # Drawdown
    dd_usd = pd.Series(0.0, index=daily_pnl.index)
    underwater_mask = daily_cap < INITIAL_CAPITAL
    dd_usd[underwater_mask] = INITIAL_CAPITAL - daily_cap[underwater_mask]
    dd_pct = (dd_usd / INITIAL_CAPITAL) * 100
    
    m['max_dd_pct'] = dd_pct.max()
    m['max_dd_usd'] = dd_usd.max()
    m['avg_dd_pct'] = dd_pct[dd_pct > 0].mean() if (dd_pct > 0).any() else 0

    is_dd = dd_pct > 0
    dd_groups = (~is_dd).cumsum()[is_dd]
    if not dd_groups.empty:
        dd_lens = dd_groups.groupby(dd_groups).apply(len)
        m['max_dd_duration'] = int(dd_lens.max())
        m['avg_dd_duration'] = float(dd_lens.mean())
    else:
        m['max_dd_duration'] = 0
        m['avg_dd_duration'] = 0.0

    # Risk-adjusted ratios
    mean_daily = daily_pnl.mean()
    std_daily = daily_pnl.std()
    n_days = len(daily_pnl)
    m['sharpe'] = (mean_daily / std_daily) * np.sqrt(365) if std_daily > 0 else 0
    neg_std = daily_pnl[daily_pnl < 0].std()
    m['sortino'] = (mean_daily / neg_std) * np.sqrt(365) if neg_std > 0 else 0
    cagr = ((daily_cap.iloc[-1] / INITIAL_CAPITAL) ** (1 / max(1/365, n_days/365)) - 1) * 100
    m['cagr'] = cagr
    m['calmar'] = cagr / m['max_dd_pct'] if m['max_dd_pct'] > 0 else 999

    # Streaks
    winning_streaks = tdf['win'].groupby((~tdf['win']).cumsum()).sum()
    losing_streaks = (~tdf['win']).groupby(tdf['win'].cumsum()).sum()
    m['max_win_streak'] = int(winning_streaks.max())
    m['max_loss_streak'] = int(losing_streaks.max())
    m['avg_win_streak'] = float(winning_streaks[winning_streaks > 0].mean())
    m['avg_loss_streak'] = float(losing_streaks[losing_streaks > 0].mean())

    # Monthly
    monthly_pnl = tdf['pnl'].resample('ME').sum()
    m['total_months'] = len(monthly_pnl)
    m['profitable_months'] = int((monthly_pnl > 0).sum())
    m['losing_months'] = int((monthly_pnl <= 0).sum())
    m['best_month'] = monthly_pnl.max()
    m['worst_month'] = monthly_pnl.min()
    m['avg_monthly_pnl'] = monthly_pnl.mean()
    m['monthly_consistency'] = m['profitable_months'] / m['total_months'] * 100 if m['total_months'] > 0 else 0

    # Statistical significance
    p_random = 0.50
    std_err = np.sqrt(p_random * (1 - p_random) / total)
    m['z_score'] = (m['win_rate'] / 100.0 - p_random) / std_err
    m['p_value'] = stats.norm.sf(abs(m['z_score'])) * 2

    # Volatility
    m['daily_std'] = std_daily
    m['monthly_std'] = monthly_pnl.std()
    m['annualized_vol'] = (std_daily / INITIAL_CAPITAL) * np.sqrt(365) * 100

    return m

# ═══════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════
print("═" * 70)
print("  Q3 FILTER ANALYSIS — Standalone Analysis Track")
print("  Strategy Code: ALPHA-BTC-15M-v2.0")
print(f"  Started: {datetime.now().strftime('%B %d, %Y %H:%M:%S')}")
print("═" * 70)

t0 = time.time()

# Load and prepare data
print("\n[1/6] Loading data and computing indicators...")
df = load_data(os.path.join(BASE_DIR, 'data', 'BTCUSDT_15m_3_years.csv'))
total_candles = len(df)
df = compute_indicators(df)
print(f"  {total_candles:,} candles loaded, {len(df):,} after indicator warmup")

# Run both simulations
print("\n[2/6] Running baseline simulation...")
baseline_tdf = run_simulation(df, apply_q3_filter=False)
print(f"  Baseline: {len(baseline_tdf):,} trades")

print("\n[3/6] Running Q3-filtered simulation...")
q3_tdf = run_simulation(df, apply_q3_filter=True)
print(f"  Q3-Filtered: {len(q3_tdf):,} trades")

# Count Q3 trades skipped
q3_skipped = len(baseline_tdf) - len(q3_tdf)
print(f"  Trades skipped by Q3 filter: {q3_skipped:,}")

# Compute metrics
print("\n[4/6] Computing metrics...")
bm = compute_full_metrics(baseline_tdf, "Baseline")
qm = compute_full_metrics(q3_tdf, "Q3-Filtered")

# ═══════════════════════════════════════════════════════════════
# STATISTICAL COMPARISON
# ═══════════════════════════════════════════════════════════════
print("\n[5/6] Running statistical comparisons...")

# Z-test on win rate difference
p1 = bm['win_rate'] / 100.0
p2 = qm['win_rate'] / 100.0
n1 = bm['total_trades']
n2 = qm['total_trades']
p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
se_diff = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
z_diff = (p2 - p1) / se_diff if se_diff > 0 else 0
p_val_diff = stats.norm.sf(abs(z_diff)) * 2

# Bootstrap comparison (1,000 iterations)
np.random.seed(42)
BOOT_N = 1000
baseline_pnls = baseline_tdf['pnl'].values
q3_pnls = q3_tdf['pnl'].values

boot_baseline_means = np.zeros(BOOT_N)
boot_q3_means = np.zeros(BOOT_N)
for i in range(BOOT_N):
    boot_baseline_means[i] = np.random.choice(baseline_pnls, size=len(baseline_pnls), replace=True).mean()
    boot_q3_means[i] = np.random.choice(q3_pnls, size=len(q3_pnls), replace=True).mean()

boot_diff = boot_q3_means - boot_baseline_means
boot_ci_lower = np.percentile(boot_diff, 2.5)
boot_ci_upper = np.percentile(boot_diff, 97.5)
boot_q3_better_pct = (boot_diff > 0).sum() / BOOT_N * 100

# Bootstrap win rate CIs
boot_baseline_wr = np.zeros(BOOT_N)
boot_q3_wr = np.zeros(BOOT_N)
baseline_wins = baseline_tdf['win'].values.astype(int)
q3_wins = q3_tdf['win'].values.astype(int)
for i in range(BOOT_N):
    boot_baseline_wr[i] = np.random.choice(baseline_wins, size=len(baseline_wins), replace=True).mean() * 100
    boot_q3_wr[i] = np.random.choice(q3_wins, size=len(q3_wins), replace=True).mean() * 100

baseline_wr_ci = (np.percentile(boot_baseline_wr, 2.5), np.percentile(boot_baseline_wr, 97.5))
q3_wr_ci = (np.percentile(boot_q3_wr, 2.5), np.percentile(boot_q3_wr, 97.5))

# ═══════════════════════════════════════════════════════════════
# GENERATE CSV OUTPUTS
# ═══════════════════════════════════════════════════════════════

# Trade log
q3_tdf_export = q3_tdf.copy()
q3_tdf_export.to_csv(os.path.join(OUTPUT_DIR, "q3_filter_trade_log.csv"))

# Daily metrics
daily_pnl_q3 = q3_tdf['pnl'].resample('1D').sum().fillna(0)
daily_cap_q3 = INITIAL_CAPITAL + daily_pnl_q3.cumsum()
dd_usd = pd.Series(0.0, index=daily_cap_q3.index)
underwater_mask = daily_cap_q3 < INITIAL_CAPITAL
dd_usd[underwater_mask] = INITIAL_CAPITAL - daily_cap_q3[underwater_mask]
daily_dd_pct = (dd_usd / INITIAL_CAPITAL * 100)
daily_trades = q3_tdf.resample('1D').size()
daily_wr = q3_tdf.resample('1D')['win'].mean() * 100

daily_df = pd.DataFrame({
    'date': daily_pnl_q3.index,
    'trades': daily_trades.reindex(daily_pnl_q3.index, fill_value=0).values,
    'pnl': daily_pnl_q3.values,
    'cumulative_pnl': daily_pnl_q3.cumsum().values,
    'capital': daily_cap_q3.values,
    'drawdown_pct': daily_dd_pct.values,
    'win_rate': daily_wr.reindex(daily_pnl_q3.index).values
})
daily_df.to_csv(os.path.join(OUTPUT_DIR, "q3_filter_daily_metrics.csv"), index=False)

# Monthly metrics
monthly_pnl_q3 = q3_tdf['pnl'].resample('ME').sum()
monthly_trades_q3 = q3_tdf.resample('ME').size()
monthly_wr_q3 = q3_tdf.resample('ME')['win'].mean() * 100
monthly_wins_q3 = q3_tdf[q3_tdf['pnl'] > 0].resample('ME')['pnl'].sum()
monthly_losses_q3 = q3_tdf[q3_tdf['pnl'] < 0].resample('ME')['pnl'].sum().abs()
monthly_pf_q3 = (monthly_wins_q3 / monthly_losses_q3).fillna(0).replace([np.inf], 5)

monthly_df = pd.DataFrame({
    'month': monthly_pnl_q3.index.strftime('%Y-%m'),
    'trades': monthly_trades_q3.values,
    'win_rate': monthly_wr_q3.values,
    'pnl': monthly_pnl_q3.values,
    'cumulative_pnl': monthly_pnl_q3.cumsum().values,
    'profit_factor': monthly_pf_q3.reindex(monthly_pnl_q3.index, fill_value=0).values
})
monthly_df.to_csv(os.path.join(OUTPUT_DIR, "q3_filter_monthly_metrics.csv"), index=False)

# Hourly metrics
hourly_grp = q3_tdf.groupby(q3_tdf.index.hour).agg(
    trades=('win', 'count'),
    wins=('win', 'sum'),
    win_rate=('win', lambda x: x.mean() * 100),
    pnl=('pnl', 'sum'),
    avg_pnl=('pnl', 'mean')
)
hourly_grp['is_q3_hour'] = False  # Q3 candles are excluded, so by definition no hour is fully Q3
# Mark which hours had candles removed
for hr in range(24):
    # The :30 candle of each hour falls in Q3 and is excluded
    hourly_grp.loc[hr, 'q3_candles_removed'] = True if hr in range(24) else False
hourly_grp.index.name = 'hour'
hourly_grp.to_csv(os.path.join(OUTPUT_DIR, "q3_filter_hourly_metrics.csv"))

# Drawdown analysis
daily_cap_q3_series = daily_cap_q3
dd_usd_q3 = pd.Series(0.0, index=daily_cap_q3_series.index)
underwater_mask = daily_cap_q3_series < INITIAL_CAPITAL
dd_usd_q3[underwater_mask] = INITIAL_CAPITAL - daily_cap_q3_series[underwater_mask]
dd_pct_q3 = (dd_usd_q3 / INITIAL_CAPITAL) * 100

# Find individual drawdown events
is_dd = dd_pct_q3 > 0
dd_groups = (~is_dd).cumsum()[is_dd]
dd_events = []
if not dd_groups.empty:
    for gid, group in dd_groups.groupby(dd_groups):
        start_date = group.index[0]
        end_date = group.index[-1]
        depth = dd_pct_q3.loc[group.index].max()
        duration = len(group)
        dd_events.append({
            'start_date': start_date,
            'end_date': end_date,
            'depth_pct': depth,
            'depth_usd': dd_usd_q3.loc[group.index].max(),
            'duration_days': duration
        })

dd_df = pd.DataFrame(dd_events)
if len(dd_df) > 0:
    dd_df = dd_df.sort_values('depth_pct', ascending=False).head(20)  # Top 20 worst drawdowns
dd_df.to_csv(os.path.join(OUTPUT_DIR, "q3_filter_drawdown_analysis.csv"), index=False)

# Regime analysis
regime_data = []
for vol in ['Low', 'Medium', 'High']:
    for trend in ['Weak', 'Medium', 'Strong']:
        mask = (q3_tdf['regime_vol'] == vol) & (q3_tdf['regime_trend'] == trend)
        subset = q3_tdf[mask]
        if len(subset) > 0:
            regime_data.append({
                'volatility': vol,
                'trend': trend,
                'trades': len(subset),
                'win_rate': subset['win'].mean() * 100,
                'pnl': subset['pnl'].sum(),
                'avg_pnl': subset['pnl'].mean(),
                'profit_factor': subset[subset['pnl'] > 0]['pnl'].sum() / abs(subset[subset['pnl'] < 0]['pnl'].sum()) if abs(subset[subset['pnl'] < 0]['pnl'].sum()) > 0 else 999
            })
regime_df = pd.DataFrame(regime_data)
regime_df.to_csv(os.path.join(OUTPUT_DIR, "q3_filter_regime_analysis.csv"), index=False)

# Risk metrics JSON
risk_metrics = {
    'strategy': 'Q3-Filtered (RSI 43/58 + ADX/ATR, skip minute 30-44)',
    'generated': datetime.now().isoformat(),
    'data_period': {
        'start': str(q3_tdf.index[0]),
        'end': str(q3_tdf.index[-1]),
        'days': int((q3_tdf.index[-1] - q3_tdf.index[0]).days)
    },
    'performance': {
        'total_trades': qm['total_trades'],
        'win_rate': round(qm['win_rate'], 4),
        'total_pnl': round(qm['total_pnl'], 2),
        'expectancy': round(qm['expectancy'], 4),
        'profit_factor': round(qm['profit_factor'], 4)
    },
    'risk_adjusted': {
        'sharpe_ratio': round(qm['sharpe'], 4),
        'sortino_ratio': round(qm['sortino'], 4),
        'calmar_ratio': round(qm['calmar'], 4),
        'cagr_pct': round(qm['cagr'], 4)
    },
    'drawdown': {
        'max_drawdown_pct': round(qm['max_dd_pct'], 4),
        'max_drawdown_usd': round(qm['max_dd_usd'], 2),
        'avg_drawdown_pct': round(qm['avg_dd_pct'], 4),
        'max_dd_duration_days': qm['max_dd_duration']
    },
    'streaks': {
        'max_win_streak': qm['max_win_streak'],
        'max_loss_streak': qm['max_loss_streak'],
        'avg_win_streak': round(qm['avg_win_streak'], 2),
        'avg_loss_streak': round(qm['avg_loss_streak'], 2)
    },
    'statistical_significance': {
        'z_score': round(qm['z_score'], 4),
        'p_value': round(qm['p_value'], 10),
        'significant_at_95pct': qm['p_value'] < 0.05,
        'significant_at_99pct': qm['p_value'] < 0.01
    },
    'q3_filter_impact': {
        'baseline_trades': bm['total_trades'],
        'q3_filtered_trades': qm['total_trades'],
        'trades_removed': q3_skipped,
        'removal_pct': round(q3_skipped / bm['total_trades'] * 100, 2),
        'win_rate_change': round(qm['win_rate'] - bm['win_rate'], 4),
        'pnl_change': round(qm['total_pnl'] - bm['total_pnl'], 2),
        'sharpe_change': round(qm['sharpe'] - bm['sharpe'], 4),
        'z_test_difference_p_value': round(p_val_diff, 6),
        'bootstrap_q3_better_pct': round(boot_q3_better_pct, 2)
    }
}
with open(os.path.join(OUTPUT_DIR, "q3_filter_risk_metrics.json"), "w") as f:
    json.dump(risk_metrics, f, indent=2, default=str)

# ═══════════════════════════════════════════════════════════════
# GENERATE INSTITUTIONAL SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════
print("\n[6/6] Generating institutional summary report...")

# Helper for comparison
def delta_str(q3_val, base_val, fmt=".2f", higher_better=True, absolute_dd=False):
    # Absolute drawdowns are stored as positive values now
    diff = q3_val - base_val
    if absolute_dd:
        # For drawdown, a negative difference means the drawdown got smaller (which is better)
        arrow = "▼" if diff < 0 else "▲" if diff > 0 else "─"
        quality = " (better)" if diff < 0 else " (worse)" if diff > 0 else ""
        return f"{diff:+{fmt}} {arrow}{quality}"

    arrow = "▲" if diff > 0 else "▼" if diff < 0 else "─"
    quality = ""
    if diff != 0:
        if higher_better:
            quality = " (better)" if diff > 0 else " (worse)"
        else:
            quality = " (better)" if diff < 0 else " (worse)"
    return f"{diff:+{fmt}} {arrow}{quality}"

# Hourly performance for baseline
baseline_hourly = baseline_tdf.groupby(baseline_tdf.index.hour).agg(
    trades=('win', 'count'), wr=('win', lambda x: x.mean() * 100), pnl=('pnl', 'sum'))

# Day-of-week performance
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
q3_dow = q3_tdf.groupby(q3_tdf.index.dayofweek).agg(
    trades=('win', 'count'), wr=('win', lambda x: x.mean() * 100), pnl=('pnl', 'sum'))

# Q3-only trades (trades that were skipped)
baseline_minutes = pd.to_datetime(baseline_tdf.index).minute
q3_only_mask = (baseline_minutes >= 30) & (baseline_minutes <= 44)
q3_only_tdf = baseline_tdf[q3_only_mask]
q3_only_trades = len(q3_only_tdf)
q3_only_wr = q3_only_tdf['win'].mean() * 100 if q3_only_trades > 0 else 0
q3_only_pnl = q3_only_tdf['pnl'].sum() if q3_only_trades > 0 else 0

report = f"""═══════════════════════════════════════════════════════════════════════
Q3 FILTER ANALYSIS — INSTITUTIONAL REPORT
ALGORITHMIC TRADING STRATEGY — Q3 TIME FILTER EVALUATION
═══════════════════════════════════════════════════════════════════════

Generated: {datetime.now().strftime('%B %d, %Y %H:%M:%S')}
Classification: CONFIDENTIAL
Strategy Code: ALPHA-BTC-15M-v2.0 (Q3 Filter Variant)

═══════════════════════════════════════════════════════════════════════
SECTION 1: STRATEGY DEFINITION
═══════════════════════════════════════════════════════════════════════

1.1 BASELINE STRATEGY RULES
────────────────────────────────────────────────────────────────────────
  Signal Generation:
    • BUY YES (Long):     RSI(14) < 43
    • BUY NO  (Short):    RSI(14) > 57
    • No signal:          43 ≤ RSI ≤ 57

  Quality Filter:
    • SKIP trade if:      ADX(14) > 25 AND ATR_percentile > 80
    • Rationale:          Avoid strong-trend + high-volatility regimes

  Execution:
    • Entry:              At candle close
    • Settlement:         Next candle close (15 minutes later)
    • Risk per trade:     ~$1.02 (fixed)
    • Fees:               1% entry + 1% exit

1.2 Q3 FILTER DEFINITION
────────────────────────────────────────────────────────────────────────
  Q3 Block:              Minutes 30–44 of each hour
  Timestamp Logic:       Trade is Q3 if candle_timestamp.minute ∈ [30, 44]
  Filter Action:         SKIP (exclude) — Q3 trades are NOT taken
  Rationale:             Evaluate whether the 30-44 minute block
                         degrades strategy performance

  For 15-minute candles, this applies to candles opening at :30
  (which covers the 30:00–44:59 window within each hour).

  Q3 trades in baseline:  {q3_only_trades:,} trades
  Q3 win rate:             {q3_only_wr:.2f}%
  Q3 total P&L:            ${q3_only_pnl:,.2f}

═══════════════════════════════════════════════════════════════════════
SECTION 2: PERFORMANCE METRICS
═══════════════════════════════════════════════════════════════════════

2.1 CORE METRICS — Q3-FILTERED STRATEGY
────────────────────────────────────────────────────────────────────────

  Total Trades:             {qm['total_trades']:,}
  Winning Trades:           {qm['wins']:,} ({qm['win_rate']:.2f}%)
  Losing Trades:            {qm['losses']:,} ({qm['losses']/qm['total_trades']*100:.2f}%)

  Total P&L:                ${qm['total_pnl']:,.2f}
  Gross Profit:             ${qm['gross_profit']:,.2f}
  Gross Loss:               -${qm['gross_loss']:,.2f}

  Expectancy per Trade:     ${qm['expectancy']:.4f}
  Profit Factor:            {qm['profit_factor']:.2f}

  Average Win:              ${qm['avg_win']:.4f}
  Average Loss:             -${qm['avg_loss']:.4f}
  Largest Win:              ${qm['largest_win']:.4f}
  Largest Loss:             ${qm['largest_loss']:.4f}

2.2 RISK-ADJUSTED RETURNS
────────────────────────────────────────────────────────────────────────

  Sharpe Ratio:             {qm['sharpe']:.2f} (annualized)
  Sortino Ratio:            {qm['sortino']:.2f}
  Calmar Ratio:             {qm['calmar']:.2f} (CAGR / Max DD)
  CAGR:                     {qm['cagr']:.1f}%

2.3 DRAWDOWN ANALYSIS
────────────────────────────────────────────────────────────────────────

  Maximum Drawdown:         {qm['max_dd_pct']:.1f}% (-${qm['max_dd_usd']:.2f})
  Average Drawdown:         {qm['avg_dd_pct']:.1f}%
  Max DD Duration:          {qm['max_dd_duration']} days
  Avg DD Duration:          {qm['avg_dd_duration']:.1f} days

2.4 STREAK ANALYSIS
────────────────────────────────────────────────────────────────────────

  Longest Winning Streak:   {qm['max_win_streak']} trades
  Longest Losing Streak:    {qm['max_loss_streak']} trades
  Average Win Streak:       {qm['avg_win_streak']:.1f} trades
  Average Loss Streak:      {qm['avg_loss_streak']:.1f} trades

2.5 MONTHLY RETURN CONSISTENCY
────────────────────────────────────────────────────────────────────────

  Total Months:             {qm['total_months']}
  Profitable Months:        {qm['profitable_months']} ({qm['monthly_consistency']:.1f}%)
  Losing Months:            {qm['losing_months']}
  Best Month:               ${qm['best_month']:,.2f}
  Worst Month:              ${qm['worst_month']:,.2f}
  Avg Monthly P&L:          ${qm['avg_monthly_pnl']:,.2f}

2.6 HOUR-OF-DAY PERFORMANCE (Q3-Filtered)
────────────────────────────────────────────────────────────────────────
Hour | Trades | Win%  | P&L
-----|--------|-------|------
"""

q3_hourly = q3_tdf.groupby(q3_tdf.index.hour).agg(
    trades=('win', 'count'), wr=('win', lambda x: x.mean() * 100), pnl=('pnl', 'sum'))
for hr, row in q3_hourly.iterrows():
    report += f"{hr:02d}   | {int(row['trades']):<6,} | {row['wr']:<5.1f}% | ${row['pnl']:<4.0f}\n"

report += f"""
2.7 DAY-OF-WEEK PERFORMANCE (Q3-Filtered)
────────────────────────────────────────────────────────────────────────
Day       | Trades | Win%  | P&L
----------|--------|-------|-------
"""

for dow, row in q3_dow.iterrows():
    report += f"{days[dow]:<9} | {int(row['trades']):<6,} | {row['wr']:<5.1f}% | ${row['pnl']:<4.0f}\n"

report += f"""
═══════════════════════════════════════════════════════════════════════
SECTION 3: BASELINE vs Q3-FILTERED COMPARISON
═══════════════════════════════════════════════════════════════════════

3.1 SIDE-BY-SIDE METRICS
────────────────────────────────────────────────────────────────────────

Metric                    | Baseline       | Q3-Filtered    | Delta
--------------------------|----------------|----------------|------------------
Total Trades              | {bm['total_trades']:<14,} | {qm['total_trades']:<14,} | {delta_str(qm['total_trades'], bm['total_trades'], ',d', False)}
Win Rate                  | {bm['win_rate']:<13.2f}% | {qm['win_rate']:<13.2f}% | {delta_str(qm['win_rate'], bm['win_rate'], '.2f')}
Total P&L                 | ${bm['total_pnl']:<13,.2f} | ${qm['total_pnl']:<13,.2f} | {delta_str(qm['total_pnl'], bm['total_pnl'], '.2f')}
Expectancy                | ${bm['expectancy']:<13.4f} | ${qm['expectancy']:<13.4f} | {delta_str(qm['expectancy'], bm['expectancy'], '.4f')}
Profit Factor             | {bm['profit_factor']:<14.2f} | {qm['profit_factor']:<14.2f} | {delta_str(qm['profit_factor'], bm['profit_factor'], '.2f')}
Sharpe Ratio              | {bm['sharpe']:<14.2f} | {qm['sharpe']:<14.2f} | {delta_str(qm['sharpe'], bm['sharpe'], '.2f')}
Sortino Ratio             | {bm['sortino']:<14.2f} | {qm['sortino']:<14.2f} | {delta_str(qm['sortino'], bm['sortino'], '.2f')}
Calmar Ratio              | {bm['calmar']:<14.2f} | {qm['calmar']:<14.2f} | {delta_str(qm['calmar'], bm['calmar'], '.2f')}
Max Drawdown              | {bm['max_dd_pct']:<13.1f}% | {qm['max_dd_pct']:<13.1f}% | {delta_str(qm['max_dd_pct'], bm['max_dd_pct'], '.1f', higher_better=False, absolute_dd=True)}
Avg Drawdown              | {bm['avg_dd_pct']:<13.1f}% | {qm['avg_dd_pct']:<13.1f}% | {delta_str(qm['avg_dd_pct'], bm['avg_dd_pct'], '.1f', higher_better=False, absolute_dd=True)}
Max DD Duration           | {bm['max_dd_duration']:<14} | {qm['max_dd_duration']:<14} | {delta_str(qm['max_dd_duration'], bm['max_dd_duration'], 'd', False)} days
Max Win Streak            | {bm['max_win_streak']:<14} | {qm['max_win_streak']:<14} | {delta_str(qm['max_win_streak'], bm['max_win_streak'], 'd')}
Max Loss Streak           | {bm['max_loss_streak']:<14} | {qm['max_loss_streak']:<14} | {delta_str(qm['max_loss_streak'], bm['max_loss_streak'], 'd', False)}
Profitable Months         | {bm['profitable_months']:<14} | {qm['profitable_months']:<14} | {delta_str(qm['profitable_months'], bm['profitable_months'], 'd')}
Best Month                | ${bm['best_month']:<13,.2f} | ${qm['best_month']:<13,.2f} | {delta_str(qm['best_month'], bm['best_month'], '.2f')}
Worst Month               | ${bm['worst_month']:<13,.2f} | ${qm['worst_month']:<13,.2f} | {delta_str(qm['worst_month'], bm['worst_month'], '.2f', False)}

3.2 IMPACT SUMMARY
────────────────────────────────────────────────────────────────────────

  Trades Removed:           {q3_skipped:,} ({q3_skipped/bm['total_trades']*100:.1f}% of baseline)
  Win Rate Change:          {delta_str(qm['win_rate'], bm['win_rate'], '.2f')}%
  P&L Change:               {delta_str(qm['total_pnl'], bm['total_pnl'], ',.2f')}
  Sharpe Change:            {delta_str(qm['sharpe'], bm['sharpe'], '.2f')}
  Max DD Change:            {delta_str(qm['max_dd_pct'], bm['max_dd_pct'], '.1f', higher_better=False, absolute_dd=True)}%
  Trade Frequency Change:   {delta_str(qm['total_trades']/qm['total_months'], bm['total_trades']/bm['total_months'], '.0f', False)} trades/month

3.3 Q3-ONLY TRADE ANALYSIS (Trades That Were Removed)
────────────────────────────────────────────────────────────────────────

  Q3-Only Trades:           {q3_only_trades:,}
  Q3-Only Win Rate:         {q3_only_wr:.2f}%
  Q3-Only Total P&L:        ${q3_only_pnl:,.2f}
  Q3-Only Expectancy:       ${q3_only_pnl/q3_only_trades:.4f} per trade
  
  vs Baseline WR:           {q3_only_wr - bm['win_rate']:+.2f}%
  vs Baseline Expectancy:   {q3_only_pnl/q3_only_trades - bm['expectancy']:+.4f}
  
  Interpretation:           {'Q3 trades UNDERPERFORM baseline — filtering improves quality' if q3_only_wr < bm['win_rate'] else 'Q3 trades perform at or ABOVE baseline — filtering may not help'}

═══════════════════════════════════════════════════════════════════════
SECTION 4: STATISTICAL DEPTH
═══════════════════════════════════════════════════════════════════════

4.1 WIN RATE SIGNIFICANCE TESTING
────────────────────────────────────────────────────────────────────────

  Baseline Win Rate:        {bm['win_rate']:.2f}% (n={bm['total_trades']:,})
  Q3-Filtered Win Rate:     {qm['win_rate']:.2f}% (n={qm['total_trades']:,})
  
  Difference:               {qm['win_rate'] - bm['win_rate']:+.2f}%
  
  Z-Test for Difference:
    Z-statistic:            {z_diff:.4f}
    P-value:                {p_val_diff:.6f}
    Significance:           {'Significant at 95%' if p_val_diff < 0.05 else 'Not significant at 95%'}
    Conclusion:             {'The win rate difference is statistically significant' if p_val_diff < 0.05 else 'The win rate difference is NOT statistically significant (likely noise)'}

4.2 BOOTSTRAP RESAMPLING COMPARISON ({BOOT_N:,} iterations)
────────────────────────────────────────────────────────────────────────

  Mean P&L per Trade Comparison:
    Baseline 95% CI:        [${np.percentile(boot_baseline_means, 2.5):.4f}, ${np.percentile(boot_baseline_means, 97.5):.4f}]
    Q3-Filtered 95% CI:     [${np.percentile(boot_q3_means, 2.5):.4f}, ${np.percentile(boot_q3_means, 97.5):.4f}]
    
  Difference (Q3 - Baseline):
    95% CI:                 [${boot_ci_lower:.4f}, ${boot_ci_upper:.4f}]
    Q3 Better:              {boot_q3_better_pct:.1f}% of bootstrap samples
    Conclusion:             {'Q3 filter consistently improves mean P&L' if boot_q3_better_pct > 95 else 'Q3 filter does NOT consistently improve mean P&L' if boot_q3_better_pct < 50 else 'Q3 filter shows marginal improvement — not definitive'}

  Win Rate Confidence Intervals:
    Baseline WR CI:         [{baseline_wr_ci[0]:.2f}%, {baseline_wr_ci[1]:.2f}%]
    Q3-Filtered WR CI:      [{q3_wr_ci[0]:.2f}%, {q3_wr_ci[1]:.2f}%]
    CIs Overlap:            {'Yes — difference may not be meaningful' if q3_wr_ci[0] < baseline_wr_ci[1] and baseline_wr_ci[0] < q3_wr_ci[1] else 'No — win rates are statistically distinct'}

4.3 REGIME ANALYSIS
────────────────────────────────────────────────────────────────────────

Volatility x Trend Win Rate (Q3-Filtered):
Vol/Trend    | Weak         | Medium       | Strong
-------------|--------------|--------------|-------------
"""

vol_order = ['Low', 'Medium', 'High']
trend_order = ['Weak', 'Medium', 'Strong']
for vol in vol_order:
    row_str = f"{vol:<12} |"
    for trend in trend_order:
        mask = (q3_tdf['regime_vol'] == vol) & (q3_tdf['regime_trend'] == trend)
        subset = q3_tdf[mask]
        if len(subset) > 0:
            wr = subset['win'].mean() * 100
            row_str += f" {wr:5.1f}% ({len(subset):,}) |"
        else:
            row_str += f" {'N/A':>12} |"
    report += row_str + "\n"

report += f"""
4.4 TRADE OUTCOME DISTRIBUTION
────────────────────────────────────────────────────────────────────────

  Baseline P&L Distribution:
    Mean:                   ${baseline_pnls.mean():.4f}
    Std Dev:                ${baseline_pnls.std():.4f}
    Skewness:               {stats.skew(baseline_pnls):.4f}
    Kurtosis:               {stats.kurtosis(baseline_pnls):.4f}

  Q3-Filtered P&L Distribution:
    Mean:                   ${q3_pnls.mean():.4f}
    Std Dev:                ${q3_pnls.std():.4f}
    Skewness:               {stats.skew(q3_pnls):.4f}
    Kurtosis:               {stats.kurtosis(q3_pnls):.4f}

  Kolmogorov-Smirnov Test:
    KS-Statistic:           {stats.ks_2samp(baseline_pnls, q3_pnls).statistic:.6f}
    P-value:                {stats.ks_2samp(baseline_pnls, q3_pnls).pvalue:.6f}
    Conclusion:             {'Distributions are significantly different' if stats.ks_2samp(baseline_pnls, q3_pnls).pvalue < 0.05 else 'Distributions are NOT significantly different'}

4.5 MEANINGFUL OR NOISE?
────────────────────────────────────────────────────────────────────────

  Evidence Summary:
    Win Rate Z-Test:        {'✅ Significant' if p_val_diff < 0.05 else '❌ Not Significant'}
    Bootstrap Advantage:    {'✅ Q3 better >' if boot_q3_better_pct > 75 else '❌ Inconclusive' if boot_q3_better_pct > 50 else '❌ Q3 worse'}75% of time: {boot_q3_better_pct:.1f}%
    Distribution Test:      {'✅ Different' if stats.ks_2samp(baseline_pnls, q3_pnls).pvalue < 0.05 else '❌ Same'}
    Q3 Trades WR vs Base:   {'✅ Underperform' if q3_only_wr < bm['win_rate'] else '❌ No underperformance'}

  Conclusion:               """

# Determine verdict
evidence_count = 0
if p_val_diff < 0.05: evidence_count += 1
if boot_q3_better_pct > 75: evidence_count += 1
if q3_only_wr < bm['win_rate']: evidence_count += 1
if stats.ks_2samp(baseline_pnls, q3_pnls).pvalue < 0.05: evidence_count += 1

if evidence_count >= 3:
    verdict_text = "LIKELY MEANINGFUL — Multiple statistical tests support the Q3 filter"
    recommendation = "RECOMMENDED — Implement Q3 filter for live trading"
elif evidence_count >= 2:
    verdict_text = "MARGINALLY MEANINGFUL — Some evidence supports the Q3 filter, but not conclusive"
    recommendation = "CONDITIONAL — Consider Q3 filter with ongoing monitoring"
else:
    verdict_text = "LIKELY NOISE — Insufficient statistical evidence for Q3 filter benefit"
    recommendation = "NOT RECOMMENDED — Q3 filter primarily reduces trade count without clear benefit"

report += f"""{verdict_text}

═══════════════════════════════════════════════════════════════════════
SECTION 5: FINAL ASSESSMENT
═══════════════════════════════════════════════════════════════════════

5.1 VERDICT
────────────────────────────────────────────────────────────────────────

  Q3 Filter Assessment:     {verdict_text}
  
  Evidence Score:           {evidence_count}/4 statistical tests passed
  
  Recommendation:           {recommendation}

5.2 KEY FINDINGS
────────────────────────────────────────────────────────────────────────

  1. Trade Frequency:       Q3 filter removes {q3_skipped:,} trades ({q3_skipped/bm['total_trades']*100:.1f}%)
  2. Win Rate Impact:       {qm['win_rate'] - bm['win_rate']:+.2f}% (baseline {bm['win_rate']:.2f}% → Q3 {qm['win_rate']:.2f}%)
  3. P&L Impact:            {qm['total_pnl'] - bm['total_pnl']:+,.2f} change in total P&L
  4. Risk Impact:           Max DD {qm['max_dd_pct'] - bm['max_dd_pct']:+.1f}% (baseline {bm['max_dd_pct']:.1f}% → Q3 {qm['max_dd_pct']:.1f}%)
  5. Risk-Adjusted:         Sharpe {qm['sharpe'] - bm['sharpe']:+.2f} (baseline {bm['sharpe']:.2f} → Q3 {qm['sharpe']:.2f})

5.3 ASSUMPTIONS & LIMITATIONS
────────────────────────────────────────────────────────────────────────

  • Q3 definition uses candle open timestamp (minute 30-44)
  • Analysis assumes identical execution quality across all time blocks
  • Historical patterns may not persist in future
  • Removal of trades reduces sample size, which may affect significance
  • Market microstructure at specific minute marks may vary
  • This is a single-dimension filter — interaction effects not tested

5.4 RECOMMENDATION
────────────────────────────────────────────────────────────────────────

  {recommendation}
  
  {'If implementing: Apply Q3 filter as a simple timestamp check before trade execution.' if evidence_count >= 2 else 'Consider investigating other time-based filters or accepting the full baseline signal set.'}
  {'Monitor live performance for at least 30 days to confirm backtest alignment.' if evidence_count >= 2 else ''}

═══════════════════════════════════════════════════════════════════════
DISCLAIMER
═══════════════════════════════════════════════════════════════════════

Past performance does not guarantee future results. This analysis is
based on historical backtest data and assumes perfect execution without
slippage, latency, or market impact. Live trading results may differ.

═══════════════════════════════════════════════════════════════════════
END OF Q3 FILTER ANALYSIS REPORT
═══════════════════════════════════════════════════════════════════════
"""

with open(os.path.join(OUTPUT_DIR, "Q3_FILTER_SUMMARY.txt"), "w", encoding='utf-8') as f:
    f.write(report)

elapsed = time.time() - t0

print(f"\n{'═'*70}")
print(f"  Q3 FILTER ANALYSIS COMPLETE")
print(f"{'═'*70}")
print(f"  Output Directory:   {OUTPUT_DIR}")
print(f"  Duration:           {elapsed:.1f}s")
print(f"  Files Generated:    8")
print(f"")
print(f"  Baseline:           {bm['total_trades']:,} trades | WR: {bm['win_rate']:.2f}% | PnL: ${bm['total_pnl']:,.2f}")
print(f"  Q3-Filtered:        {qm['total_trades']:,} trades | WR: {qm['win_rate']:.2f}% | PnL: ${qm['total_pnl']:,.2f}")
print(f"  Trades Removed:     {q3_skipped:,} ({q3_skipped/bm['total_trades']*100:.1f}%)")
print(f"  Verdict:            {verdict_text}")
print(f"{'═'*70}")
