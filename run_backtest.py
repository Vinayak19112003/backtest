"""
EXACT-MATCH BACKTEST
RSI 43, 58 + Block if (ADX > 25 & ATR > 80th)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.data_loader import load_data

# ── BOT CONFIG ──
FEE_RATE = 0.01          
INITIAL_CAPITAL = 100.0  
RSI_OVERSOLD = 43
RSI_OVERBOUGHT = 58
VOL_THRESHOLD = 0.05     
SIM_ENTRY_PRICE = 0.50   

# Weekend boost
ENABLE_WEEKEND_BOOST = True
WEEKEND_MULTIPLIER = 1.0
MAX_WEEKEND_POSITION_PCT = 0.50

STRATEGY_NAME = f"rsi_{RSI_OVERSOLD}_{RSI_OVERBOUGHT}_adx25_atr80_baseline"
OUTPUT_DIR = f"results/{STRATEGY_NAME}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Config: FEE={FEE_RATE*100}%, RISK=$1 Fixed, Cap=${INITIAL_CAPITAL}")
print(f"RSI Thresholds: {RSI_OVERSOLD} / {RSI_OVERBOUGHT}")
print(f"Filters: Block trade if ADX > 25 & ATR > 80th")
print()

# ── LOAD DATA ──
print("Loading data...")
df = load_data('data/BTCUSDT_15m_3_years.csv')

# ── INDICATORS ──
print("Computing indicators...")

# RSI(14)
delta = df['close'].diff()
gain = (delta.where(delta>0,0)).ewm(alpha=1/14, adjust=False).mean()
loss_s = (-delta.where(delta<0,0)).ewm(alpha=1/14, adjust=False).mean()
df['rsi'] = 100 - (100/(1 + gain/loss_s))

# ATR
tr = pd.concat([df['high']-df['low'], (df['high']-df['close'].shift(1)).abs(), (df['low']-df['close'].shift(1)).abs()], axis=1).max(axis=1)
df['atr_14'] = tr.rolling(14).mean()

# ADX(14)
up_m, dn_m = df['high'].diff(), -df['low'].diff()
plus_dm = pd.Series(np.where((up_m>dn_m)&(up_m>0), up_m, 0.0), index=df.index)
minus_dm = pd.Series(np.where((dn_m>up_m)&(dn_m>0), dn_m, 0.0), index=df.index)
atr_s = tr.ewm(alpha=1/14, adjust=False).mean()
plus_di = 100*(plus_dm.ewm(alpha=1/14, adjust=False).mean()/atr_s)
minus_di = 100*(minus_dm.ewm(alpha=1/14, adjust=False).mean()/atr_s)
di_sum = (plus_di+minus_di).replace(0, np.nan)
dx = (abs(plus_di-minus_di)/di_sum)*100
df['adx_14'] = dx.ewm(alpha=1/14, adjust=False).mean()

# "adx > 25"
df['regime_trend'] = np.where(df['adx_14']>25, 'Strong', 'Weak')

# Rolling ATR percentile
atr_ser = df['atr_14']
rv = []
for i in range(len(df)):
    if i<28 or pd.isna(atr_ser.iloc[i]): rv.append('Medium'); continue
    lb = min(96, i)
    rec = atr_ser.iloc[max(0,i-lb):i+1].dropna()
    if len(rec)<5: rv.append('Medium'); continue
    p = (rec<atr_ser.iloc[i]).sum()/len(rec)*100
    # "atr > 80th"
    rv.append('High' if p>=80 else 'Medium')
df['regime_vol'] = rv

# 5% Volatility filter
log_ret = np.log(df['close'] / df['close'].shift(1))
df['vol_96'] = log_ret.rolling(96).std() * np.sqrt(96)

df = df.dropna(subset=['rsi', 'adx_14', 'atr_14']).copy()

# ── GENERATE TRADES ──
print("Generating trades...")
trades = []
capital = INITIAL_CAPITAL

for i in range(1, len(df) - 1):
    sc = df.iloc[i]
    oc = df.iloc[i + 1]
    
    rsi = sc['rsi']
    if pd.isna(rsi): continue
    
    # 1. SMARTFILTER BLOCK:
    if sc['regime_vol'] == 'High' and sc['regime_trend'] == 'Strong':
        continue
    
    # 2. VOLATILITY FILTER
    vol = sc.get('vol_96', 0)
    if not pd.isna(vol) and vol > VOL_THRESHOLD:
        continue
    
    # 3. RSI SIGNAL
    buy_yes = rsi < RSI_OVERSOLD
    buy_no = rsi > RSI_OVERBOUGHT
    if not (buy_yes or buy_no): continue
    
    signal = 'YES' if buy_yes else 'NO'
    
    # 4. CONFIDENCE CHECK
    confidence = 50
    hour = sc.name.hour
    day = sc.name.weekday()
    if signal == 'YES':
        if rsi < 40: confidence += 20
        elif rsi < 43: confidence += 10
    else:
        if rsi > 60: confidence += 20
        elif rsi > 58: confidence += 10
    if hour in [19, 20, 21, 22]: confidence += 10
    if day in [5, 6]: confidence += 15
    if confidence < 60:
        continue
    
    # 5. SIZING ($1 Fixed Risk)
    limit_price = SIM_ENTRY_PRICE
    
    base_target_cost = 1.0  
    target_cost = base_target_cost
    
    if ENABLE_WEEKEND_BOOST and day in [5, 6]:
        target_cost = base_target_cost * WEEKEND_MULTIPLIER
        max_weekend_cost = capital * MAX_WEEKEND_POSITION_PCT
        target_cost = min(target_cost, max_weekend_cost)
    
    target_shares = int(target_cost / limit_price)
    shares = max(1, target_shares)  
    
    bet_amount = shares * limit_price
    if bet_amount > capital: continue
    
    # 6. SETTLEMENT
    entry_price_btc = sc['close']
    settle_price_btc = oc['close']
    
    market_result = 'YES' if settle_price_btc > entry_price_btc else 'NO'
    won = (signal == market_result)
    
    fees = bet_amount * FEE_RATE * 2
    pnl = (shares * 1.0) - bet_amount - fees if won else -bet_amount - fees
    
    capital += pnl
    
    trades.append({
        'timestamp': sc.name, 'date': str(sc.name.date()), 'time': str(sc.name.time())[:5],
        'rsi': round(rsi,1), 'adx': round(sc['adx_14'],1),
        'signal': signal, 'shares': shares, 'entry_price': limit_price,
        'bet_amount': round(bet_amount,2), 'fees': round(fees,4),
        'regime': f"{sc['regime_vol']}/{sc['regime_trend']}",
        'btc_entry': round(entry_price_btc,2), 'btc_exit': round(settle_price_btc,2),
        'win': won, 'pnl': round(pnl,4), 'capital': round(capital,2),
        'confidence': confidence, 'hour': hour, 'day_name': sc.name.day_name()
    })

tdf = pd.DataFrame(trades)
tdf['timestamp'] = pd.to_datetime(tdf['timestamp'])

if len(tdf) == 0:
    print("No trades!"); exit()

# ── METRICS ──
total = len(tdf); wins = int(tdf['win'].sum()); losses = total-wins
wr = wins/total*100; total_pnl = tdf['pnl'].sum()
avg_pnl = tdf['pnl'].mean()
avg_win = tdf[tdf['win']]['pnl'].mean()
avg_loss = tdf[~tdf['win']]['pnl'].mean()
gp = tdf[tdf['win']]['pnl'].sum(); gl = abs(tdf[~tdf['win']]['pnl'].sum())
pf = gp/gl if gl>0 else float('inf')
cum = tdf['pnl'].cumsum()
max_dd = (cum - cum.cummax()).min()
max_dd_pct = max_dd / INITIAL_CAPITAL * 100

pm = len(tdf.groupby(tdf['timestamp'].dt.to_period('M')).apply(lambda x: x['pnl'].sum() > 0))
tm = len(tdf.groupby(tdf['timestamp'].dt.to_period('M')))

# Daily/Monthly breakdown
tdf['idate'] = tdf['timestamp'].dt.date
daily = tdf.groupby('idate').agg(Trades=('pnl','count'), Wins=('win','sum'), DPnL=('pnl','sum')).reset_index()
daily['WR'] = daily['Wins']/daily['Trades']*100

tdf['mperiod'] = tdf['timestamp'].dt.to_period('M').astype(str)
monthly = tdf.groupby('mperiod').agg(Trades=('pnl','count'), Wins=('win','sum'), MPnL=('pnl','sum')).reset_index()
monthly['WR'] = monthly['Wins']/monthly['Trades']*100

# Daily Drawdown Duration, Depth & Losing Days
daily['Capital'] = INITIAL_CAPITAL + daily['DPnL'].cumsum()
daily['HighWater'] = daily['Capital'].cummax()
dd_durations, dd_usd_list, dd_pct_list = [], [], []
dd_details = [] 
curr_dd, curr_dd_usd, curr_dd_pct = 0, 0, 0
dd_start_date = None

for i in range(len(daily)):
    cap, hw = daily.loc[i, 'Capital'], daily.loc[i, 'HighWater']
    
    # We are in a drawdown
    if cap < hw:
        if curr_dd == 0:
            dd_start_date = daily.loc[i, 'idate']
        curr_dd += 1
        curr_dd_usd = max(curr_dd_usd, hw - cap)
        curr_dd_pct = max(curr_dd_pct, (hw - cap) / hw * 100)
    
    # We hit a new all-time high (recovery)
    else:
        if curr_dd > 0:
            dd_durations.append(curr_dd)
            dd_usd_list.append(curr_dd_usd)
            dd_pct_list.append(curr_dd_pct)
            dd_details.append({
                'start_date': str(dd_start_date),
                'recovery_date': str(daily.loc[i, 'idate']),
                'duration_days': curr_dd,
                'max_usd': curr_dd_usd,
                'max_pct': curr_dd_pct
            })
        curr_dd, curr_dd_usd, curr_dd_pct = 0, 0, 0

max_dd_days = max(dd_durations) if dd_durations else 0

loss_day_streaks, curr_loss = [], 0
for dpnl in daily['DPnL']:
    if dpnl < 0: curr_loss += 1
    else:
        if curr_loss > 0: loss_day_streaks.append(curr_loss)
        curr_loss = 0
if curr_loss > 0: loss_day_streaks.append(curr_loss)
max_losing_days = max(loss_day_streaks) if loss_day_streaks else 0


# ── BUILD REPORT ──
L = []
L.append('='*90)
L.append(f'  EXACT LIVE BOT BACKTEST - RSI {RSI_OVERSOLD}/{RSI_OVERBOUGHT} - $1 FIXED RISK')
L.append(f'  ** BLOCKING IF ADX > 25 & ATR > 80% (Baseline Model) **')
L.append('='*90)

L.append('')
L.append('-'*90)
L.append('  1. CORE PERFORMANCE')
L.append('-'*90)
L.append(f'  Total Trades        : {total:,}')
total_days = (tdf['timestamp'].max() - tdf['timestamp'].min()).days
L.append(f'  Trades Per Day      : {total/total_days if total_days>0 else 0:.2f}')
L.append(f'  Wins / Losses       : {wins:,} / {losses:,}')
L.append(f'  Win Rate            : {wr:.2f}%')
L.append(f'  Total PnL           : ${total_pnl:.2f}')
L.append(f'  Final Capital       : ${capital:.2f}')
L.append(f'  Total Return        : {total_pnl/INITIAL_CAPITAL*100:.2f}%')
L.append(f'  Avg PnL Per Trade   : ${avg_pnl:.4f}')
L.append(f'  Avg Win             : +${avg_win:.4f}')
L.append(f'  Avg Loss            : ${avg_loss:.4f}')
L.append(f'  Profit Factor       : {pf:.3f}')
L.append(f'  Max Drawdown        : ${max_dd:.2f} ({max_dd_pct:.1f}%)')
L.append(f'  Max DD Duration     : {max_dd_days} Days')
L.append(f'  Max Losing Days     : {max_losing_days} Days')
L.append(f'  Profitable Months   : {pm}/{tm} ({pm/tm*100:.1f}%)')

# Hourly
L.append('')
L.append('-'*90)
L.append('  2. HOURLY PERFORMANCE')
L.append('-'*90)
hourly = tdf.groupby('hour').agg(Trades=('pnl','count'), Wins=('win','sum'), HPnL=('pnl','sum')).reset_index()
hourly['WR'] = hourly['Wins']/hourly['Trades']*100
L.append(f'    Hour |  Trades | Win Rate |    PnL')
L.append(f'  ------+--------+---------+--------')
for _, r in hourly.iterrows():
    L.append(f'  {int(r.hour):02d}:00 | {r.Trades:>6,.0f} |   {r.WR:.1f}% | ${r.HPnL:>8.2f}')

# Monthly
L.append('')
L.append('-'*90)
L.append('  3. MONTHLY BREAKDOWN')
L.append('-'*90)
L.append(f'      Month |  Trades |    WR |        PnL |    Capital | Max DD Days')
L.append(f'  ---------+--------+------+-----------+----------+-------------')
rc = INITIAL_CAPITAL
for _, r in monthly.iterrows():
    rc += r.MPnL
    L.append(f'    {r.mperiod:>7} | {r.Trades:>6,.0f} | {r.WR:>4.0f}% | ${r.MPnL:>9.2f} | ${rc:>9.2f}')

# Regime
L.append('')
L.append('-'*90)
L.append('  4. REGIME PERFORMANCE')
L.append('-'*90)
rg = tdf.groupby('regime').agg(Trades=('pnl','count'), Wins=('win','sum'), RPnL=('pnl','sum')).reset_index()
rg['WR'] = rg['Wins']/rg['Trades']*100
for _, r in rg.sort_values('RPnL', ascending=False).iterrows():
    L.append(f'  {r.regime:>15}: {r.Trades:>6,} trades | WR: {r.WR:.1f}% | PnL: ${r.RPnL:>9.2f}')

L.append('')
L.append('='*90)
L.append('  END OF REPORT')
L.append('='*90)

report = '\n'.join(L)
out = f'{OUTPUT_DIR}/deep_analysis.txt'
with open(out, 'w', encoding='utf-8') as f:
    f.write(report)

tdf.to_csv(f'{OUTPUT_DIR}/trades.csv', index=False)

print(report)
print(f'\nSaved detailed metrics to: {out}')
