import os
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import requests
import time
import yfinance as yf  # we import it once here


# --- Auth & API config from secrets or env ---
APP_USERNAME = st.secrets.get("APP_USERNAME", os.getenv("APP_USERNAME", "hari"))
APP_PASSWORD = st.secrets.get("APP_PASSWORD", os.getenv("APP_PASSWORD", "mysecret"))
ALPHAVANTAGE_API_KEY_DEFAULT = st.secrets.get(
    "ALPHAVANTAGE_API_KEY",
    os.getenv("ALPHAVANTAGE_API_KEY", "XYW7EDBAM10QN35M")
)

def login_gate():
    # set_page_config MUST be called before any other UI, but only once
    if "page_config_set" not in st.session_state:
        st.set_page_config(page_title="Dip Strategy Dashboard", layout="wide")
        st.session_state.page_config_set = True

    # Initialize auth flag if missing
    if "auth" not in st.session_state:
        st.session_state.auth = False

    # If already authenticated, show only a Logout button in sidebar and return
    if st.session_state.auth:
        with st.sidebar:
            if st.button("🔐 Logout"):
                st.session_state.clear()
                st.experimental_rerun()
        return  # ✅ do NOT show login form when already logged in

    # 🔒 Not authenticated → show login screen
    st.title("📉📈 Dip Strategy Trading Dashboard – Login")

    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    login_btn = st.button("Login")

    if login_btn:
        if user == APP_USERNAME and pwd == APP_PASSWORD:
            st.session_state.auth = True
            st.experimental_rerun()
        else:
            st.error("Invalid username or password.")

    # Block the rest of the app if still not logged in
    if not st.session_state.auth:
        st.stop()


# 🔒 Call login gate BEFORE any other UI or config that draws things on the page
login_gate()

# From here down is your existing app: config, portfolio, tabs, etc.

# =========================
# CONFIG
# =========================
# Default starting universe (only used if tickers.csv not found)
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO", "ADBE",
    "BRK-B", "JNJ", "PG", "PEP", "COST", "HD", "WMT", "V", "MA", "MCD", "XOM", "JPM"
]

TICKERS_FILE = "tickers.csv"

LOOKBACK_DAYS_UNIVERSE = 260   # days of data for signals
LOOKBACK_DAYS_POSITIONS = 30   # days of data for pricing open positions
BACKTEST_START_YEAR = 2010

RSI_PERIOD = 14
EMA_PERIOD = 200
HIGH_LOOKBACK = 21

# Default strategy params
DIP_THRESHOLD_DEFAULT = -7.0     # % below 21-day high
RSI_BUY_MAX_DEFAULT = 40.0       # RSI must be <= this to buy
EMA_TREND_FILTER_DEFAULT = True  # require price > EMA200 for BUY

# Capital / sizing defaults
TOTAL_CAPITAL_DEFAULT = 50_000.0
PER_TRADE_PCT_DEFAULT = 5.0
MAX_POSITIONS_DEFAULT = 6

OPEN_POSITIONS_FILE = "open_positions.csv"
CLOSED_POSITIONS_FILE = "closed_positions.csv"

# ---- Alpha Vantage config ----
ALPHAVANTAGE_API_KEY_DEFAULT = "XYW7EDBAM10QN35M"
ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co/query"
ALPHA_DAILY_CACHE: Dict[str, pd.DataFrame] = {}  # per-run cache


# =========================
# UNIVERSE (TICKERS) HELPERS
# =========================

def load_tickers() -> List[str]:
    """Load tickers from tickers.csv, or initialize with DEFAULT_TICKERS."""
    try:
        df = pd.read_csv(TICKERS_FILE)
        if "ticker" in df.columns:
            tickers = [str(t).strip().upper() for t in df["ticker"].tolist() if str(t).strip()]
            if tickers:
                return tickers
    except FileNotFoundError:
        pass

    # fallback to default
    tickers = DEFAULT_TICKERS.copy()
    save_tickers(tickers)
    return tickers


def save_tickers(tickers: List[str]) -> None:
    """Save tickers to tickers.csv."""
    unique_sorted = sorted(set([t.strip().upper() for t in tickers if str(t).strip()]))
    df = pd.DataFrame({"ticker": unique_sorted})
    df.to_csv(TICKERS_FILE, index=False)


# =========================
# INDICATORS
# =========================

def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_ema(series: pd.Series, span: int = EMA_PERIOD) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]

    if "close" not in df.columns:
        candidates = [c for c in df.columns if "close" in c]
        if not candidates:
            raise ValueError(f"No close column in: {df.columns}")
        close_col = candidates[0]
    else:
        close_col = "close"

    close = pd.to_numeric(df[close_col], errors="coerce")
    df["close"] = close

    df["rsi"] = compute_rsi(close, RSI_PERIOD)
    df["ema200"] = compute_ema(close, EMA_PERIOD)
    high21 = close.rolling(window=HIGH_LOOKBACK).max()
    df["21d_high"] = high21
    df["pct_from_high"] = (close - high21) / high21 * 100.0
    return df


# =========================
# YAHOO FINANCE HELPERS
# =========================

def yf_get_daily(symbol: str, lookback_days: int) -> pd.DataFrame:
    if yf is None:
        raise ImportError("yfinance is not installed.")
    end = dt.date.today()
    start = end - dt.timedelta(days=lookback_days)
    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No Yahoo data for {symbol}")
    return df


def yf_get_daily_from_year(symbol: str, start_year: int) -> pd.DataFrame:
    if yf is None:
        raise ImportError("yfinance is not installed.")
    start = dt.date(start_year, 1, 1)
    end = dt.date.today()
    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No Yahoo data for {symbol}")
    return df


def load_universe_data_yf(tickers: List[str], lookback_days: int) -> Dict[str, pd.DataFrame]:
    data: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            df_raw = yf_get_daily(t, lookback_days)
            df = prepare_indicators(df_raw)
            data[t] = df
        except Exception as e:
            st.warning(f"Error loading {t} from Yahoo: {e}")
    return data


def fetch_last_price_yf(symbol: str) -> float:
    df = yf_get_daily(symbol, LOOKBACK_DAYS_POSITIONS)
    close_col = "Close" if "Close" in df.columns else df.columns[-1]
    return float(df[close_col].iloc[-1])


# =========================
# ALPHA VANTAGE HELPERS
# =========================

def alpha_get_daily(symbol: str, api_key: str) -> pd.DataFrame:
    cache_key = f"{symbol}:{api_key}"
    if cache_key in ALPHA_DAILY_CACHE:
        return ALPHA_DAILY_CACHE[cache_key]

    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "outputsize": "compact",
        "apikey": api_key,
    }
    resp = requests.get(ALPHAVANTAGE_BASE_URL, params=params)
    data = resp.json()

    ts = data.get("Time Series (Daily)")
    if ts is None:
        raise ValueError(
            f"Alpha Vantage error for {symbol}: "
            f"{data.get('Note') or data.get('Error Message') or 'Unknown error'}"
        )

    df = pd.DataFrame.from_dict(ts, orient="index", dtype=float)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    df = df.rename(columns={
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close",
        "5. adjusted close": "adj_close",
        "6. volume": "volume",
        "7. dividend amount": "dividend",
        "8. split coefficient": "split",
    })

    if "close" not in df.columns:
        if "adj_close" in df.columns:
            df["close"] = df["adj_close"]
        else:
            raise ValueError(f"No 'close' column found for {symbol} in Alpha Vantage data.")

    ALPHA_DAILY_CACHE[cache_key] = df
    time.sleep(5)  # free tier: ~1 call per 5 seconds
    return df


# =========================
# ENTRY SIGNALS
# =========================

@dataclass
class Signal:
    ticker: str
    date: dt.date
    close: float
    pct_from_high: float
    rsi: float
    ema200: float
    action: str   # "BUY", "WATCH", "-"


def generate_signals(
    data: Dict[str, pd.DataFrame],
    dip_threshold: float,
    rsi_buy_max: float,
    ema_trend_filter: bool
) -> List[Signal]:
    signals: List[Signal] = []
    today = dt.date.today()

    for t, df in data.items():
        row = df.iloc[-1]
        close = float(row["close"])
        pct_from_high = float(row["pct_from_high"]) if not np.isnan(row["pct_from_high"]) else np.nan
        rsi = float(row["rsi"]) if not np.isnan(row["rsi"]) else np.nan
        ema200 = float(row["ema200"]) if not np.isnan(row["ema200"]) else np.nan

        action = "-"

        if (
            not np.isnan(pct_from_high) and pct_from_high <= dip_threshold and
            not np.isnan(rsi) and rsi <= rsi_buy_max
        ):
            trend_ok = True
            if ema_trend_filter and not np.isnan(ema200):
                trend_ok = close > ema200

            if trend_ok:
                action = "BUY"
            else:
                action = "WATCH"

        signals.append(Signal(
            ticker=t,
            date=today,
            close=close,
            pct_from_high=pct_from_high,
            rsi=rsi,
            ema200=ema200,
            action=action,
        ))

    return signals


# =========================
# OPEN POSITIONS & EVAL (YAHOO)
# =========================

@dataclass
class PositionStatus:
    ticker: str
    entry_date: dt.date
    days_held: int
    entry_price: float
    shares: float
    last_price: float
    pnl_pct: float
    target_price: float
    stop_price: float
    take_profit_pct: float
    stop_loss_pct: float
    max_hold_days: int
    action: str    # "SELL" or "HOLD"
    reason: str
    notes: str


def load_open_positions(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=[
            "ticker", "entry_date", "entry_price", "shares",
            "take_profit_pct", "stop_loss_pct", "max_hold_days", "notes"
        ])
        df.to_csv(path, index=False)
        return df

    if df.empty:
        return df

    df.columns = [c.strip().lower() for c in df.columns]
    if "notes" not in df.columns:
        df["notes"] = ""
    return df


def save_open_positions(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


def load_closed_positions(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=[
            "ticker", "entry_date", "exit_date", "entry_price",
            "exit_price", "shares", "pnl_pct", "take_profit_pct",
            "stop_loss_pct", "max_hold_days", "reason", "notes"
        ])
        df.to_csv(path, index=False)
        return df

    return df


def evaluate_positions(df: pd.DataFrame) -> List[PositionStatus]:
    today = dt.date.today()
    results: List[PositionStatus] = []

    if df.empty:
        return results

    df.columns = [c.strip().lower() for c in df.columns]

    for _, row in df.iterrows():
        t = str(row["ticker"]).strip().upper()
        entry_date = pd.to_datetime(row["entry_date"]).date()
        entry_price = float(row["entry_price"])
        shares = float(row["shares"])
        tp_pct = float(row["take_profit_pct"])
        sl_pct = float(row["stop_loss_pct"])
        max_hold = int(row["max_hold_days"])
        notes = str(row.get("notes", ""))

        days_held = (today - entry_date).days
        last_price = fetch_last_price_yf(t)
        pnl_pct = (last_price / entry_price - 1.0) * 100.0

        target_price = entry_price * (1.0 + tp_pct / 100.0)
        stop_price = entry_price * (1.0 - sl_pct / 100.0)

        action = "HOLD"
        reason = "Within normal range"

        if last_price >= target_price:
            action = "SELL"
            reason = f"Target hit (+{tp_pct:.1f}% or more)"
        elif last_price <= stop_price:
            action = "SELL"
            reason = f"Stop loss hit ({-sl_pct:.1f}% or worse)"
        elif days_held > max_hold:
            action = "SELL"
            reason = f"Max hold days exceeded ({days_held} > {max_hold})"

        status = PositionStatus(
            ticker=t,
            entry_date=entry_date,
            days_held=days_held,
            entry_price=entry_price,
            shares=shares,
            last_price=last_price,
            pnl_pct=pnl_pct,
            target_price=target_price,
            stop_price=stop_price,
            take_profit_pct=tp_pct,
            stop_loss_pct=sl_pct,
            max_hold_days=max_hold,
            action=action,
            reason=reason,
            notes=notes,
        )
        results.append(status)

    return results


def append_closed_positions(status_list: List[PositionStatus], csv_file: str) -> None:
    sell_positions = [s for s in status_list if s.action == "SELL"]
    if not sell_positions:
        return

    rows = []
    today = dt.date.today()
    for s in sell_positions:
        rows.append({
            "ticker": s.ticker,
            "entry_date": s.entry_date,
            "exit_date": today,
            "entry_price": s.entry_price,
            "exit_price": s.last_price,
            "shares": s.shares,
            "pnl_pct": s.pnl_pct,
            "take_profit_pct": s.take_profit_pct,
            "stop_loss_pct": s.stop_loss_pct,
            "max_hold_days": s.max_hold_days,
            "reason": s.reason,
            "notes": s.notes,
        })

    df_new = pd.DataFrame(rows)
    try:
        existing = pd.read_csv(csv_file)
        df_combined = pd.concat([existing, df_new], ignore_index=True)
    except FileNotFoundError:
        df_combined = df_new

    df_combined.to_csv(csv_file, index=False)


def portfolio_summary(total_capital: float) -> Dict[str, float]:
    open_df = load_open_positions(OPEN_POSITIONS_FILE)
    closed_df = load_closed_positions(CLOSED_POSITIONS_FILE)

    summary = {
        "total_invested": 0.0,
        "current_value": 0.0,
        "open_pnl": 0.0,
        "open_pnl_pct_capital": 0.0,
        "open_risk": 0.0,
        "realized_pnl": 0.0,
        "realized_pnl_pct_capital": 0.0,
    }

    # Open positions
    if not open_df.empty:
        status_list = evaluate_positions(open_df)
        total_invested = sum(s.entry_price * s.shares for s in status_list)
        current_value = sum(s.last_price * s.shares for s in status_list)
        open_pnl = current_value - total_invested

        open_risk = sum(s.entry_price * s.shares * (s.stop_loss_pct / 100.0) for s in status_list)

        summary["total_invested"] = total_invested
        summary["current_value"] = current_value
        summary["open_pnl"] = open_pnl
        summary["open_pnl_pct_capital"] = (open_pnl / total_capital * 100.0) if total_capital > 0 else 0.0
        summary["open_risk"] = open_risk

    # Realized PnL
    if not closed_df.empty:
        closed_df.columns = [c.strip().lower() for c in closed_df.columns]
        if "shares" in closed_df.columns:
            closed_df["entry_price"] = pd.to_numeric(closed_df["entry_price"], errors="coerce")
            closed_df["exit_price"] = pd.to_numeric(closed_df["exit_price"], errors="coerce")
            closed_df["shares"] = pd.to_numeric(closed_df["shares"], errors="coerce")
            realized_pnl = ((closed_df["exit_price"] - closed_df["entry_price"]) * closed_df["shares"]).sum()
        else:
            realized_pnl = 0.0

        summary["realized_pnl"] = realized_pnl
        summary["realized_pnl_pct_capital"] = (realized_pnl / total_capital * 100.0) if total_capital > 0 else 0.0

    return summary


# =========================
# SIMPLE BACKTEST (YAHOO)
# =========================

@dataclass
class SimpleTrade:
    ticker: str
    entry_date: dt.date
    exit_date: dt.date
    entry_price: float
    exit_price: float

    @property
    def return_pct(self) -> float:
        return (self.exit_price / self.entry_price - 1.0) * 100.0

    @property
    def holding_days(self) -> int:
        return (self.exit_date - self.entry_date).days


def run_simple_backtest(
    tickers: List[str],
    dip_threshold: float,
    rsi_buy_max: float,
    ema_trend_filter: bool,
    tp_pct: float = 12.0,
    sl_pct: float = 7.0,
    max_hold_days: int = 60,
) -> Dict[str, float]:
    trades: List[SimpleTrade] = []

    for t in tickers:
        try:
            df_raw = yf_get_daily_from_year(t, BACKTEST_START_YEAR)
            df = prepare_indicators(df_raw)
        except Exception:
            continue

        in_trade = False
        entry_price = None
        entry_date = None
        hold_counter = 0

        for current_date, row in df.iterrows():
            price = float(row["close"])
            rsi = row["rsi"]
            ema200 = row["ema200"]
            pct_from_high = row["pct_from_high"]

            if not in_trade:
                if (
                    not np.isnan(pct_from_high) and pct_from_high <= dip_threshold and
                    not np.isnan(rsi) and rsi <= rsi_buy_max
                ):
                    trend_ok = True
                    if ema_trend_filter and not np.isnan(ema200):
                        trend_ok = price > ema200
                    if trend_ok:
                        in_trade = True
                        entry_price = price
                        entry_date = current_date.date()
                        hold_counter = 0
            else:
                hold_counter += 1
                target_price = entry_price * (1.0 + tp_pct / 100.0)
                stop_price = entry_price * (1.0 - sl_pct / 100.0)

                exit_flag = False
                if price >= target_price or price <= stop_price or hold_counter > max_hold_days:
                    exit_flag = True

                if exit_flag:
                    trades.append(
                        SimpleTrade(
                            ticker=t,
                            entry_date=entry_date,
                            exit_date=current_date.date(),
                            entry_price=entry_price,
                            exit_price=price,
                        )
                    )
                    in_trade = False
                    entry_price = None
                    entry_date = None

    if not trades:
        return {}

    capital = 1.0
    returns = []
    for tr in trades:
        r = tr.return_pct / 100.0
        capital *= (1.0 + r)
        returns.append(tr.return_pct)

    first_date = min(tr.entry_date for tr in trades)
    last_date = max(tr.exit_date for tr in trades)
    num_years = (last_date - first_date).days / 365.25 if first_date and last_date else 0.0

    total_return_pct = (capital - 1.0) * 100.0
    cagr = ((capital ** (1.0 / num_years)) - 1.0) * 100.0 if num_years > 0 else np.nan

    daily_returns = [r / max(tr.holding_days, 1) / 100.0 for r, tr in zip(returns, trades)]
    if daily_returns:
        mean_daily = np.mean(daily_returns)
        std_daily = np.std(daily_returns)
        sharpe = (mean_daily / std_daily) * np.sqrt(252) if std_daily != 0 else np.nan
    else:
        sharpe = np.nan

    win_rate = sum(1 for r in returns if r > 0) / len(returns) * 100.0
    avg_trade_return = float(np.mean(returns))
    avg_hold_days = float(np.mean([tr.holding_days for tr in trades]))

    metrics = {
        "Total Return % (approx)": total_return_pct,
        "CAGR % (approx)": cagr,
        "Sharpe Ratio (approx)": sharpe,
        "Number of Trades": len(trades),
        "Win Rate %": win_rate,
        "Avg Trade Return %": avg_trade_return,
        "Avg Holding Days": avg_hold_days,
    }
    return metrics


# =========================
# STREAMLIT UI
# =========================

st.set_page_config(page_title="Dip Strategy Dashboard", layout="wide")
st.title("📉📈 Dip Strategy Trading Dashboard")

st.caption("Dynamic universe · Dip-buy strategy · Yahoo for daily signals + on-demand Alpha Vantage comparison")

# ---- Load current universe (tickers) ----
tickers = load_tickers()

# Sidebar: strategy + capital + AV key + ticker management + usage note
st.sidebar.header("Strategy Settings")
dip_threshold = st.sidebar.number_input("Dip threshold (% from 21D high)", value=DIP_THRESHOLD_DEFAULT, step=0.5)
rsi_buy_max = st.sidebar.number_input("Max RSI to Buy", value=RSI_BUY_MAX_DEFAULT, step=1.0)
ema_trend_filter = st.sidebar.checkbox("Require price > EMA200 for BUY", value=EMA_TREND_FILTER_DEFAULT)

st.sidebar.markdown("---")
st.sidebar.header("Capital & Position Sizing")
total_capital = st.sidebar.number_input("Total Trading Capital ($)", value=TOTAL_CAPITAL_DEFAULT, step=1000.0)
per_trade_pct = st.sidebar.number_input("Per-Trade Allocation (%)", value=PER_TRADE_PCT_DEFAULT, step=0.5)
max_positions = st.sidebar.number_input("Max Concurrent Positions", value=MAX_POSITIONS_DEFAULT, step=1)

st.sidebar.markdown("---")
st.sidebar.header("Universe (Tickers)")
st.sidebar.write("**Current tickers:**")
st.sidebar.caption(", ".join(tickers))

new_ticker = st.sidebar.text_input("Add ticker (e.g. NFLX)", value="")
if st.sidebar.button("➕ Add Ticker"):
    nt = new_ticker.strip().upper()
    if not nt:
        st.sidebar.error("Please enter a ticker.")
    elif nt in tickers:
        st.sidebar.warning(f"{nt} is already in the universe.")
    else:
        tickers.append(nt)
        save_tickers(tickers)
        st.sidebar.success(f"Added {nt} to universe. (App will re-run with it.)")

remove_selection = st.sidebar.multiselect("Remove tickers", options=tickers)
if st.sidebar.button("🗑 Remove selected tickers"):
    if not remove_selection:
        st.sidebar.warning("Select at least one ticker to remove.")
    else:
        tickers = [t for t in tickers if t not in remove_selection]
        save_tickers(tickers)
        st.sidebar.success("Removed selected tickers from universe. (App will re-run with updated list.)")

st.sidebar.markdown("---")
st.sidebar.header("Alpha Vantage API")
user_api_key = st.sidebar.text_input(
    "API Key (optional override)",
    value="",
    type="password",
    help="Leave empty to use the built-in default key. For production, use your own key here."
)
if user_api_key.strip():
    api_key_in_use = user_api_key.strip()
else:
    api_key_in_use = ALPHAVANTAGE_API_KEY_DEFAULT

st.sidebar.markdown("**Alpha Vantage Free Tier Notes:**")
st.sidebar.markdown(
    "- ~25 API calls per day (free tier)\n"
    "- ~1 request every 5 seconds\n"
    "- We **only** call Alpha Vantage when you explicitly request comparison.\n"
    "- Regular signals & monitoring use Yahoo (no API limits)."
)

st.sidebar.markdown("---")
st.sidebar.write("Files used:")
st.sidebar.code(OPEN_POSITIONS_FILE)
st.sidebar.code(CLOSED_POSITIONS_FILE)
st.sidebar.code(TICKERS_FILE)


# ---------- PORTFOLIO SUMMARY ----------
summary = portfolio_summary(total_capital)

st.subheader("📊 Portfolio Summary")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric(
        "Total Invested (Open)",
        f"${summary['total_invested']:.0f}",
        help="Sum of entry value (price × shares) of all open positions."
    )
    cash_suggested = max(total_capital - summary["total_invested"], 0.0)
    st.metric("Suggested Cash (Unallocated)", f"${cash_suggested:.0f}")
with c2:
    st.metric(
        "Open PnL ($)",
        f"${summary['open_pnl']:.0f}",
        f"{summary['open_pnl_pct_capital']:.2f}%",
        help="Unrealized profit/loss based on Yahoo latest prices."
    )
with c3:
    st.metric(
        "Realized PnL ($)",
        f"${summary['realized_pnl']:.0f}",
        f"{summary['realized_pnl_pct_capital']:.2f}%",
        help="Realized profit/loss from closed_positions.csv"
    )
with c4:
    st.metric(
        "Total Open Risk ($)",
        f"${summary['open_risk']:.0f}",
        help="Approx: entry_value × stop_loss_pct summed across open trades."
    )

tab_signals, tab_open, tab_monitor = st.tabs(["🔍 Signals", "📒 Open Positions", "📊 Monitor / Close"])


# ------------- TAB 1: SIGNALS (YAHOO + CAPITAL + AV COMPARE + BACKTEST) -------------
with tab_signals:
    st.subheader("Entry Signals (Yahoo Finance · 15-min delayed)")

    df_sig = None
    if st.button("🔄 Refresh Signals (Yahoo)"):
        try:
            universe_data = load_universe_data_yf(tickers, LOOKBACK_DAYS_UNIVERSE)
            signals = generate_signals(universe_data, dip_threshold, rsi_buy_max, ema_trend_filter)
            if not signals:
                st.info("No signals generated.")
            else:
                rows = []
                per_trade_value = total_capital * (per_trade_pct / 100.0)
                for s in signals:
                    pos_value = per_trade_value if s.action == "BUY" else 0.0
                    approx_shares = np.floor(pos_value / s.close) if (s.action == "BUY" and s.close > 0) else 0
                    rows.append({
                        "Ticker": s.ticker,
                        "Price (Yahoo)": s.close,
                        "% From 21D High": s.pct_from_high,
                        "RSI": s.rsi,
                        "EMA200": s.ema200,
                        "Signal": s.action,
                        "Pos Size ($)": pos_value if s.action == "BUY" else np.nan,
                        "Approx Shares": approx_shares if s.action == "BUY" else np.nan,
                    })
                df_sig = pd.DataFrame(rows)

                def highlight_signal(row):
                    if row["Signal"] == "BUY":
                        return ["background-color: #c8e6c9"] * len(row)
                    elif row["Signal"] == "WATCH":
                        return ["background-color: #fff9c4"] * len(row)
                    return [""] * len(row)

                st.dataframe(
                    df_sig.style.apply(highlight_signal, axis=1).format({
                        "Price (Yahoo)": "{:.2f}",
                        "% From 21D High": "{:.2f}",
                        "RSI": "{:.1f}",
                        "EMA200": "{:.2f}",
                        "Pos Size ($)": "{:.2f}",
                        "Approx Shares": "{:.0f}",
                    }),
                    use_container_width=True,
                    height=500,
                )

                buy_df = df_sig[df_sig["Signal"] == "BUY"]
                if not buy_df.empty:
                    st.markdown("**✅ BUY opportunities today (with suggested sizing):**")
                    for _, row in buy_df.iterrows():
                        st.write(
                            f"- **{row['Ticker']}** @ {row['Price (Yahoo)']:.2f} "
                            f"({row['% From 21D High']:.2f}% from 21D high, RSI={row['RSI']:.1f}) → "
                            f"~${row['Pos Size ($)']:.0f} ≈ {int(row['Approx Shares'])} shares"
                        )
                else:
                    st.info("No BUY signals under current rules.")
        except Exception as e:
            st.error(f"Error while generating signals: {e}")
    else:
        st.info("Click **Refresh Signals (Yahoo)** to scan the universe.")

    # Alpha Vantage comparison (ALL TICKERS)
    st.markdown("---")
    st.subheader("On-Demand Alpha Vantage Price Comparison (ALL Tickers)")
    st.caption(
        "This calls Alpha Vantage once per ticker. "
        "On free tier this can take time because of rate limits."
    )

    if df_sig is not None and not df_sig.empty:
        if st.button("🔎 Fetch Alpha Vantage prices for ALL tickers in universe"):
            compare_rows = []
            for t in df_sig["Ticker"].tolist():
                try:
                    df_av = alpha_get_daily(t, api_key_in_use)
                    av_last_close = float(df_av["close"].iloc[-1])
                    av_last_date = df_av.index[-1].date()

                    yahoo_price = float(df_sig.loc[df_sig["Ticker"] == t, "Price (Yahoo)"].iloc[0])
                    diff_pct = (av_last_close / yahoo_price - 1.0) * 100.0 if yahoo_price else np.nan

                    compare_rows.append({
                        "Ticker": t,
                        "Yahoo Price": yahoo_price,
                        "Alpha Vantage Close": av_last_close,
                        "AV Last Date": av_last_date,
                        "Diff % (AV vs Yahoo)": diff_pct,
                    })
                except Exception as e:
                    st.warning(f"Alpha Vantage error for {t}: {e}")

            if compare_rows:
                df_compare = pd.DataFrame(compare_rows)
                st.dataframe(
                    df_compare.style.format({
                        "Yahoo Price": "{:.2f}",
                        "Alpha Vantage Close": "{:.2f}",
                        "Diff % (AV vs Yahoo)": "{:.2f}",
                    }),
                    use_container_width=True,
                )
            else:
                st.info("No comparison data available.")
    else:
        st.info("Run signals first to enable Alpha Vantage comparison.")

    # Backtest summary snippet
    st.markdown("---")
    st.subheader("Historical Backtest Summary (Yahoo · simplified)")
    st.caption(
        "Runs a simplified historical backtest using the current universe and dip logic "
        "(1 share per trade, approximate compounding)."
    )

    if st.button("🔁 Run Historical Backtest Summary (Yahoo)"):
        with st.spinner("Running backtest across current universe..."):
            try:
                metrics = run_simple_backtest(
                    tickers,
                    dip_threshold,
                    rsi_buy_max,
                    ema_trend_filter,
                    tp_pct=12.0,
                    sl_pct=7.0,
                    max_hold_days=60,
                )
                if not metrics:
                    st.warning("Backtest produced no trades. Try loosening the rules.")
                else:
                    for k, v in metrics.items():
                        if isinstance(v, float):
                            st.write(f"**{k}**: {v:,.2f}")
                        else:
                            st.write(f"**{k}**: {v}")
            except Exception as e:
                st.error(f"Backtest error: {e}")
    else:
        st.info("Click the button to run a one-click historical backtest summary.")


# ------------- TAB 2: OPEN POSITIONS -------------
with tab_open:
    st.subheader("Open Positions (You Maintain These)")

    df_open = load_open_positions(OPEN_POSITIONS_FILE)

    st.markdown("### Current Open Positions")
    if df_open.empty:
        st.info("No open positions yet. Add a new trade below after you place it with your broker.")
    else:
        st.dataframe(df_open, use_container_width=True)

    st.markdown("### Add New Position")

    with st.form("add_position_form"):
        # ticker list includes current universe + OTHER
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            ticker_input = st.selectbox("Ticker", options=tickers + ["OTHER"])
            if ticker_input == "OTHER":
                ticker_input = st.text_input("Custom Ticker").upper()
        with col2:
            entry_date_input = st.date_input("Entry Date", value=dt.date.today())
        with col3:
            entry_price_input = st.number_input("Entry Price", min_value=0.0, step=0.01, format="%.2f")
        with col4:
            shares_input = st.number_input("Shares", min_value=0.0, step=1.0)

        col5, col6, col7 = st.columns(3)
        with col5:
            tp_pct_input = st.number_input("Take Profit %", value=12.0, step=0.5)
        with col6:
            sl_pct_input = st.number_input("Stop Loss %", value=7.0, step=0.5)
        with col7:
            max_hold_input = st.number_input("Max Hold Days", value=60, step=1)

        notes_input = st.text_input("Notes", value="", help="Optional tag (e.g. 'first trade', 'earnings avoided', etc.)")

        submitted = st.form_submit_button("➕ Add Position")

        if submitted:
            if not ticker_input or entry_price_input <= 0 or shares_input <= 0:
                st.error("Please fill in at least ticker, entry price, and shares.")
            else:
                new_row = {
                    "ticker": ticker_input,
                    "entry_date": entry_date_input,
                    "entry_price": entry_price_input,
                    "shares": shares_input,
                    "take_profit_pct": tp_pct_input,
                    "stop_loss_pct": sl_pct_input,
                    "max_hold_days": max_hold_input,
                    "notes": notes_input,
                }
                df_open = pd.concat([df_open, pd.DataFrame([new_row])], ignore_index=True)
                save_open_positions(df_open, OPEN_POSITIONS_FILE)
                st.success(f"Added position: {ticker_input} ({shares_input} @ {entry_price_input:.2f})")


# ------------- TAB 3: MONITOR / CLOSE (YAHOO) -------------
with tab_monitor:
    st.subheader("Monitor Open Positions (HOLD / SELL · Yahoo prices)")

    df_open = load_open_positions(OPEN_POSITIONS_FILE)
    if df_open.empty:
        st.info("No open positions to monitor. Add some in the 'Open Positions' tab after you take trades.")
    else:
        if st.button("🔍 Evaluate Positions (Yahoo)"):
            try:
                status_list = evaluate_positions(df_open)
                if not status_list:
                    st.info("No positions to evaluate.")
                else:
                    df_status = pd.DataFrame([{
                        "Ticker": s.ticker,
                        "Entry Date": s.entry_date,
                        "Days Held": s.days_held,
                        "Entry Price": s.entry_price,
                        "Shares": s.shares,
                        "Last Price (Yahoo)": s.last_price,
                        "PnL %": s.pnl_pct,
                        "Target": s.target_price,
                        "Stop": s.stop_price,
                        "Action": s.action,
                        "Reason": s.reason,
                        "Notes": s.notes,
                    } for s in status_list])

                    def highlight_action(row):
                        if row["Action"] == "SELL":
                            return ["background-color: #ffcdd2"] * len(row)
                        return [""] * len(row)

                    st.dataframe(
                        df_status.style.apply(highlight_action, axis=1).format({
                            "Entry Price": "{:.2f}",
                            "Last Price (Yahoo)": "{:.2f}",
                            "PnL %": "{:.2f}",
                            "Target": "{:.2f}",
                            "Stop": "{:.2f}",
                        }),
                        use_container_width=True,
                        height=500,
                    )

                    sells = [s for s in status_list if s.action == "SELL"]
                    if sells:
                        st.markdown("### Positions flagged as **SELL** under current rules:")
                        for s in sells:
                            st.write(
                                f"- **{s.ticker}**: {s.reason} (PnL {s.pnl_pct:.2f}%, "
                                f"Entry {s.entry_price:.2f} → Last {s.last_price:.2f}, Shares {s.shares:.0f})"
                            )

                        if st.button("✅ Apply SELL actions (log & update files)"):
                            append_closed_positions(status_list, CLOSED_POSITIONS_FILE)
                            hold_mask = [s.action == "HOLD" for s in status_list]
                            df_hold = df_open.iloc[hold_mask].copy()
                            save_open_positions(df_hold, OPEN_POSITIONS_FILE)
                            st.success(
                                f"Applied SELL actions. "
                                f"{len(sells)} closed → logged to {CLOSED_POSITIONS_FILE}. "
                                f"{len(df_hold)} positions remain open."
                            )
                    else:
                        st.success("No positions to SELL today under current rules.")
            except Exception as e:
                st.error(f"Error evaluating positions: {e}")
        else:
            st.info("Click **Evaluate Positions (Yahoo)** to check HOLD/SELL suggestions.")
