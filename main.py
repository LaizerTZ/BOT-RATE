"""FX Live Data Automation: fetch Bank of Tanzania rates, store, alert, chart, report and email.

Run once with:  python main.py
"""
import matplotlib

matplotlib.use("Agg")  # draw charts without a screen, so it also works from the scheduler
import json
import logging
import os
import sqlite3
from pathlib import Path
from dotenv import load_dotenv
import sys
import requests, pandas, matplotlib, seaborn, dotenv, schedule
import requests
from datetime import datetime
from datetime import date, timedelta
import pandas as pd
from datetime import datetime, timezone
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image as PILImage
import base64
from html import escape
import smtplib
from email.message import EmailMessage
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                Table, TableStyle, Image, KeepTogether)

PROJECT = Path(__file__).resolve().parent  # the folder that holds main.py
load_dotenv(PROJECT / ".env", override=True)

with open(PROJECT / "config.json", encoding="utf-8") as f:
    CONFIG = json.load(f)

BASE_URL = CONFIG["api"]["base_url"]
PROVIDER = CONFIG["api"]["provider"]
TIMEOUT = CONFIG["api"]["timeout_seconds"]
PAIRS = [(p["base"], p["quote"]) for p in CONFIG["pairs"]]
HISTORY_DAYS = CONFIG["history_days"]
DB_PATH = PROJECT / CONFIG["files"]["database"]
RAW_DIR = PROJECT / CONFIG["folders"]["raw"]
LOG_PATH = PROJECT / CONFIG["files"]["log"]
SHORT_WINDOW = CONFIG["alerts"]["short_ma_window"]
LONG_WINDOW = CONFIG["alerts"]["long_ma_window"]
DAILY_LIMIT = CONFIG["alerts"]["daily_change_percent"]
CHARTS_DIR = PROJECT / CONFIG["folders"]["charts"]
HIGHLIGHT = "#a40000"
PAIR_COLORS = {"USD/TZS": "#a40000", "EUR/TZS": "#1c6fb8", "GBP/TZS": "#2f9e8f"}
INK, MUTED, GRID = "#222222", "#666666", "#e6e6e6"
DPI = 150
BACKFILL_DAYS = 190
HISTORY_START = date.fromisoformat(CONFIG["history_start_date"])
GOOD, BAD = "#16a368", "#d6303f"
REPORTS_DIR = PROJECT / CONFIG["folders"]["reports"]
CHART_FILES = [
    ("07_usd_tzs_trend.png",         "USD/TZS daily trend, last 30 days"),
    ("08_usd_tzs_year_trend.png",    "USD/TZS daily trend, 2026 to date"),
    ("01_usd_tzs_line.png",          "All pairs, last 30 days"),
    ("02_pct_change_bar.png",        "Percentage change by pair"),
    ("05_monthly_average.png",       "USD/TZS monthly average"),
    ("06_current_month_daily.png",   "Current month, daily rate and change"),
    ("03_correlation_heatmap.png",   "Correlation between pairs"),
    ("04_daily_change_histogram.png", "Distribution of daily changes"),
]

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("fx_automation")
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(file_handler)

DB_PATH.parent.mkdir(parents=True, exist_ok=True)
conn = sqlite3.connect(DB_PATH)


def fetch_latest(base, quote):
    """Fetch the latest BOTA rate for one pair, with retries. Returns a dict, or None."""
    response = request_with_retry(f"{BASE_URL}/rate/{base}/{quote}", {"providers": PROVIDER})
    if response is None:
        return None
    try:
        data = response.json()
        if data.get("rate") is None:
            logger.warning(f"No rate in answer for {base}/{quote}: {data}")
            return None
        return {
            "date": datetime.strptime(data.get("date"), "%Y-%m-%d").date(),
            "base": data.get("base"),
            "quote": data.get("quote"),
            "rate": float(data.get("rate")),
        }
    except (ValueError, TypeError) as error:
        logger.error(f"Bad data for {base}/{quote}: {error}")
        return None


def fetch_history(base, quote, lookback_days=45):
    """Fetch daily BOTA rates for one pair, with retries. Returns a list of records, or None."""
    today = date.today()
    params = {
        "from": (today - timedelta(days=lookback_days)).isoformat(),
        "to": today.isoformat(),
        "base": base,
        "quotes": quote,
        "providers": PROVIDER,
    }
    response = request_with_retry(f"{BASE_URL}/rates", params)
    if response is None:
        return None
    try:
        records = response.json()
    except ValueError as error:
        logger.error(f"Bad history data for {base}/{quote}: {error}")
        return None
    if not isinstance(records, list) or len(records) == 0:
        logger.warning(f"No history returned for {base}/{quote}: {records}")
        return None
    logger.info(f"Fetched {len(records)} records for {base}/{quote}")
    return records


def fetch_data():
    """Fetch the latest 30 daily BOTA rates for every pair, save the raw JSON, and fill days
    with no published rate using the previous day's rate. Returns one pandas table with
    columns: date, base, quote, rate, is_filled."""
    frames = []
    for base, quote in PAIRS:
        records = fetch_history(base, quote)
        if records is None:
            continue
        save_raw_json(records, base, quote)
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df["rate"] = df["rate"].astype(float)
        df = fill_missing_days(df).tail(HISTORY_DAYS)
        filled = int(df["is_filled"].sum())
        if filled:
            logger.info(f"Filled {filled} day(s) with the previous rate for {base}/{quote}")
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["date", "base", "quote", "rate", "is_filled"])
    return pd.concat(frames, ignore_index=True)


def save_raw_json(records, base, quote):
    """Save the API answer for one pair as a JSON file in Data/raw. Returns the file path."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = RAW_DIR / f"{base}_{quote}_{stamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    return path


def save_data(df, conn):
    """Insert the rows of a rates table into fx_rates with a UTC timestamp. An existing row is
    kept, except a filled row (is_filled = 1), which is replaced when the real rate arrives.
    Returns the number of rows added or updated."""
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = [
        (row.date.strftime("%Y-%m-%d"), row.base, row.quote, float(row.rate), fetched_at, int(row.is_filled))
        for row in df.itertuples(index=False)
    ]
    before = conn.total_changes
    conn.executemany(
        """INSERT INTO fx_rates (date, base, quote, rate, fetched_at_utc, is_filled)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT (date, base, quote) DO UPDATE SET
               rate = excluded.rate,
               fetched_at_utc = excluded.fetched_at_utc,
               is_filled = excluded.is_filled
           WHERE fx_rates.is_filled = 1 AND excluded.is_filled = 0""",
        rows,
    )
    conn.commit()
    return conn.total_changes - before


def request_with_retry(url, params=None):
    """GET a URL with a timeout. On timeouts, network errors, HTTP 429 or 5xx it waits
    (1s, 2s, 4s ... or the Retry-After header) and tries again, up to max_attempts.
    Returns the response, or None if it gave up or the request itself was wrong."""
    max_attempts = CONFIG["retry"]["max_attempts"]
    base_delay = CONFIG["retry"]["base_delay_seconds"]

    for attempt in range(1, max_attempts + 1):
        wait = base_delay * 2 ** (attempt - 1)
        try:
            response = requests.get(url, params=params, timeout=TIMEOUT)
            if response.status_code == 429 or response.status_code >= 500:
                retry_after = response.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    wait = int(retry_after)
                problem = f"HTTP {response.status_code}"
            else:
                response.raise_for_status()
                return response
        except requests.exceptions.HTTPError as error:
            logger.error(f"Request rejected, not retrying: {error}")
            return None
        except requests.exceptions.RequestException as error:
            problem = type(error).__name__

        if attempt == max_attempts:
            logger.error(f"Giving up after {max_attempts} attempts: {problem}")
            return None
        logger.warning(f"Attempt {attempt} failed ({problem}); waiting {wait}s before retry")
        time.sleep(wait)
    return None


def run_once():
    logger.info("Run started")
    df = fetch_data()
    if df.empty:
        logger.error("Run finished with no data")
        return []
    pairs_fetched = df[["base", "quote"]].drop_duplicates().shape[0]
    rows_changed = save_data(df, conn)
    rates = load_rates()
    alerts = check_alerts(rates, last_n_days=1)
    try:
        make_charts(rates)
    except Exception as error:
        logger.error(f"Charts failed: {type(error).__name__}: {error}")

    email_sent = False
    try:
        summary = build_summary(conn)
        files = []
        for builder in (build_pdf_report, build_report):
            try:
                files.append(builder(summary, alerts))
            except Exception as error:
                logger.error(f"{builder.__name__} failed: {type(error).__name__}: {error}")
        if files:
            email_sent = send_report_email(files, summary, alerts)
    except Exception as error:
        logger.error(f"Report step failed: {type(error).__name__}: {error}")

    logger.info(f"Run finished: pairs fetched {pairs_fetched}/{len(PAIRS)}, rows received {len(df)}, rows added or updated {rows_changed}, alerts {len(alerts)}, email sent {email_sent}")
    return alerts


def load_rates():
    """Read the clean rates from the database, add moving averages. Returns a pandas table."""
    df = pd.read_sql_query(
        "SELECT date, base, quote, rate, is_repeat, change_pct FROM fx_rates_clean ORDER BY base, quote, date",
        conn,
    )
    df["date"] = pd.to_datetime(df["date"])
    df["pair"] = df["base"] + "/" + df["quote"]
    grouped = df.groupby("pair")["rate"]
    df["ma_short"] = grouped.transform(lambda s: s.rolling(SHORT_WINDOW).mean())
    df["ma_long"] = grouped.transform(lambda s: s.rolling(LONG_WINDOW).mean())
    return df


def check_alerts(df, last_n_days=1):
    """Check the newest rows against two rules: a daily change above the limit, and a
    5-day / 20-day moving-average crossover. Prints and logs each alert. Returns a list of alerts."""
    alerts = []
    for pair, g in df.groupby("pair"):
        g = g.sort_values("date").reset_index(drop=True)
        g["gap"] = g["ma_short"] - g["ma_long"]
        g["prev_gap"] = g["gap"].shift(1)

        for _, row in g.tail(last_n_days).iterrows():
            day = row["date"].strftime("%Y-%m-%d")

            if pd.notna(row["change_pct"]) and abs(row["change_pct"]) > DAILY_LIMIT:
                alerts.append({
                    "date": day, "pair": pair, "rule": "Daily change",
                    "message": f"{pair} moved {row['change_pct']:+.4f}% on {day} (limit {DAILY_LIMIT}%)",
                })

            if pd.notna(row["gap"]) and pd.notna(row["prev_gap"]) and row["gap"] * row["prev_gap"] < 0:
                direction = "above" if row["gap"] > 0 else "below"
                alerts.append({
                    "date": day, "pair": pair, "rule": "Moving-average crossover",
                    "message": f"{pair} {SHORT_WINDOW}-day average crossed {direction} the "
                               f"{LONG_WINDOW}-day average on {day}",
                })

    for alert in alerts:
        print("ALERT:", alert["message"])
        logger.warning(f"ALERT | {alert['rule']} | {alert['message']}")
    return alerts


def fill_missing_days(df):
    """Make sure every calendar day up to today has a rate. A day with no published rate
    gets the previous day's rate and is marked is_filled = 1."""
    df = df.sort_values("date").drop_duplicates("date")
    all_days = pd.date_range(df["date"].min(), pd.Timestamp(date.today()), freq="D")
    df = df.set_index("date").reindex(all_days)
    df["is_filled"] = df["rate"].isna().astype(int)
    df[["base", "quote", "rate"]] = df[["base", "quote", "rate"]].ffill()
    df.index.name = "date"
    return df.reset_index()


def show_today(label):
    print(label)
    print(pd.read_sql_query(
        f"SELECT base, rate, is_filled FROM fx_rates WHERE date = '{today_str}' ORDER BY base", test_conn))


def add_footnote(fig, df):
    """Write the data source and the time of the last update at the bottom of a chart."""
    last_rate = df["date"].max().strftime("%d %b %Y")
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    fig.text(0.01, 0.005,
             f"Source: Bank of Tanzania (BOTA) via Frankfurter API. Latest rate date: {last_rate}. "
             f"Chart updated: {updated}.",
             fontsize=8, color=MUTED, ha="left", va="bottom")


def save_chart(fig, name):
    """Save a figure into the charts folder at 150 dpi and close it. Returns the file path."""
    path = CHARTS_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def make_line_chart(df):
    """Line chart of the USD/TZS rate with its 5-day moving average. Returns the saved file path."""
    g = df[df["pair"] == "USD/TZS"].sort_values("date")
    first, last = g.iloc[0], g.iloc[-1]
    change = (last["rate"] - first["rate"]) / first["rate"] * 100
    direction = "down" if change < 0 else "up"
    title = (f"The dollar is {direction} {abs(change):.1f}% against the shilling "
             f"since {first['date'].strftime('%d %b')}")

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(g["date"], g["rate"], color=HIGHLIGHT, linewidth=2, label="USD/TZS daily rate")
    ax.plot(g["date"], g["ma_short"], color=INK, linewidth=2, linestyle="--",
            label=f"{SHORT_WINDOW}-day moving average")
    ax.annotate(f"{last['rate']:,.2f}", (last["date"], last["rate"]),
                textcoords="offset points", xytext=(6, 0), color=INK, fontsize=9, va="center")

    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK)
    ax.set_ylabel("TZS per 1 USD")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(loc="lower left", frameon=False)
    fig.subplots_adjust(bottom=0.14)
    add_footnote(fig, df)
    return save_chart(fig, "01_usd_tzs_line.png")


def make_bar_chart(df):
    """Bar chart of the percentage change of each pair since the first day. Returns the saved file path."""
    rows = []
    for pair, g in df.groupby("pair"):
        g = g.sort_values("date")
        change = (g["rate"].iloc[-1] - g["rate"].iloc[0]) / g["rate"].iloc[0] * 100
        rows.append({"pair": pair, "change": change})
    bars = pd.DataFrame(rows).sort_values("change")

    start = df["date"].min().strftime("%d %b")
    top = bars.loc[bars["change"].abs().idxmax()]
    title = f"{top['pair']} moved the most: {top['change']:+.1f}% since {start}"

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(bars["pair"], bars["change"], color=[PAIR_COLORS[p] for p in bars["pair"]], width=0.55)
    ax.axhline(0, color=MUTED, linewidth=1)
    for i, value in enumerate(bars["change"]):
        ax.annotate(f"{value:+.2f}%", (i, value), textcoords="offset points",
                    xytext=(0, 6 if value >= 0 else -14), ha="center", color=INK, fontsize=10)

    ax.margins(y=0.25)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK)
    ax.set_ylabel(f"Change since {start} (%)")
    fig.subplots_adjust(bottom=0.14)
    add_footnote(fig, df)
    return save_chart(fig, "02_pct_change_bar.png")


def make_heatmap(df):
    """Heatmap of the correlation between the daily changes of the pairs. Returns the saved file path."""
    changes = df.pivot(index="date", columns="pair", values="change_pct").dropna()
    corr = changes.corr()
    pairs = list(corr.columns)

    best = None
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            value = corr.iloc[i, j]
            if best is None or abs(value) > abs(best[2]):
                best = (pairs[i], pairs[j], value)
    kind = "move together" if best[2] >= 0 else "move in opposite directions"
    title = f"{best[0]} and {best[1]} {kind} the most (correlation {best[2]:.2f})"

    cmap = LinearSegmentedColormap.from_list("diverging", ["#6a4c93", "#f2f2f2", "#d9822b"])
    fig, ax = plt.subplots(figsize=(7, 5.6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, vmin=-1, vmax=1,
                linewidths=2, linecolor="white", square=True, ax=ax,
                annot_kws={"fontsize": 11, "color": INK},
                cbar_kws={"label": "Correlation of daily changes", "shrink": 0.8})
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK)
    fig.subplots_adjust(bottom=0.14)
    add_footnote(fig, df)
    return save_chart(fig, "03_correlation_heatmap.png")


def make_histogram(df):
    """Histogram of the daily percentage changes, one panel per pair. Returns the saved file path."""
    moves = df[(df["is_repeat"] == 0) & df["change_pct"].notna()]
    share = (moves["change_pct"].abs() <= DAILY_LIMIT).mean() * 100
    title = f"{share:.0f}% of daily moves stayed within ±{DAILY_LIMIT}%"

    pairs = sorted(moves["pair"].unique())
    fig, axes = plt.subplots(1, len(pairs), figsize=(11, 4.6), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, pair in zip(axes, pairs):
        sns.histplot(data=moves[moves["pair"] == pair], x="change_pct", bins=10,
                     color=PAIR_COLORS[pair], edgecolor="white", ax=ax)
        ax.axvline(0, color=MUTED, linewidth=1)
        ax.set_title(pair, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("Daily change (%)")
        ax.set_ylabel("Number of days" if ax is axes[0] else "")

    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.subplots_adjust(top=0.82, bottom=0.2)
    add_footnote(fig, df)
    return save_chart(fig, "04_daily_change_histogram.png")


def make_charts(df):
    """Create all 8 charts and save them in the charts folder. Charts 1 to 4 use the last 30 days,
    charts 5 to 8 use the longer history. Returns the list of saved file paths."""
    recent = df[df["date"] >= df["date"].max() - pd.Timedelta(days=HISTORY_DAYS - 1)]
    paths = [
        make_line_chart(recent),
        make_bar_chart(recent),
        make_heatmap(recent),
        make_histogram(recent),
        make_monthly_average_chart(df),
        make_month_daily_chart(df),
        make_trend_chart(df),
        make_trend_chart(df, start=HISTORY_START, filename="08_usd_tzs_year_trend.png"),
    ]
    logger.info(f"Charts saved: {len(paths)} files in {CHARTS_DIR.name}/")
    return paths


def make_monthly_average_chart(df):
    """Line chart of the USD/TZS monthly average rate. Partial months have hollow markers.
    Returns the saved file path."""
    d = df[df["pair"] == "USD/TZS"].copy()
    d["month"] = d["date"].dt.to_period("M")
    monthly = d.groupby("month")["rate"].mean().reset_index()
    monthly["month_start"] = monthly["month"].dt.to_timestamp()

    days_per_month = d.groupby("month")["date"].nunique()
    partial = {m for m, n in days_per_month.items() if n < m.days_in_month}

    top = monthly.loc[monthly["rate"].idxmax()]
    title = f"USD/TZS monthly average was highest in {top['month_start'].strftime('%B')} ({top['rate']:,.0f} TZS)"

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(monthly["month_start"], monthly["rate"], color=HIGHLIGHT, linewidth=2, marker="o", markersize=8)
    part = monthly[monthly["month"].isin(partial)]
    ax.plot(part["month_start"], part["rate"], linestyle="", marker="o", markersize=8,
            markerfacecolor="white", markeredgecolor=HIGHLIGHT, markeredgewidth=2)
    for _, row in monthly.iterrows():
        ax.annotate(f"{row['rate']:,.0f}", (row["month_start"], row["rate"]),
                    textcoords="offset points", xytext=(0, 10), ha="center", color=INK, fontsize=10)

    ax.margins(y=0.2)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK)
    ax.set_ylabel("Monthly average (TZS per 1 USD)")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    note = "Hollow markers: partial months (" + ", ".join(m.strftime("%b") for m in sorted(partial)) + ")."
    fig.subplots_adjust(bottom=0.2)
    fig.text(0.01, 0.045, note, fontsize=8, color=MUTED, ha="left", va="bottom")
    add_footnote(fig, df)
    return save_chart(fig, "05_monthly_average.png")


def make_month_daily_chart(df):
    """Two-panel chart for the current month: daily USD/TZS rate (top) and daily change in % (bottom).
    Returns the saved file path."""
    today = pd.Timestamp(date.today())
    month_start = today.replace(day=1)
    month_end = month_start + pd.offsets.MonthEnd(0)
    month_name = today.strftime("%B")

    g = df[(df["pair"] == "USD/TZS") & (df["date"] >= month_start) & (df["date"] <= month_end)]
    g = g.sort_values("date").reset_index(drop=True)
    moves = g["change_pct"].fillna(0)

    total = (g["rate"].iloc[-1] - g["rate"].iloc[0]) / g["rate"].iloc[0] * 100
    big = moves.abs().idxmax()
    title = (f"{month_name} so far: USD/TZS {total:+.1f}%, biggest daily move "
             f"{moves[big]:+.2f}% on {g['date'][big].strftime('%d %b')}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    ax1.plot(g["date"], g["rate"], color=HIGHLIGHT, linewidth=2, marker="o", markersize=4)
    ax1.annotate(f"{g['rate'].iloc[-1]:,.2f}", (g["date"].iloc[-1], g["rate"].iloc[-1]),
                 textcoords="offset points", xytext=(8, 0), color=INK, fontsize=9, va="center")
    ax1.set_ylabel("TZS per 1 USD")

    ax2.bar(g["date"], moves, color=HIGHLIGHT, width=0.7)
    ax2.axhline(0, color=MUTED, linewidth=1)
    ax2.set_ylabel("Daily change (%)")
    ax2.set_xlabel(f"Day of {month_name} {today.year}")
    ax2.set_xlim(month_start - pd.Timedelta(days=0.7), month_end + pd.Timedelta(days=0.7))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d"))

    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.subplots_adjust(top=0.9, bottom=0.12, hspace=0.12)
    add_footnote(fig, df)
    return save_chart(fig, "06_current_month_daily.png")


def make_trend_chart(df, start=None, filename="07_usd_tzs_trend.png", pair="USD/TZS"):
    """Daily trend chart for one pair. The area is green above the starting rate and red below it.
    Uses the last 30 days, or all days from the start date when it is given. Returns the saved file path."""
    g = df[df["pair"] == pair].sort_values("date")
    g = g[g["date"] >= pd.Timestamp(start)] if start else g.tail(HISTORY_DAYS)
    ref, last = g["rate"].iloc[0], g["rate"].iloc[-1]
    change = (last - ref) / ref * 100
    colour = GOOD if last >= ref else BAD
    where = "above" if change >= 0 else "below"
    title = (f"{pair} is {abs(change):.1f}% {where} its level of "
             f"{g['date'].iloc[0].strftime('%d %b %Y')} ({last:,.2f} TZS)")

    x = mdates.date2num(g["date"])
    xs = np.linspace(x.min(), x.max(), max(800, len(g) * 4))
    ys = np.interp(xs, x, g["rate"])
    up = ys >= ref

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(xs, ys, ref, where=up, interpolate=True, color=GOOD, alpha=0.18, linewidth=0)
    ax.fill_between(xs, ys, ref, where=~up, interpolate=True, color=BAD, alpha=0.18, linewidth=0)
    ax.plot(xs, np.where(up, ys, np.nan), color=GOOD, linewidth=2)
    ax.plot(xs, np.where(~up, ys, np.nan), color=BAD, linewidth=2)
    ax.axhline(ref, color=MUTED, linewidth=1, linestyle=":")

    ax.annotate(f"{ref:,.2f}", xy=(0, ref), xycoords=("axes fraction", "data"),
                xytext=(6, 6), textcoords="offset points", fontsize=9, color=INK,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=GRID))
    ax.annotate(f"{last:,.2f}", xy=(1, last), xycoords=("axes fraction", "data"),
                xytext=(6, 0), textcoords="offset points", va="center", ha="left",
                fontsize=10, fontweight="bold", color="white", annotation_clip=False,
                bbox=dict(boxstyle="round,pad=0.3", fc=colour, ec=colour))

    span = ys.max() - ys.min()
    ax.set_ylim(ys.min() - span * 0.15, ys.max() + span * 0.15)
    ax.set_xlim(xs.min(), xs.max())
    ax.xaxis_date()
    if len(g) > 90:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    else:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_ylabel("TZS per 1 unit of " + pair.split("/")[0])
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK)

    fig.subplots_adjust(bottom=0.18, right=0.9)
    fig.text(0.01, 0.045, "Green: rate above the starting level (dotted line). Red: below it.",
             fontsize=8, color=MUTED, ha="left", va="bottom")
    add_footnote(fig, df)
    return save_chart(fig, filename)


def build_summary(conn):
    """Return one row per pair with the latest rate, the change since the previous day,
    and the minimum, maximum and average rate of the last 30 days."""
    rows = pd.read_sql_query(
        "SELECT date, base, quote, rate, prev_rate, change_pct, is_filled FROM fx_rates_clean", conn)
    rows["pair"] = rows["base"] + "/" + rows["quote"]
    latest = rows.sort_values("date").groupby("pair").tail(1).set_index("pair")

    stats = pd.read_sql_query(
        """SELECT base || '/' || quote AS pair,
                  MIN(rate) AS min_rate, MAX(rate) AS max_rate, AVG(rate) AS avg_rate
           FROM fx_rates_clean
           WHERE date >= date((SELECT MAX(date) FROM fx_rates_clean), ?)
           GROUP BY base, quote""",
        conn, params=(f"-{HISTORY_DAYS - 1} days",),
    ).set_index("pair")

    summary = latest[["date", "rate", "prev_rate", "change_pct", "is_filled"]].join(stats)
    summary = summary.rename(columns={"rate": "latest_rate"})
    summary["change_abs"] = summary["latest_rate"] - summary["prev_rate"]
    return summary


def image_to_html(path, alt):
    """Read a PNG file and return an <img> tag with the picture embedded in the HTML itself."""
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" alt="{alt}" style="max-width:100%;height:auto;">'


def build_report(summary, alerts):
    """Build the HTML dashboard, save it in Reports/, and return its path."""
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    rows_html = ""
    for pair, r in summary.iterrows():
        note = " (carried forward)" if r["is_filled"] == 1 else ""
        rows_html += (
            f"<tr><td>{escape(pair)}</td>"
            f"<td>{r['latest_rate']:,.2f}{note}</td>"
            f"<td>{r['change_abs']:+,.2f} ({r['change_pct']:+.2f}%)</td>"
            f"<td>{r['min_rate']:,.2f}</td><td>{r['max_rate']:,.2f}</td>"
            f"<td>{r['avg_rate']:,.2f}</td><td>{escape(str(r['date']))}</td></tr>"
        )

    if alerts:
        items = "".join(f"<li>{escape(str(a))}</li>" for a in alerts)
        alerts_html = f"<ul>{items}</ul>"
    else:
        alerts_html = "<p>No alerts today.</p>"

    charts_html = ""
    for filename, alt in CHART_FILES:
        path = CHARTS_DIR / filename
        if path.exists():
            charts_html += f"<h3>{escape(alt)}</h3>{image_to_html(path, alt)}"

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FX Daily Report</title>
<style>
body{{font-family:Arial,sans-serif;color:#222;max-width:900px;margin:0 auto;padding:16px}}
h1{{color:#a40000;margin-bottom:0}} .sub{{color:#666;margin-top:4px}}
table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{border:1px solid #ddd;padding:8px;text-align:right}}
th{{background:#f3f3f3}} td:first-child,th:first-child{{text-align:left}}
</style></head><body>
<h1>FX Daily Report: Bank of Tanzania rates</h1>
<p class="sub">Source: Bank of Tanzania via Frankfurter API. Generated {now_utc}.</p>
<h2>Latest rates (min, max, average over the last {HISTORY_DAYS} days)</h2>
<table><tr><th>Pair</th><th>Latest (TZS)</th><th>Change vs previous day</th>
<th>Min</th><th>Max</th><th>Average</th><th>Rate date</th></tr>{rows_html}</table>
<h2>Alerts</h2>{alerts_html}
<h2>Charts</h2>{charts_html}
</body></html>"""

    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / "fx_report.html"
    out.write_text(html, encoding="utf-8")
    logger.info(f"Report saved: {out.name} ({out.stat().st_size // 1024} KB)")
    return out


def send_report_email(attachments, summary, alerts):
    """Email a short summary with the report files attached. Returns True if sent."""
    sender = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_APP_PASSWORD")
    receiver = os.getenv("EMAIL_RECEIVER")
    server = os.getenv("SMTP_SERVER")
    port = int(os.getenv("SMTP_PORT", "587"))

    if not all([sender, password, receiver, server]):
        logger.error("Email not sent: missing email settings in .env")
        return False

    lines = []
    for pair, r in summary.iterrows():
        note = " (carried forward)" if r["is_filled"] == 1 else ""
        lines.append(f"{pair}: {r['latest_rate']:,.2f}{note}, change {r['change_pct']:+.2f}%")
    alert_text = "\n".join(f"- {a}" for a in alerts) if alerts else "No alerts today."
    report_date = summary["date"].max()

    body = (
        f"FX Daily Report: Bank of Tanzania rates ({report_date})\n\n"
        + "\n".join(lines)
        + f"\n\nAlerts:\n{alert_text}\n\n"
        + "The PDF report is attached. The HTML dashboard is attached too (open it in a browser)."
    )

    msg = EmailMessage()
    msg["Subject"] = f"FX Daily Report {report_date}"
    msg["From"] = sender
    msg["To"] = receiver
    msg.set_content(body)

    names = []
    for path in attachments:
        path = Path(path)
        maintype, subtype = ("application", "pdf") if path.suffix == ".pdf" else ("text", "html")
        msg.add_attachment(path.read_bytes(), maintype=maintype, subtype=subtype, filename=path.name)
        names.append(path.name)

    try:
        with smtplib.SMTP(server, port, timeout=30) as smtp:
            smtp.starttls()
            smtp.login(sender, password)
            smtp.send_message(msg)
        logger.info(f"Email sent to {receiver} with {', '.join(names)} attached")
        return True
    except Exception as error:
        logger.error(f"Email failed: {type(error).__name__}: {error}")
        return False


def build_pdf_report(summary, alerts):
    """Build the PDF report in Reports/ and return its path."""
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    out = REPORTS_DIR / "fx_report.pdf"
    REPORTS_DIR.mkdir(exist_ok=True)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("title", parent=styles["Title"], alignment=0,
                                 textColor=colors.HexColor("#a40000"), fontSize=18)
    small = ParagraphStyle("small", parent=styles["Normal"], fontSize=8,
                           textColor=colors.HexColor("#666666"))

    doc = SimpleDocTemplate(str(out), pagesize=A4, title="FX Daily Report",
                            leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=15 * mm, bottomMargin=15 * mm)

    story = [
        Paragraph("FX Daily Report: Bank of Tanzania rates", title_style),
        Paragraph(f"Source: Bank of Tanzania via Frankfurter API. Generated {now_utc}.", small),
        Spacer(1, 8),
        Paragraph(f"Latest rates (min, max, average over the last {HISTORY_DAYS} days)", styles["Heading2"]),
    ]

    data = [["Pair", "Latest (TZS)", "Change vs previous day", "Min", "Max", "Average", "Rate date"]]
    has_filled = False
    for pair, r in summary.iterrows():
        mark = "*" if r["is_filled"] == 1 else ""
        has_filled = has_filled or bool(mark)
        data.append([
            pair, f"{r['latest_rate']:,.2f}{mark}",
            f"{r['change_abs']:+,.2f} ({r['change_pct']:+.2f}%)",
            f"{r['min_rate']:,.2f}", f"{r['max_rate']:,.2f}",
            f"{r['avg_rate']:,.2f}", str(r["date"]),
        ])
    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f3f3")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
    ]))
    story.append(table)
    if has_filled:
        story.append(Paragraph("* carried forward from the previous day (no new Bank of Tanzania rate yet)", small))

    story.append(Paragraph("Alerts", styles["Heading2"]))
    if alerts:
        for a in alerts:
            story.append(Paragraph(f"&bull; {escape(str(a))}", styles["Normal"]))
    else:
        story.append(Paragraph("No alerts today.", styles["Normal"]))

    story.append(Paragraph("Charts", styles["Heading2"]))
    for filename, alt in CHART_FILES:
        path = CHARTS_DIR / filename
        if path.exists():
            iw, ih = ImageReader(str(path)).getSize()
            story.append(KeepTogether([
                Paragraph(escape(alt), styles["Heading3"]),
                Image(str(path), width=doc.width, height=doc.width * ih / iw),
                Spacer(1, 10),
            ]))

    doc.build(story)
    logger.info(f"PDF report saved: {out.name} ({out.stat().st_size // 1024} KB)")
    return out


def init_database():
    """Create the table and the clean view if they do not exist yet."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fx_rates (
            date            TEXT NOT NULL,
            base            TEXT NOT NULL,
            quote           TEXT NOT NULL,
            rate            REAL NOT NULL,
            fetched_at_utc  TEXT NOT NULL,
            is_filled       INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (date, base, quote)
        )
    """)
    columns = [row[1] for row in conn.execute("PRAGMA table_info(fx_rates)")]
    if "is_filled" not in columns:
        conn.execute("ALTER TABLE fx_rates ADD COLUMN is_filled INTEGER NOT NULL DEFAULT 0")
    conn.execute("DROP VIEW IF EXISTS fx_rates_clean")
    conn.execute("""
        CREATE VIEW fx_rates_clean AS
        SELECT
            date, base, quote, rate, is_filled, fetched_at_utc,
            LAG(rate) OVER w AS prev_rate,
            CASE WHEN rate = LAG(rate) OVER w THEN 1 ELSE 0 END AS is_repeat,
            ROUND((rate - LAG(rate) OVER w) / LAG(rate) OVER w * 100, 4) AS change_pct
        FROM fx_rates
        WHERE rate > 0
        WINDOW w AS (PARTITION BY base, quote ORDER BY date)
    """)
    conn.commit()


def main():
    """Entry point: prepare folders and database, then run the whole pipeline once."""
    for folder in CONFIG["folders"].values():
        (PROJECT / folder).mkdir(parents=True, exist_ok=True)
    init_database()
    try:
        run_once()
    except Exception as error:
        logger.error(f"Run crashed: {type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
