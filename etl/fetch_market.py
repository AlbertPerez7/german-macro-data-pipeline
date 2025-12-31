from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def fetch_stooq_daily(symbol: str) -> pd.DataFrame:
    url = "https://stooq.com/q/d/l/"
    params = {"s": symbol, "i": "d"}  # daily
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    df = pd.read_csv(StringIO(r.text))
    if df.empty or "Date" not in df.columns:
        raise RuntimeError(
            f"Stooq returned unexpected data for symbol={symbol}. "
            f"Try '^dax' or '^gdaxi'."
        )

    df = df.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["ticker"] = symbol
    return df


def market_to_monthly(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Convert daily market data to monthly frequency:
    - month_end_close (last close of month)
    - monthly_return_pct
    - realized_vol_annualized (std(daily returns within month) * sqrt(252))
    """
    d = df_daily.dropna(subset=["close"]).copy()
    d["daily_return"] = d["close"].pct_change()

    d["month"] = d["date"].dt.to_period("M")

    month_end = (
        d.sort_values("date")
        .groupby("month", as_index=False)
        .tail(1)[["month", "date", "close"]]
        .rename(columns={"date": "month_end_date", "close": "month_end_close"})
    )

    vol = (
        d.groupby("month")["daily_return"]
        .std()
        .mul(252 ** 0.5)
        .reset_index()
        .rename(columns={"daily_return": "realized_vol_annualized"})
    )

    out = month_end.merge(vol, on="month", how="left").sort_values("month")

    out["monthly_return_pct"] = out["month_end_close"].pct_change() * 100.0
    out["year"] = out["month"].dt.year
    out["month_num"] = out["month"].dt.month
    out["ticker"] = df_daily["ticker"].iloc[0]

    # Monthly key for BI: YYYY-MM-01
    out["date"] = pd.to_datetime(
        out["year"].astype(str) + "-" + out["month_num"].astype(str).str.zfill(2) + "-01"
    )

    return out[
        [
            "date",
            "year",
            "month_num",
            "ticker",
            "month_end_date",
            "month_end_close",
            "monthly_return_pct",
            "realized_vol_annualized",
        ]
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch German market data from Stooq and export MONTHLY to ../data/market_de.csv"
    )
    p.add_argument("--symbol", default="^dax", help="Stooq symbol (default: ^dax). Try ^gdaxi if needed.")
    p.add_argument("--start", default="2000-01-01", help="Start date YYYY-MM-DD (default: 2000-01-01)")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD (optional)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print(f"Fetching daily market data from Stooq: {args.symbol} ...")
    daily = fetch_stooq_daily(args.symbol)

    daily = daily[daily["date"] >= pd.to_datetime(args.start)]
    if args.end:
        daily = daily[daily["date"] <= pd.to_datetime(args.end)]

    print("Aggregating to MONTHLY frequency ...")
    monthly = market_to_monthly(daily)

    out_path = DATA_DIR / "market_de.csv"

    # IMPORTANT: European CSV formatting for Power BI (Spain)
    monthly.to_csv(
        out_path,
        index=False,
        sep=";",       # delimiter ;
        decimal=",",   # decimal comma
        encoding="utf-8",
    )

    print(f"✅ Saved → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
