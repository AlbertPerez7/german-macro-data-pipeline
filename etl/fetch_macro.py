from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List
import xml.etree.ElementTree as ET

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

EUROSTAT_BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
ECB_BASE = "https://data-api.ecb.europa.eu/service/data"

HEADERS = {
    "User-Agent": "german-market-macro-powerbi (student project)",
    "Accept": "*/*",
}


# -------------------------
# Eurostat (robust parser)
# -------------------------
def eurostat_get(dataset: str, params: Dict[str, str]) -> dict:
    url = f"{EUROSTAT_BASE}/{dataset}"
    r = requests.get(url, params=params, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()


def eurostat_extract_time_series(payload: dict, value_col: str) -> pd.DataFrame:
    """
    Robust Eurostat parser:
    - Supports obs keys like "0:1:2:3" (colon-separated indices)
    - Supports flattened obs keys like "123" (single integer index into the N-D cube)

    Returns DF with columns: time_period, <value_col>
    """
    values = payload.get("value", {})
    if not values:
        return pd.DataFrame(columns=["time_period", value_col])

    dim_ids: List[str] = payload.get("id", [])
    dim_sizes: List[int] = payload.get("size", [])
    if "time" not in dim_ids:
        raise RuntimeError(f"Eurostat payload has no 'time' dimension. Dimensions: {dim_ids}")

    time_pos = dim_ids.index("time")

    # Map time index -> label
    time_dim = payload["dimension"]["time"]
    label_to_idx = time_dim["category"]["index"]  # label -> idx
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    def unflatten(flat_idx: int) -> List[int]:
        """Convert flattened index to N-D indices using dim_sizes (row-major)."""
        idxs = []
        rem = flat_idx
        for size in dim_sizes:
            idxs.append(rem % size)
            rem //= size
        return idxs  # same order as dim_ids

    rows = []
    for obs_key, v in values.items():
        key_str = str(obs_key)

        # Case A: colon-separated indices "a:b:c"
        if ":" in key_str:
            parts = key_str.split(":")
            if time_pos >= len(parts):
                continue
            t_idx = int(parts[time_pos])

        # Case B: flattened index "123"
        else:
            try:
                flat = int(key_str)
            except ValueError:
                continue
            if not dim_sizes or time_pos >= len(dim_sizes):
                continue
            idxs = unflatten(flat)
            t_idx = idxs[time_pos]

        label = idx_to_label.get(t_idx)
        if label is None:
            continue
        rows.append((label, float(v)))

    df = pd.DataFrame(rows, columns=["time_period", value_col])
    if df.empty:
        return df

    # If multiple series still exist, average them by month
    df = df.groupby("time_period", as_index=False)[value_col].mean()
    return df.sort_values("time_period")



def fetch_inflation_hicp_de(start_ym: str) -> pd.DataFrame:
    # HICP annual rate of change (% YoY), monthly, all-items, Germany
    payload = eurostat_get(
        "PRC_HICP_MANR",
        {
            "freq": "M",
            "unit": "RCH_A",
            "coicop": "CP00",
            "geo": "DE",
            "sinceTimePeriod": start_ym,
        },
    )
    return eurostat_extract_time_series(payload, "inflation_yoy_pct")


def fetch_unemployment_de(start_ym: str) -> pd.DataFrame:
    payload = eurostat_get(
        "UNE_RT_M",
        {
            "freq": "M",
            "unit": "PC_ACT",
            "sex": "T",
            "age": "TOTAL",   # <-- CANVI IMPORTANT
            "s_adj": "SA",
            "geo": "DE",
            "sinceTimePeriod": start_ym,
        },
    )
    return eurostat_extract_time_series(payload, "unemployment_rate_pct")



# -------------------------
# ECB (XML SDMX) -> monthly rate
# -------------------------
def ecb_fetch_deposit_facility_rate_daily(start_date: str) -> pd.DataFrame:
    """
    ECB Deposit Facility Rate (DFR), daily series.
    We request SDMX-ML (XML) and parse it (robust vs JSON issues).
    """
    # Flow FM, series key for DFR (daily)
    # If ECB changes minor formatting, this still often works:
    key = "D.U2.EUR.4F.KR.DFR.LEV"
    url = f"{ECB_BASE}/FM/{key}"
    params = {
        "startPeriod": start_date,
        "format": "sdmx-ml",  # XML
    }
    r = requests.get(url, params=params, headers=HEADERS, timeout=60)
    r.raise_for_status()

    # Parse SDMX GenericData XML:
    # We look for Obs elements and read:
    #  - ObsDimension/@value (date)
    #  - ObsValue/@value (rate)
    root = ET.fromstring(r.text)

    ns = {
        "message": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/message",
        "generic": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic",
    }

    rows = []
    for obs in root.findall(".//generic:Obs", ns):
        dim = obs.find("generic:ObsDimension", ns)
        val = obs.find("generic:ObsValue", ns)
        if dim is None or val is None:
            continue
        d = dim.attrib.get("value")
        v = val.attrib.get("value")
        if d is None or v is None:
            continue
        rows.append((d, float(v)))

    df = pd.DataFrame(rows, columns=["date", "ecb_deposit_rate_pct"])
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    return df


def to_monthly_last(df_daily: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if df_daily.empty:
        return pd.DataFrame(columns=["time_period", value_col])

    d = df_daily.copy()
    d["month"] = d["date"].dt.to_period("M")
    monthly = (
        d.sort_values("date")
        .groupby("month", as_index=False)
        .tail(1)[["month", value_col]]
        .sort_values("month")
    )
    monthly["time_period"] = monthly["month"].astype(str)  # "YYYY-MM"
    return monthly[["time_period", value_col]]


# -------------------------
# Build final macro monthly
# -------------------------
def build_macro_monthly(start_ym: str) -> pd.DataFrame:
    infl = fetch_inflation_hicp_de(start_ym)
    unemp = fetch_unemployment_de(start_ym)

    dfr_daily = ecb_fetch_deposit_facility_rate_daily(start_date=f"{start_ym}-01")
    rate = to_monthly_last(dfr_daily, "ecb_deposit_rate_pct")

    df = infl.merge(unemp, on="time_period", how="outer").merge(rate, on="time_period", how="outer")
    df = df.sort_values("time_period")

    df["date"] = pd.to_datetime(df["time_period"] + "-01", errors="coerce")
    df = df.dropna(subset=["date"])

    start_date = pd.to_datetime(start_ym + "-01")
    df = df[df["date"] >= start_date]

    df["year"] = df["date"].dt.year
    df["month_num"] = df["date"].dt.month

    return df[
        ["date", "year", "month_num", "inflation_yoy_pct", "unemployment_rate_pct", "ecb_deposit_rate_pct"]
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch Germany monthly macro (Eurostat + ECB XML) -> ../data/macro_de.csv")
    p.add_argument("--start", default="2000-01", help="Start month YYYY-MM (default: 2000-01)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print(f"Fetching monthly macro data from {args.start} ...")
    df = build_macro_monthly(start_ym=args.start)

    out_path = DATA_DIR / "macro_de.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Saved → {out_path}")

    # Quick sanity prints
    print("Non-null counts:")
    print(df[["inflation_yoy_pct", "unemployment_rate_pct", "ecb_deposit_rate_pct"]].notna().sum())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
