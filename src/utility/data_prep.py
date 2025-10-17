from typing import Optional
import pandas as pd

def build_trading_features(
    df: Optional[pd.DataFrame] = None,
    source: Optional[str] = None,
    volume_col: str = "Volume USD",
    window: int = 7*24,
) -> pd.DataFrame:
    if df is None:
        if source is None:
            raise ValueError("Provide either df or source.")
        df = pd.read_csv(source, parse_dates=["date"], index_col="date")
        df.sort_index(inplace=True)
        df.dropna(how="all", inplace=True)
        df.drop_duplicates(inplace=True)

    # case-insensitive name map
    cols = {c.lower(): c for c in df.columns}
    for need in ["open", "high", "low", "close"]:
        if need not in cols:
            raise KeyError(f"Missing required column '{need}'")

    # find volume column robustly
    if volume_col not in df.columns:
        candidates = [c for c in df.columns
                      if c.lower().replace(" ", "") == volume_col.lower().replace(" ", "")]
        if not candidates:
            raise KeyError(f"Volume column '{volume_col}' not found.")
        volume_col = candidates[0]

    df["feature_close"]  = df[cols["close"]].pct_change()
    df["feature_open"]   = df[cols["open"]] / df[cols["close"]]
    df["feature_high"]   = df[cols["high"]] / df[cols["close"]]
    df["feature_low"]    = df[cols["low"]]  / df[cols["close"]]
    df["feature_volume"] = df[volume_col] / df[volume_col].rolling(window).max()

    return df.dropna().copy()
