import pandas as pd

def build_trading_features(df: pd.DataFrame | None = None,
                           source: str | None = None,
                           volume_col: str = "Volume USD",
                           window: int = 7*24) -> pd.DataFrame:
    """
    Create trading features on an OHLCV dataframe.

    Inputs:
      - df: DataFrame already loaded (preferably with DatetimeIndex named 'date').
      - source: CSV URL or file path to load if df is None.
      - volume_col: column to use for USD volume (defaults to 'Volume USD').
      - window: rolling window size (in rows/hours) for max-volume normalization.

    Returns:
      - DataFrame with new feature columns and NaNs dropped.
    """
    if df is None:
        if source is None:
            raise ValueError("Provide either df or source.")
        df = pd.read_csv(source, parse_dates=["date"], index_col="date")
        df.sort_index(inplace=True)
        df.dropna(how="all", inplace=True)
        df.drop_duplicates(inplace=True)

    # Safety: normalize column names we use (case-insensitive)
    cols = {c.lower(): c for c in df.columns}
    need = ["open", "high", "low", "close"]
    for n in need:
        if n not in cols:
            raise KeyError(f"Missing required column '{n}' in DataFrame.")
    # volume column may have spaces/case, resolve using provided name
    if volume_col not in df.columns:
        # try to find case-insensitively
        candidates = [c for c in df.columns if c.lower().replace(" ", "") == volume_col.lower().replace(" ", "")]
        if not candidates:
            raise KeyError(f"Volume column '{volume_col}' not found.")
        volume_col = candidates[0]

    # Features
    df["feature_close"]  = df[cols["close"]].pct_change()
    df["feature_open"]   = df[cols["open"]] / df[cols["close"]]
    df["feature_high"]   = df[cols["high"]] / df[cols["close"]]
    df["feature_low"]    = df[cols["low"]]  / df[cols["close"]]
    df["feature_volume"] = df[volume_col] / df[volume_col].rolling(window).max()

    # Drop first rows created by pct_change/rolling
    df = df.dropna().copy()
    return df
