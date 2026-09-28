import pandas as pd


def doy365(dates):
    """Map month/day to 2001; callers must explicitly exclude February 29."""
    dates = pd.Series(pd.to_datetime(dates), index=dates.index if isinstance(dates, pd.Series) else None)
    if dates.isna().any():
        raise ValueError("Dates must not be missing")
    if ((dates.dt.month == 2) & (dates.dt.day == 29)).any():
        raise ValueError("February 29 has no day in the fixed 365-day calendar")
    return pd.to_datetime("2001-" + dates.dt.strftime("%m-%d")).dt.dayofyear
