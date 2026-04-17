# =================================================================================
# Project:      PlasmaCast — Plasma Donor Demand Forecasting
# Contributor:  Puranjay Wadhera (GitHub: @PW13052003)
# File:         features.py
# Purpose:      Transforms the raw donor dataset into a model-ready featured
#               dataset by engineering calendar features, lag features, and
#               rolling average features. These additional features give the
#               XGBoost model richer context to learn patterns from.
# Language:     Python 3.12
# Libraries:    pandas, numpy, os
# Frameworks:   None
# APIs:         None
# GitHub:       https://github.com/PW13052003/plasmacast/blob/main/src/features.py
# =================================================================================


# Import the necessary modules
import pandas as pd
import numpy as np
import os


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def engineer_features(df):
    """
        Takes the raw donor dataset and engineers 9 additional features that
        help the model learn temporal and behavioral patterns.

        Features added:
            Calendar  — month, year, day_of_year, is_weekend, season
            Lag       — donor_lag_7, donor_lag_14
            Rolling   — rolling_7day_avg, rolling_14day_avg

        All lag and rolling features are computed per center_id to prevent
        data from one city influencing another city's features.

        Args:
            df [pd.DataFrame]: Raw donor dataset from data_gen.py

        Returns:
            pd.DataFrame: Featured dataset with 17 columns, ready for model training.
                          The first 14 rows per center are dropped due to NaN values
                          introduced by the 14-day lag window.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Sort by center then date to ensure correct time ordering before computing lag and rolling features.
    # Order matters for time-series
    df = df.sort_values(["center_id", "date"]).reset_index(drop=True)

    # --- Calendar Features --- #
    # These give the model an explicit sense of time without relying on raw date strings, which the model cannot process directly.
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["season"] = df["month"].map({
        12: "winter", 1: "winter", 2: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "fall",   10: "fall",  11: "fall"
    })
    df["season"] = df["season"].map({
        "winter": 0, "spring": 1, "summer": 2, "fall": 3
    })

    # --- Lag Features --- #
    # Lag features tell the model what donor count was N days ago at the same center.
    # The same day last week is a strong predictor of today.
    # shift(N) moves values forward by N rows, so row i gets the value from row i-N — i.e. N days in the past.
    df["donor_lag_7"] = df.groupby("center_id")["donor_count"].shift(7)
    df["donor_lag_14"] = df.groupby("center_id")["donor_count"].shift(14)

    # --- Rolling Average Features --- #
    # Rolling averages capture the recent trend at each center.
    # shift(1) is applied before rolling to ensure we only look at data BEFORE the current day.
    # Without this, the current day's value would leak into its own rolling average.
    df["rolling_7day_avg"] = (
        df.groupby("center_id")["donor_count"]
        .transform(lambda x: x.shift(1).rolling(window=7).mean())
    )
    df["rolling_14day_avg"] = (
        df.groupby("center_id")["donor_count"]
        .transform(lambda x: x.shift(1).rolling(window=14).mean())
    )

    # Drop rows with NaN from lag/rolling (first 14 days per center)
    df = df.dropna().reset_index(drop=True)

    return df

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(DATA_DIR, "donor_data.csv"))
    df_featured = engineer_features(df)
    df_featured.to_csv(os.path.join(DATA_DIR, "donor_data_featured.csv"), index=False)
    print(f"Featured dataset saved! {len(df_featured)} rows, {len(df_featured.columns)} columns")
    print(df_featured.columns.tolist())