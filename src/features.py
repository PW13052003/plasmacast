import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def engineer_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["center_id", "date"]).reset_index(drop=True)

    # Calendar features
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

    # Lag features (per center)
    df["donor_lag_7"] = df.groupby("center_id")["donor_count"].shift(7)
    df["donor_lag_14"] = df.groupby("center_id")["donor_count"].shift(14)

    # Rolling average features (per center)
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

