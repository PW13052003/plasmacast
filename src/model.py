import pandas as pd
import numpy as np

def load_and_split(path="data/donor_data_featured.csv"):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])

    # Define features and target
    features = [
        "temp_max", "precipitation", "day_of_week", "is_holiday",
        "month", "year", "day_of_year", "is_weekend", "season",
        "donor_lag_7", "donor_lag_14", "rolling_7day_avg", "rolling_14day_avg"
    ]
    target = "donor_count"

    # Time based split — train on 2022, test on 2023
    train = df[df["date"].dt.year == 2022]
    test = df[df["date"].dt.year == 2023]

    X_train = train[features]
    y_train = train[target]
    X_test = test[features]
    y_test = test[target]

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Features: {features}")

    return X_train, y_train, X_test, y_test, features


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, features = load_and_split()