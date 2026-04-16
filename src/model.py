import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt
import joblib
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def load_and_split(path=os.path.join(DATA_DIR, "donor_data_featured.csv")):
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

def train_model(X_train, y_train):
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("Model training complete!")
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)

    print("\n=== MODEL EVALUATION ===")
    print(f"MAE:  {mae:.2f} donors")
    print(f"RMSE: {rmse:.2f} donors")
    print(f"Mean actual donor count: {y_test.mean():.2f}")
    print(f"MAE as % of mean: {(mae / y_test.mean() * 100):.1f}%")

    return predictions


def plot_predictions(model, X_test, y_test, features):
    predictions = model.predict(X_test)

    # Plot predicted vs actual for one city
    test_df = X_test.copy()
    test_df["actual"] = y_test.values
    test_df["predicted"] = predictions

    plt.figure(figsize=(14, 5))
    plt.plot(y_test.values[:365], label="Actual", alpha=0.7, color="steelblue")
    plt.plot(predictions[:365], label="Predicted", alpha=0.7, color="darkorange", linestyle="--")
    plt.title("Predicted vs Actual Donor Count (2023 - Philadelphia)")
    plt.xlabel("Day")
    plt.ylabel("Donor Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "predictions_plot.png"), dpi=150)
    plt.show()
    print("Plot saved to src/data/predictions_plot.png")

def save_model(model, features, path=os.path.join(DATA_DIR, "model.pkl")):
    os.makedirs(DATA_DIR, exist_ok=True)
    joblib.dump({"model": model, "features": features}, path)
    print(f"Model saved to {path}")


def load_model(path=os.path.join(DATA_DIR, "model.pkl")):
    data = joblib.load(path)
    return data["model"], data["features"]


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, features = load_and_split()
    model = train_model(X_train, y_train)
    predictions = evaluate_model(model, X_test, y_test)
    plot_predictions(model, X_test, y_test, features)
    save_model(model, features)