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

    # Time-based split — train on 2022, test on 2023
    train = df[df["date"].dt.year < 2023]
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

def plot_residuals(model, X_test, y_test):
    """
    Generates a residual plot showing prediction error over time for each city.
    Residual = Actual - Predicted. A well-performing model should show residuals
    randomly scattered around zero with no systematic pattern.
    Saved as: src/data/plot_residuals.png
    """
    predictions = model.predict(X_test)

    df_full = pd.read_csv(os.path.join(DATA_DIR, "donor_data_featured.csv"))
    df_full["date"] = pd.to_datetime(df_full["date"])
    test_df = df_full[df_full["date"].dt.year == 2023].copy().reset_index(drop=True)
    test_df["predicted"] = predictions
    test_df["actual"] = y_test.values
    test_df["residual"] = test_df["actual"] - test_df["predicted"]

    cities = sorted(test_df["city"].unique())
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(5, 2, figsize=(16, 20), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, city in enumerate(cities):
        city_df = test_df[test_df["city"] == city].reset_index(drop=True)
        axes[i].plot(city_df["residual"], color=colors[i], alpha=0.7, linewidth=0.8)
        axes[i].axhline(y=0, color="black", linestyle="--", linewidth=1)
        axes[i].set_ylim(-70, 70)
        axes[i].set_title(city, fontsize=11)
        axes[i].set_ylabel("Residual (donors)")

    fig.suptitle("Prediction Residuals by City (2023)\nActual − Predicted", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "plot_residuals.png"), dpi=150)
    plt.show()
    print("Residual plot saved to src/data/plot_residuals.png")


def plot_mae_by_city(model, X_test, y_test):
    """
    Generates a horizontal bar chart showing MAE per city as a percentage
    of each city's average daily donor count. Using percentage rather than
    absolute MAE allows fair comparison across cities of different sizes —
    an MAE of 15 donors means very different things for NYC (318 avg donors)
    vs Jacksonville (38 avg donors).
    Saved as: src/data/plot_mae_by_city.png
    """
    predictions = model.predict(X_test)

    df_full = pd.read_csv(os.path.join(DATA_DIR, "donor_data_featured.csv"))
    df_full["date"] = pd.to_datetime(df_full["date"])
    test_df = df_full[df_full["date"].dt.year == 2023].copy().reset_index(drop=True)
    test_df["predicted"] = predictions
    test_df["actual"] = y_test.values

    # Calculate MAE as a percentage of each city's average donor count
    # This normalizes for city size — making all 10 cities directly comparable
    city_mae = (
        test_df.groupby("city")
        .apply(lambda x: (mean_absolute_error(x["actual"], x["predicted"]) / x["actual"].mean()) * 100)
        .sort_values(ascending=False)
        .reset_index()
    )
    city_mae.columns = ["city", "mae_pct"]

    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(12, 7))

    bars = ax.barh(
        city_mae["city"],
        city_mae["mae_pct"],
        color=colors[:len(city_mae)],
        alpha=0.8
    )

    # Add percentage labels on each bar
    for bar, mae_val in zip(bars, city_mae["mae_pct"]):
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{mae_val:.1f}%",
            va="center",
            fontsize=10
        )

    # Overall MAE as percentage of overall mean donor count
    overall_mae_pct = (mean_absolute_error(y_test.values, predictions) / y_test.mean()) * 100
    ax.axvline(
        x=overall_mae_pct,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Overall MAE: {overall_mae_pct:.1f}%"
    )

    ax.set_xlabel("MAE as % of Average Daily Donor Count")
    ax.set_title("Model MAE by City as % of City Average (2023)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "plot_mae_by_city.png"), dpi=150)
    plt.show()
    print("MAE plot saved to src/data/plot_mae_by_city.png")

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
    plot_residuals(model, X_test, y_test)
    plot_mae_by_city(model, X_test, y_test)
    save_model(model, features)