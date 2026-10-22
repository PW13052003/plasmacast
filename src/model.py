# ================================================================================
# Project:      PlasmaCast — Plasma Donor Demand Forecasting
# Contributor:  Puranjay Wadhera (GitHub: @PW13052003)
# File:         model.py
# Purpose:      Trains an XGBoost regression model to predict daily plasma
#               donor counts across 10 US cities. Evaluates model performance
#               using MAE and RMSE, and generates three diagnostic visualizations:
#               a residual plot per city, a MAE percentage bar chart, and a
#               feature importance chart.
# Language:     Python 3.12
# Libraries:    pandas, numpy, xgboost, scikit-learn, matplotlib, joblib, os, json
# Frameworks:   None
# APIs:         None
# GitHub:       https://github.com/PW13052003/plasmacast/blob/main/src/model.py
# ================================================================================


# Import the necessary modules
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt
import joblib
import json
import os


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_and_split(path=os.path.join(DATA_DIR, "donor_data_featured.csv")):
    """
        Loads the featured donor dataset and splits it into training and test sets
        using a time-based split — training on 2020-2022, testing on 2023.

        A time-based split is used instead of a random split because this is a
        time-series problem. A random split would allow future data to leak into
        training, producing misleadingly optimistic accuracy metrics.

        This gives us a ~75/25 split:
            Training: 10,820 rows (2020-2022, 10 cities)
            Testing:   3,650 rows (2023, 10 cities)

        Args:
            path (str): Path to the featured CSV dataset.

        Returns:
            X_train, y_train, X_test, y_test (pd.DataFrame/Series): Split datasets.
            features (list): List of feature column names used for training.
    """
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])

    # Define features and target
    features = [
        "temp_max", "precipitation", "day_of_week", "is_holiday",
        "month", "year", "day_of_year", "is_weekend", "season",
        "donor_lag_7", "donor_lag_14", "rolling_7day_avg", "rolling_14day_avg"
    ]
    target = "donor_count"

    # Time-based split — train on 2020-22, test on 2023
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
    """
        Trains an XGBoost regression model on the training dataset.

        XGBoost (Extreme Gradient Boosting) was chosen over alternatives like
        ARIMA or LSTM because:
            - It handles tabular data with mixed feature types very well
            - It is robust to outliers and missing patterns
            - It is highly interpretable via feature importance
            - It does not require feature scaling
            - It trains significantly faster than deep learning alternatives

        Hyperparameter rationale:
            n_estimators=500     — 500 trees gives a good balance of accuracy
                                  and training speed
            learning_rate=0.05   — conservative rate prevents overfitting
            max_depth=6          — allows complex patterns without overfitting
            subsample=0.8        — each tree sees 80% of rows, adds variety
            colsample_bytree=0.8 — each tree sees 80% of features, adds variety
            random_state=42      — ensures reproducible results every run
            n_jobs=-1            — uses all available CPU cores for speed

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target (donor counts).

        Returns:
            XGBRegressor: Trained XGBoost model.
    """
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
    """
        Evaluates the trained model on the test set using two metrics:

            MAE (Mean Absolute Error): Average number of donors the prediction
            is off by per day. Simple and intuitive — directly interpretable
            in the same units as the target variable.

            RMSE (Root Mean Squared Error): Similar to MAE but penalizes large
            errors more heavily. If RMSE is much higher than MAE, the model
            has some days where it is significantly wrong.

            MAE as % of mean: The most useful metric for cross-city comparison.
            Normalizes MAE by the average donor count so performance can be
            compared fairly across cities of very different sizes.

        Args:
            model (XGBRegressor): Trained XGBoost model.
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Actual donor counts for test period.

        Returns:
            np.ndarray: Array of predicted donor counts.
    """
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
        Generates a 5x2 grid of residual plots, one per city.

        Residuals are expressed as a percentage of the actual donor count
        to remove city-size bias — a 100-donor error means very different
        things for NYC vs Jacksonville.

        Saved as: src/data/plot_residuals.png

        Args:
            model (XGBRegressor): Trained XGBoost model.
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Actual donor counts for test period.

        Returns:
            NONE
    """
    predictions = model.predict(X_test)

    df_full = pd.read_csv(os.path.join(DATA_DIR, "donor_data_featured.csv"))
    df_full["date"] = pd.to_datetime(df_full["date"])
    test_df = df_full[df_full["date"].dt.year == 2023].copy().reset_index(drop=True)
    test_df["predicted"] = predictions
    test_df["actual"] = y_test.values
    test_df["residual"] = test_df["actual"] - test_df["predicted"]
    test_df["residual_pct"] = test_df["residual"] / test_df["actual"] * 100

    cities = sorted(test_df["city"].unique())
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(5, 2, figsize=(16, 20), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, city in enumerate(cities):
        city_df = test_df[test_df["city"] == city].reset_index(drop=True)
        axes[i].plot(city_df["residual_pct"], color=colors[i], alpha=0.7, linewidth=0.8)
        axes[i].axhline(y=0, color="black", linestyle="--", linewidth=1)
        axes[i].set_title(city, fontsize=11)
        axes[i].set_ylabel("Residual (% of actual)")

    fig.suptitle("Prediction Residuals by City (2023)\n(Actual − Predicted) as % of Actual", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "plot_residuals.png"), dpi=150)
    plt.show()
    print("Residual plot saved to src/data/plot_residuals.png")


def plot_mae_by_city(model, X_test, y_test):
    """
        Generates a horizontal bar chart showing MAE per city as a percentage
        of each city's average daily donor count.

        Args:
            model (XGBRegressor): Trained XGBoost model.
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Actual donor counts for test period.

        Returns:
            NONE
    """
    predictions = model.predict(X_test)

    df_full = pd.read_csv(os.path.join(DATA_DIR, "donor_data_featured.csv"))
    df_full["date"] = pd.to_datetime(df_full["date"])
    test_df = df_full[df_full["date"].dt.year == 2023].copy().reset_index(drop=True)
    test_df["predicted"] = predictions
    test_df["actual"] = y_test.values

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

    for bar, mae_val in zip(bars, city_mae["mae_pct"]):
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{mae_val:.1f}%",
            va="center",
            fontsize=10
        )

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


def plot_feature_importance(model, features):
    """
        Generates a horizontal bar chart of XGBoost feature importances,
        ranked from most to least important.

        Uses the model's built-in 'gain' importance — the average improvement
        in loss brought by a feature across all trees it appears in. Gain is
        more informative than 'weight' (split count) because it reflects how
        much each feature actually improves predictions, not just how often
        it is used.

        Saved as: src/data/plot_feature_importance.png

        Args:
            model (XGBRegressor): Trained XGBoost model.
            features (list): Feature names in the same order as training columns.

        Returns:
            NONE
    """
    importances = model.get_booster().get_score(importance_type="gain")

    # Map internal feature names (f0, f1, ...) back to readable names
    feature_map = {f"f{i}": name for i, name in enumerate(features)}
    named = {feature_map.get(k, k): v for k, v in importances.items()}

    # Sort descending
    sorted_features = sorted(named.items(), key=lambda x: x[1], reverse=True)
    names  = [f for f, _ in sorted_features]
    values = [v for _, v in sorted_features]

    # Normalize to 0-100% for readability
    total = sum(values)
    pct   = [v / total * 100 for v in values]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(names[::-1], pct[::-1], color="#4C72B0", alpha=0.85)

    for bar, p in zip(bars, pct[::-1]):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{p:.1f}%",
            va="center",
            fontsize=10
        )

    ax.set_xlabel("Relative Importance (% of total gain)")
    ax.set_title("XGBoost Feature Importance by Gain")
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "plot_feature_importance.png"), dpi=150)
    plt.show()
    print("Feature importance plot saved to src/data/plot_feature_importance.png")


def save_metrics(model, X_test, y_test, features):
    """
        Computes and saves all model performance metrics to src/data/metrics.json.
        This file is the single source of truth for all numbers shown in the
        dashboard, README, and static site.

        Args:
            model (XGBRegressor): Trained XGBoost model.
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Actual donor counts for test period.
            features (list): Feature names.

        Returns:
            NONE
    """
    predictions = model.predict(X_test)

    mae      = mean_absolute_error(y_test, predictions)
    rmse     = root_mean_squared_error(y_test, predictions)
    mae_pct  = round(mae / y_test.mean() * 100, 1)
    mean_donors = round(y_test.mean(), 1)

    # Naive baseline: predict same as 7 days ago (donor_lag_7)
    df_full = pd.read_csv(os.path.join(DATA_DIR, "donor_data_featured.csv"))
    df_full["date"] = pd.to_datetime(df_full["date"])
    test_df = df_full[df_full["date"].dt.year == 2023].copy().reset_index(drop=True)
    naive_mae     = mean_absolute_error(test_df["donor_count"], test_df["donor_lag_7"])
    naive_mae_pct = round(naive_mae / test_df["donor_count"].mean() * 100, 1)

    # Per-city MAE %
    test_df["predicted"] = predictions
    test_df["actual"]    = y_test.values
    city_mae = (
        test_df.groupby("city")
        .apply(lambda x: round(
            mean_absolute_error(x["actual"], x["predicted"]) / x["actual"].mean() * 100, 1
        ))
        .to_dict()
    )

    metrics = {
        "mae":                round(mae, 2),
        "rmse":               round(rmse, 2),
        "mae_pct":            mae_pct,
        "naive_lag7_mae_pct": naive_mae_pct,
        "mean_daily_donors":  mean_donors,
        "mae_pct_by_city":    city_mae,
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to src/data/metrics.json")
    print(json.dumps(metrics, indent=2))


def save_model(model, features, path=os.path.join(DATA_DIR, "model.pkl")):
    """
        Saves the trained model and feature list together as a single .pkl file
        using joblib. Saving both together ensures api.py always loads a model
        and its expected features in sync — preventing mismatches if features
        are ever added or removed during future retraining.

        Args:
            model (XGBRegressor): Trained XGBoost model.
            features (list): List of feature names the model was trained on.
            path (str): Destination path for the saved model file.

        Returns:
            NONE
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    joblib.dump({"model": model, "features": features}, path)
    print(f"Model saved to {path}")


def load_model(path=os.path.join(DATA_DIR, "model.pkl")):
    """
        Loads a previously saved model and its feature list from disk.
        Used by api.py at startup to load the model without retraining.

        Args:
            path (str): Path to the saved .pkl model file.

        Returns:
            model (XGBRegressor): The loaded trained model.
            features (list): The feature list the model expects.
    """
    data = joblib.load(path)
    return data["model"], data["features"]


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, features = load_and_split()
    model = train_model(X_train, y_train)
    predictions = evaluate_model(model, X_test, y_test)
    plot_residuals(model, X_test, y_test)
    plot_mae_by_city(model, X_test, y_test)
    plot_feature_importance(model, features)
    save_metrics(model, X_test, y_test, features)
    save_model(model, features)