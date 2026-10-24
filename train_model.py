"""
Week 4: Predictive Modeling and Optimization in Logistics Systems
Creates a synthetic dataset, trains baseline and regression models,
prints evaluation metrics, tunes a Random Forest, and saves metrics.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

def make_data(n=2400, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "distance_km": rng.uniform(2, 180, n),
        "traffic_level": rng.integers(1, 6, n),
        "weather_severity": rng.integers(0, 4, n),
        "dispatch_delay_min": rng.gamma(shape=2.0, scale=8.0, size=n),
        "package_weight_kg": rng.uniform(0.2, 35, n),
        "number_of_stops": rng.integers(1, 16, n),
        "warehouse_processing_min": rng.gamma(shape=2.2, scale=7.0, size=n),
    })
    noise = rng.normal(0, 0.35, n)
    df["delivery_time_hours"] = (
        0.12 + 0.018 * df["distance_km"]
        + 0.16 * df["traffic_level"]
        + 0.22 * df["weather_severity"]
        + 0.012 * df["dispatch_delay_min"]
        + 0.018 * df["number_of_stops"]
        + 0.010 * df["warehouse_processing_min"]
        + 0.002 * df["package_weight_kg"]
        + noise
    ).clip(lower=0.15)
    return df

def main():
    df = make_data()
    target = "delivery_time_hours"
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    models = {
        "Median baseline": DummyRegressor(strategy="median"),
        "Linear Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=250, max_depth=12,
                min_samples_leaf=3, random_state=42, n_jobs=-1
            )),
        ]),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({
            "model": name,
            "MAE_hours": mean_absolute_error(y_test, pred),
            "RMSE_hours": mean_squared_error(y_test, pred) ** 0.5,
            "R2": r2_score(y_test, pred),
        })

    results = pd.DataFrame(rows).sort_values("MAE_hours")
    print("\nInitial test-set comparison:")
    print(results.round(4).to_string(index=False))

    forest = models["Random Forest"]
    grid = {
        "model__n_estimators": [150, 250],
        "model__max_depth": [8, 12, None],
        "model__min_samples_leaf": [2, 3, 5],
    }
    search = GridSearchCV(
        forest, param_grid=grid, scoring="neg_mean_absolute_error",
        cv=5, n_jobs=-1
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    tuned_pred = best.predict(X_test)

    tuned_result = pd.DataFrame([{
        "model": "Tuned Random Forest",
        "MAE_hours": mean_absolute_error(y_test, tuned_pred),
        "RMSE_hours": mean_squared_error(y_test, tuned_pred) ** 0.5,
        "R2": r2_score(y_test, tuned_pred),
    }])
    all_results = pd.concat([results, tuned_result], ignore_index=True)
    all_results.to_csv(OUT / "evaluation_metrics.csv", index=False)
    print("\nBest Random Forest parameters:", search.best_params_)
    print("\nMetrics saved to outputs/evaluation_metrics.csv")
    print(tuned_result.round(4).to_string(index=False))

if __name__ == "__main__":
    main()
