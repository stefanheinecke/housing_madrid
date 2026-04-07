"""
ML model for Madrid housing price prediction.

Uses a Gradient Boosting Regressor with district encoding.
Supports partial input — missing features are filled with training medians.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "price_model.pkl")
META_PATH = os.path.join(MODEL_DIR, "model_meta.pkl")

NUMERIC_FEATURES = ["size_m2", "rooms", "bathrooms", "floor"]
BOOLEAN_FEATURES = ["has_elevator", "is_exterior", "has_parking"]
CATEGORICAL_FEATURES = ["district"]
ALL_FEATURES = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES


def _build_pipeline(districts: list[str]) -> Pipeline:
    """Build the sklearn pipeline with preprocessing and model."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        categories=[districts], handle_unknown="infrequent_if_exist", sparse_output=False
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES + BOOLEAN_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", GradientBoostingRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
        ))
    ])
    return pipeline


def train_model(df: pd.DataFrame) -> dict:
    """Train the pricing model and save to disk.

    Returns a dict with evaluation metrics.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Prepare data
    df = df.dropna(subset=["price"])
    X = df[ALL_FEATURES].copy()
    y = df["price"].values

    # Get sorted district list for consistent encoding
    districts = sorted(X["district"].unique().tolist())

    # Compute training medians/modes for imputation at prediction time
    feature_defaults = {}
    for col in NUMERIC_FEATURES:
        feature_defaults[col] = float(X[col].median())
    for col in BOOLEAN_FEATURES:
        feature_defaults[col] = int(X[col].mode().iloc[0]) if len(X[col].mode()) > 0 else 0
    feature_defaults["district"] = X["district"].mode().iloc[0] if len(X["district"].mode()) > 0 else districts[0]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build and train
    pipeline = _build_pipeline(districts)
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    metrics = {
        "r2": round(r2_score(y_test, y_pred), 4),
        "mae": round(mean_absolute_error(y_test, y_pred), 2),
        "mape": round(mean_absolute_percentage_error(y_test, y_pred) * 100, 2),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    # Save model and metadata
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump({
        "feature_defaults": feature_defaults,
        "districts": districts,
        "metrics": metrics,
        "features": ALL_FEATURES,
    }, META_PATH)

    print(f"✓ Model trained — R²={metrics['r2']}, MAE=€{metrics['mae']:,.0f}, MAPE={metrics['mape']}%")
    return metrics


def load_model() -> tuple[Pipeline, dict]:
    """Load trained model and metadata from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model not found. Run train_model() first or start the app to auto-train."
        )
    pipeline = joblib.load(MODEL_PATH)
    meta = joblib.load(META_PATH)
    return pipeline, meta


def predict_price(user_input: dict) -> dict:
    """Predict price from (possibly partial) user input.

    user_input: dict with any subset of feature keys.
    Missing features are filled with training defaults.

    Returns dict with predicted price, confidence range, and price_per_m2.
    """
    pipeline, meta = load_model()
    defaults = meta["feature_defaults"]

    # Build a complete feature row, filling missing with defaults
    row = {}
    for feat in ALL_FEATURES:
        if feat in user_input and user_input[feat] is not None and user_input[feat] != "":
            row[feat] = user_input[feat]
        else:
            row[feat] = defaults[feat]

    # Type casting
    for col in NUMERIC_FEATURES:
        row[col] = float(row[col])
    for col in BOOLEAN_FEATURES:
        row[col] = int(row[col])
    row["district"] = str(row["district"])

    X = pd.DataFrame([row])
    predicted_price = float(pipeline.predict(X)[0])

    # Confidence range based on model MAPE
    mape = meta["metrics"]["mape"] / 100
    lower = predicted_price * (1 - mape)
    upper = predicted_price * (1 + mape)

    size = row["size_m2"]
    price_per_m2 = predicted_price / size if size > 0 else 0

    return {
        "predicted_price": round(predicted_price, 2),
        "price_lower": round(lower, 2),
        "price_upper": round(upper, 2),
        "price_per_m2": round(price_per_m2, 2),
        "features_used": row,
        "defaults_applied": [k for k in ALL_FEATURES if k not in user_input or user_input.get(k) in (None, "")],
    }


if __name__ == "__main__":
    from data_loader import load_data

    df = load_data()
    metrics = train_model(df)
    print(f"\nMetrics: {metrics}")

    # Test prediction
    result = predict_price({"size_m2": 80, "rooms": 3, "bathrooms": 2, "district": "Salamanca"})
    print(f"\nPrediction: €{result['predicted_price']:,.0f}")
    print(f"Range: €{result['price_lower']:,.0f} – €{result['price_upper']:,.0f}")
    print(f"Price/m²: €{result['price_per_m2']:,.0f}")
