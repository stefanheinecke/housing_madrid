"""
Flask web application for Madrid Housing Price Predictor.
"""

import os
import sys

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from flask import Flask, jsonify, render_template, request
from src.data_loader import load_data
from src.model import ALL_FEATURES, BOOLEAN_FEATURES, NUMERIC_FEATURES, load_model, predict_price, train_model

app = Flask(__name__, template_folder="templates", static_folder="static")


def ensure_model_ready():
    """Train the model if it doesn't exist yet."""
    from src.model import MODEL_PATH
    if not os.path.exists(MODEL_PATH):
        print("No trained model found — training now...")
        df = load_data()
        train_model(df)


@app.route("/")
def index():
    """Serve the main page."""
    try:
        _, meta = load_model()
        districts = meta["districts"]
        defaults = meta["feature_defaults"]
        metrics = meta["metrics"]
    except FileNotFoundError:
        districts = []
        defaults = {}
        metrics = {}

    return render_template(
        "index.html",
        districts=districts,
        defaults=defaults,
        metrics=metrics,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint for price prediction."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    # Sanitize input — only accept known feature keys
    user_input = {}
    for key in ALL_FEATURES:
        if key in data and data[key] is not None and str(data[key]).strip() != "":
            val = data[key]
            if key in NUMERIC_FEATURES:
                try:
                    val = float(val)
                    if val < 0:
                        return jsonify({"error": f"'{key}' must be non-negative"}), 400
                except (ValueError, TypeError):
                    return jsonify({"error": f"Invalid value for '{key}'"}), 400
            elif key in BOOLEAN_FEATURES:
                try:
                    val = int(val)
                    if val not in (0, 1):
                        return jsonify({"error": f"'{key}' must be 0 or 1"}), 400
                except (ValueError, TypeError):
                    return jsonify({"error": f"Invalid value for '{key}'"}), 400
            elif key == "district":
                val = str(val).strip()
            user_input[key] = val

    if not user_input:
        return jsonify({"error": "Provide at least one feature (e.g. size_m2, rooms)"}), 400

    try:
        result = predict_price(user_input)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/model-info")
def api_model_info():
    """Return model metadata and available features."""
    try:
        _, meta = load_model()
        return jsonify({
            "metrics": meta["metrics"],
            "districts": meta["districts"],
            "features": meta["features"],
            "defaults": meta["feature_defaults"],
        })
    except FileNotFoundError:
        return jsonify({"error": "Model not trained yet"}), 503


@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    """Re-download data and retrain the model."""
    try:
        # Remove cached data to force re-download
        cached = os.path.join("data", "madrid_housing.csv")
        if os.path.exists(cached):
            os.remove(cached)
        df = load_data()
        metrics = train_model(df)
        return jsonify({"status": "ok", "metrics": metrics})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    ensure_model_ready()
    app.run(debug=True, host="127.0.0.1", port=5000)
