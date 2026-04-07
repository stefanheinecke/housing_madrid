"""
Flask web application for Madrid Housing Price Predictor.
"""

import os
import sys
import threading

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from flask import Flask, jsonify, render_template, request
from src.data_loader import load_data
from src.model import ALL_FEATURES, BOOLEAN_FEATURES, NUMERIC_FEATURES, load_model, predict_price, train_model

app = Flask(__name__, template_folder="templates", static_folder="static")

# Track model training state
_model_ready = threading.Event()
_training_error = None


def _train_in_background():
    """Train the model in a background thread so the app can serve requests immediately."""
    global _training_error
    from src.model import MODEL_PATH
    try:
        if not os.path.exists(MODEL_PATH):
            print("No trained model found — training now...", flush=True)
            df = load_data()
            train_model(df)
        _model_ready.set()
        print("✓ Model ready to serve requests", flush=True)
    except Exception as e:
        _training_error = str(e)
        _model_ready.set()  # Unblock waiting requests so they can see the error
        print(f"✗ Model training failed: {e}", flush=True)


@app.route("/health")
def health():
    """Health check endpoint for Railway."""
    return jsonify({"status": "ok", "model_ready": _model_ready.is_set()})


@app.route("/")
def index():
    """Serve the main page."""
    model_loading = not _model_ready.is_set()
    try:
        _, meta = load_model()
        districts = meta["districts"]
        neighborhoods = meta["neighborhoods"]
        district_neighborhoods = meta["district_neighborhoods"]
        defaults = meta["feature_defaults"]
        metrics = meta["metrics"]
    except FileNotFoundError:
        districts = []
        neighborhoods = []
        district_neighborhoods = {}
        defaults = {}
        metrics = {}

    return render_template(
        "index.html",
        districts=districts,
        neighborhoods=neighborhoods,
        district_neighborhoods=district_neighborhoods,
        defaults=defaults,
        metrics=metrics,
        model_loading=model_loading,
        training_error=_training_error,
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
            elif key == "neighborhood":
                val = str(val).strip()
            user_input[key] = val

    if not user_input:
        return jsonify({"error": "Provide at least one feature (e.g. size_m2, rooms)"}), 400

    if not _model_ready.is_set():
        return jsonify({"error": "Model is still training. Please try again in a moment."}), 503
    if _training_error:
        return jsonify({"error": f"Model training failed: {_training_error}"}), 503

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


# Start model training in background thread (non-blocking so the port opens immediately)
threading.Thread(target=_train_in_background, daemon=True).start()

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
