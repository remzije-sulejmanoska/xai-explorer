"""
app.py — XAI Explorer Flask application.

Serves an interactive dashboard for exploring a Random Forest model's
predictions on the UCI Student Performance dataset through SHAP and
LIME explainability methods, alongside a Membership Inference Attack
privacy analysis and an original Feature Risk Score metric.

See notebooks/eda-and-model-training.ipynb for the full data science
workflow behind this application — this file only wires HTTP routes
to the logic implemented in ml_pipeline.py.

Bachelor's Thesis Project — 2026
"""

import traceback

import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import config
import ml_pipeline as ml

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────
# Load dataset, model and explainers once at startup
# ─────────────────────────────────────────────────────────────
print("Loading dataset and model...")
dataset = ml.prepare_dataset()
model = ml.load_model()
xai = ml.Explainability(model, dataset)
print(f"Ready — {len(dataset.feature_names)} features, "
      f"{len(dataset.X_train)} training / {len(dataset.X_test)} test students.")


def error_response(e, status=500):
    """Uniform JSON error payload; stack traces only included in debug mode."""
    payload = {"error": str(e)}
    if config.DEBUG:
        payload["traceback"] = traceback.format_exc()
    return jsonify(payload), status


@app.errorhandler(500)
def internal_error(e):
    return error_response(e, 500)


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": f"Route not found: {request.path}"}), 404


# ═══════════════════════════════════════════════════════
# Page
# ═══════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html", feature_names=dataset.feature_names)


# ═══════════════════════════════════════════════════════
# Debug / introspection
# ═══════════════════════════════════════════════════════
@app.route("/debug_features")
def debug_features():
    return jsonify({
        "feature_count": len(dataset.feature_names),
        "feature_names": dataset.feature_names,
        "X_train_shape": list(dataset.X_train.shape),
    })


# ═══════════════════════════════════════════════════════
# Dashboard stats
# ═══════════════════════════════════════════════════════
@app.route("/stats")
def stats():
    try:
        y_pred = model.predict(dataset.X_test)
        return jsonify({
            "students": len(dataset.X_train) + len(dataset.X_test),
            "features": len(dataset.feature_names),
            "algorithm": "Random Forest",
            "dataset": "UCI Student Performance",
            "accuracy": f"{accuracy_score(dataset.y_test, y_pred):.2%}",
            "precision": f"{precision_score(dataset.y_test, y_pred):.2%}",
            "recall": f"{recall_score(dataset.y_test, y_pred):.2%}",
            "f1": f"{f1_score(dataset.y_test, y_pred):.2%}",
        })
    except Exception as e:
        return error_response(e)


# ═══════════════════════════════════════════════════════
# Prediction
# ═══════════════════════════════════════════════════════
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "No valid JSON body received"}), 400

        values = data.get("values", [])
        expected = len(dataset.feature_names)

        if len(values) != expected:
            return jsonify({
                "error": f"Expected {expected} values, received {len(values)}",
                "expected": expected,
                "received": len(values),
                "feature_names": dataset.feature_names,
            }), 400

        input_array = np.array([values], dtype=float)
        prediction = int(model.predict(input_array)[0])
        probability = float(model.predict_proba(input_array)[0][1])
        prob_pct = round(probability * 100, 2)
        risk_level = ml.risk_level_from_probability(prob_pct)

        shap_vals, _ = xai.explain_instance_shap(input_array)
        positive, negative = ml.rank_factors(dataset.feature_names, shap_vals)
        top_positive = [f["name"] for f in positive]
        top_negative = [f["name"] for f in negative] or ["No significant risk factors detected"]

        return jsonify({
            "prediction": prediction,
            "probability": prob_pct,
            "label": "PASS" if prediction == 1 else "FAIL",
            "risk_level": risk_level,
            "top_positive": top_positive,
            "top_negative": top_negative,
        })
    except Exception as e:
        return error_response(e)


# ═══════════════════════════════════════════════════════
# SHAP — local explanation
# ═══════════════════════════════════════════════════════
@app.route("/shap_local", methods=["POST"])
def shap_local():
    try:
        data = request.get_json()
        input_array = np.array([data["values"]], dtype=float)

        shap_vals, waterfall_img = ml.local_shap_explanation(xai, input_array)
        positive, negative = ml.rank_factors(dataset.feature_names, shap_vals)
        if not negative:
            negative = [{"name": "No significant risk factors identified", "value": 0}]

        return jsonify({
            "waterfall_img": waterfall_img,
            "positive": positive,
            "negative": negative,
            "shap_values": [round(float(v), 4) for v in shap_vals],
        })
    except Exception as e:
        return error_response(e)


# ═══════════════════════════════════════════════════════
# SHAP — global explanation
# ═══════════════════════════════════════════════════════
@app.route("/global_shap")
def global_shap():
    try:
        summary_img, bar_img, top5 = ml.global_shap_plots(xai)

        top1 = top5[0]["name"] if top5 else "failures"
        top2 = top5[1]["name"] if len(top5) > 1 else "absences"
        insight = (f"'{top1}' is the most influential global factor. "
                   f"'{top2}' also has a substantial impact on the prediction.")

        return jsonify({
            "summary_img": summary_img,
            "bar_img": bar_img,
            "top5": top5,
            "insight": insight,
        })
    except Exception as e:
        return error_response(e)


# ═══════════════════════════════════════════════════════
# LIME — local explanation
# ═══════════════════════════════════════════════════════
@app.route("/lime_local", methods=["POST"])
def lime_local():
    try:
        data = request.get_json()
        input_array = np.array([data["values"]], dtype=float)

        lime_img, lime_list = ml.lime_local_plot(xai, input_array)
        return jsonify({"lime_img": lime_img, "lime_list": lime_list})
    except Exception as e:
        return error_response(e)


# ═══════════════════════════════════════════════════════
# SHAP vs LIME comparison
# ═══════════════════════════════════════════════════════
@app.route("/comparison", methods=["POST"])
def comparison():
    try:
        data = request.get_json()
        input_array = np.array([data["values"]], dtype=float)

        table, comparison_img, conclusion = ml.comparison_plot(xai, input_array)
        return jsonify({
            "table": table,
            "comparison_img": comparison_img,
            "conclusion": conclusion,
        })
    except Exception as e:
        return error_response(e)


# ═══════════════════════════════════════════════════════
# Privacy & Security
# ═══════════════════════════════════════════════════════
@app.route("/privacy_features")
def privacy_features():
    try:
        return jsonify(ml.privacy_analysis(xai))
    except Exception as e:
        return error_response(e)


# ═══════════════════════════════════════════════════════
# Results & Conclusions
# ═══════════════════════════════════════════════════════
@app.route("/results")
def results():
    try:
        y_pred = model.predict(dataset.X_test)
        return jsonify({
            "accuracy": f"{accuracy_score(dataset.y_test, y_pred):.2%}",
            "precision": f"{precision_score(dataset.y_test, y_pred):.2%}",
            "recall": f"{recall_score(dataset.y_test, y_pred):.2%}",
            "f1": f"{f1_score(dataset.y_test, y_pred):.2%}",
            "findings": config.RESULT_FINDINGS,
        })
    except Exception as e:
        return error_response(e)


# ═══════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(debug=config.DEBUG)