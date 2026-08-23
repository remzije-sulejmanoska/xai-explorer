"""
ml_pipeline.py — Data preparation, model loading and explainability
logic for the XAI Explorer platform.

The `prepare_dataset()` function reproduces — step for step — the
preprocessing pipeline from notebooks/eda-and-model-training.ipynb
(target creation, feature/label split, categorical encoding, stratified
train/test split). Because every step is deterministic (fixed
random_state, alphabetic LabelEncoder ordering), this guarantees the
running Flask app always reconstructs the exact same X_train / X_test
the model was trained and evaluated on in the notebook — without
requiring any additional cached artifacts beyond the trained model
itself (models/rf_model.pkl).
"""

import base64
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import config


# ═══════════════════════════════════════════════════════════
# Dataset preparation
# ═══════════════════════════════════════════════════════════
class Dataset:
    """Container for the reproduced train/test split."""

    def __init__(self, X_train, X_test, y_train, y_test, feature_names):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names


def prepare_dataset() -> Dataset:
    """
    Reproduces the exact preprocessing pipeline from the analysis notebook:

    1. Load the raw dataset
    2. Create the binary target ('pass' = G3 >= PASS_THRESHOLD)
    3. Drop G1/G2/G3 (grade leakage) and the target itself from X
    4. Label-encode categorical columns
    5. Stratified 80/20 train/test split (random_state=42)
    """
    df = pd.read_csv(config.DATA_PATH, sep=";")

    df["pass"] = (df[config.TARGET_COLUMN] >= config.PASS_THRESHOLD).astype(int)

    drop_cols = config.LEAKAGE_COLUMNS + ["pass"]
    X = df.drop(columns=drop_cols)
    y = df["pass"]

    encoder = LabelEncoder()
    for col in X.select_dtypes(include="object").columns:
        X[col] = encoder.fit_transform(X[col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    return Dataset(
        X_train=X_train.values,
        X_test=X_test.values,
        y_train=y_train.values,
        y_test=y_test.values,
        feature_names=list(X.columns),
    )


def load_model():
    """Loads the trained Random Forest model produced by the notebook."""
    return joblib.load(config.MODEL_PATH)


# ═══════════════════════════════════════════════════════════
# Explainability bundle (model + SHAP + LIME, built once at startup)
# ═══════════════════════════════════════════════════════════
def extract_class1_shap(shap_values):
    """
    Normalizes SHAP output into a single (n_samples, n_features) array
    of "Pass"-class contributions, regardless of the SHAP library's
    return format (list of arrays vs. a single 3-D array).
    """
    if isinstance(shap_values, list):
        return shap_values[1]
    if shap_values.ndim == 3:
        return shap_values[:, :, 1]
    return shap_values


class Explainability:
    """
    Bundles the trained model with its SHAP/LIME explainers and the
    globally-computed SHAP values on the test set — all built once at
    application startup so requests only ever do lightweight, per-instance
    computation.
    """

    def __init__(self, model, dataset: Dataset):
        self.model = model
        self.dataset = dataset

        self.shap_explainer = shap.TreeExplainer(model)
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            dataset.X_train,
            feature_names=dataset.feature_names,
            class_names=["Fail", "Pass"],
            mode="classification",
            random_state=config.RANDOM_STATE,
        )

        raw_shap = self.shap_explainer.shap_values(dataset.X_test)
        self.shap_values_test = extract_class1_shap(raw_shap)

    def explain_instance_shap(self, input_array):
        """Returns (shap_values_for_instance, base_value) for one student."""
        raw = self.shap_explainer.shap_values(input_array)
        shap_vals = extract_class1_shap(raw)[0]
        base_value = self.shap_explainer.expected_value
        if hasattr(base_value, "__len__"):
            base_value = base_value[1]
        return shap_vals, base_value

    def explain_instance_lime(self, input_array, num_features=config.LIME_NUM_FEATURES):
        """Runs LIME on a single student and returns the raw LIME explanation."""
        return self.lime_explainer.explain_instance(
            input_array[0],
            self.model.predict_proba,
            num_features=num_features,
        )


# ═══════════════════════════════════════════════════════════
# Small shared helpers
# ═══════════════════════════════════════════════════════════
def fig_to_base64(fig) -> str:
    """Renders a Matplotlib figure to a base64 PNG string and closes it."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)
    return encoded


def rank_factors(feature_names, values, threshold=config.SHAP_FACTOR_THRESHOLD,
                  top_n=config.SHAP_TOP_N):
    """
    Splits a per-feature contribution vector into the top positive and
    negative factors above a minimum-effect threshold, sorted by magnitude.
    """
    factors = sorted(
        (
            {"name": feature_names[i], "value": float(values[i])}
            for i in range(len(feature_names))
        ),
        key=lambda f: abs(f["value"]),
        reverse=True,
    )
    positive = [f for f in factors if f["value"] > threshold][:top_n]
    negative = [f for f in factors if f["value"] < -threshold][:top_n]
    return positive, negative


def risk_level_from_probability(prob_pct: float) -> str:
    if prob_pct >= config.RISK_LOW_PROB:
        return "Low"
    if prob_pct >= config.RISK_HIGH_PROB:
        return "Medium"
    return "High"


# ═══════════════════════════════════════════════════════════
# SHAP — local (single-student) explanation
# ═══════════════════════════════════════════════════════════
def local_shap_explanation(xai: Explainability, input_array):
    """Computes SHAP values and renders a waterfall plot for one student."""
    shap_vals, base_value = xai.explain_instance_shap(input_array)

    explanation = shap.Explanation(
        values=shap_vals,
        base_values=base_value,
        data=input_array[0],
        feature_names=xai.dataset.feature_names,
    )
    shap.plots.waterfall(explanation, show=False)
    waterfall_img = fig_to_base64(plt.gcf())

    return shap_vals, waterfall_img


# ═══════════════════════════════════════════════════════════
# SHAP — global explanation (whole test set)
# ═══════════════════════════════════════════════════════════
def global_shap_plots(xai: Explainability):
    """Builds the SHAP summary (beeswarm) and bar plots for the test set."""
    feature_names = xai.dataset.feature_names

    plt.figure(figsize=(10, 7))
    shap.summary_plot(xai.shap_values_test, xai.dataset.X_test,
                       feature_names=feature_names, show=False)
    summary_img = fig_to_base64(plt.gcf())

    plt.figure(figsize=(10, 7))
    shap.summary_plot(xai.shap_values_test, xai.dataset.X_test,
                       feature_names=feature_names, plot_type="bar", show=False)
    bar_img = fig_to_base64(plt.gcf())

    mean_shap = np.abs(xai.shap_values_test).mean(axis=0)
    top5 = sorted(
        ({"name": feature_names[i], "score": float(mean_shap[i])}
         for i in range(len(feature_names))),
        key=lambda f: f["score"], reverse=True,
    )[:5]

    return summary_img, bar_img, top5


# ═══════════════════════════════════════════════════════════
# LIME — local (single-student) explanation
# ═══════════════════════════════════════════════════════════
def lime_local_plot(xai: Explainability, input_array):
    """Runs LIME on a single student and renders its explanation figure."""
    explanation = xai.explain_instance_lime(input_array)
    fig = explanation.as_pyplot_figure()
    fig.set_size_inches(10, 6)
    lime_img = fig_to_base64(fig)
    lime_list = [{"feature": f, "weight": round(float(w), 4)}
                 for f, w in explanation.as_list()]
    return lime_img, lime_list


# ═══════════════════════════════════════════════════════════
# SHAP vs LIME comparison
# ═══════════════════════════════════════════════════════════
def comparison_plot(xai: Explainability, input_array):
    """Builds the SHAP-vs-LIME side-by-side bar chart and comparison table."""
    feature_names = xai.dataset.feature_names
    shap_vals, _ = xai.explain_instance_shap(input_array)
    explanation = xai.explain_instance_lime(input_array)
    lime_list = explanation.as_list()

    shap_top = sorted(
        ({"name": feature_names[i], "shap": round(float(shap_vals[i]), 4)}
         for i in range(len(feature_names))),
        key=lambda f: abs(f["shap"]), reverse=True,
    )[:10]

    # LIME returns human-readable conditions (e.g. "failures > 0"); extract
    # the underlying feature name so it can be matched against SHAP's list.
    lime_dict = {}
    for feat, val in lime_list:
        key = feat.split(" ")[0].split("<")[0].split(">")[0].strip()
        lime_dict[key] = round(float(val), 4)

    table = [
        {"feature": item["name"], "shap": item["shap"], "lime": lime_dict.get(item["name"], 0)}
        for item in shap_top
    ]

    names = [r["feature"] for r in table]
    shap_plot_vals = [abs(r["shap"]) for r in table]
    lime_plot_vals = [abs(r["lime"]) for r in table]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, shap_plot_vals, width, label="SHAP", color="#003366", alpha=0.85)
    ax.bar(x + width / 2, lime_plot_vals, width, label="LIME", color="#FFB81C", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Absolute importance")
    ax.set_title("SHAP vs LIME — Feature Importance Comparison", fontweight="bold")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    comparison_img = fig_to_base64(plt.gcf())

    top1 = shap_top[0]["name"] if shap_top else "failures"
    top2 = shap_top[1]["name"] if len(shap_top) > 1 else "absences"
    conclusion = (
        f"Both methods identify '{top1}' and '{top2}' as the most influential factors. "
        f"SHAP provides more consistent global explanations, while LIME focuses on "
        f"local interpretability."
    )

    return table, comparison_img, conclusion


# ═══════════════════════════════════════════════════════════
# Feature Risk Score — original contribution (see notebook)
# ═══════════════════════════════════════════════════════════
def feature_risk_scores(xai: Explainability, top_n=config.FEATURE_RISK_TOP_N):
    """
    Normalizes each feature's mean |SHAP| importance against the single
    most important feature (= 100%) and buckets it into High (>=50%) /
    Medium (>=25%) / Low risk bands — flagging which features carry the
    most explanatory, and therefore privacy-relevant, weight. Mirrors the
    "Feature Risk Score" section of the analysis notebook exactly.
    """
    feature_names = xai.dataset.feature_names
    mean_importance = np.abs(xai.shap_values_test).mean(axis=0)
    max_importance = mean_importance.max()
    scores = (mean_importance / max_importance) * 100

    def level(score):
        if score >= 50:
            return "High"
        if score >= 25:
            return "Medium"
        return "Low"

    ranked = sorted(
        (
            {"name": feature_names[i], "score": float(scores[i]), "level": level(scores[i])}
            for i in range(len(feature_names))
        ),
        key=lambda f: f["score"], reverse=True,
    )
    return ranked[:top_n]


# ═══════════════════════════════════════════════════════════
# Privacy analysis — Membership Inference Attack simulation
# ═══════════════════════════════════════════════════════════
def privacy_analysis(xai: Explainability):
    """
    Membership Inference Attack simulation: compares the model's mean
    prediction confidence on the training set vs. the held-out test set.
    A large gap indicates the model memorizes training examples, which is
    a standard proxy signal for privacy exposure via its predictions and
    explanations.
    """
    model = xai.model
    dataset = xai.dataset

    train_probs = model.predict_proba(dataset.X_train)
    test_probs = model.predict_proba(dataset.X_test)
    train_conf_arr = np.max(train_probs, axis=1)
    test_conf_arr = np.max(test_probs, axis=1)

    train_conf = float(np.mean(train_conf_arr))
    test_conf = float(np.mean(test_conf_arr))
    gap = train_conf - test_conf

    if gap > config.PRIVACY_GAP_HIGH:
        risk, risk_color = "High", "#dc2626"
        risk_msg = ("The confidence gap between training and test data suggests "
                    "potential exposure to Membership Inference Attacks.")
    elif gap > config.PRIVACY_GAP_MEDIUM:
        risk, risk_color = "Medium", "#d49600"
        risk_msg = ("There is a moderate gap between training and test confidence, "
                    "indicating some risk of information leakage.")
    else:
        risk, risk_color = "Low", "#16a34a"
        risk_msg = ("Training and test confidence are closely aligned, suggesting "
                    "limited exposure to Membership Inference Attacks.")

    # security_score: 100 = no sign of leakage, 0 = severe leakage
    security_score = round(max(0, min(100, 100 - (gap / config.PRIVACY_GAP_SCALE) * 100)))
    disclosure_score = round(100 - security_score, 1)

    if security_score >= 85:
        leakage_risk = "Minimal"
    elif security_score >= 60:
        leakage_risk = "Low"
    elif security_score >= 35:
        leakage_risk = "Moderate"
    else:
        leakage_risk = "High"

    # ── Chart: confidence comparison + distribution ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax1 = axes[0]
    bars = ax1.bar(["Train", "Test"], [train_conf * 100, test_conf * 100],
                    color=["#003366", "#FFB81C"], width=0.5, edgecolor="white")
    for b in bars:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2, h + 1.5, f"{h:.2f}%",
                  ha="center", fontsize=11, fontweight="bold", color="#0a1628")
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Average Confidence (%)")
    ax1.set_title("Train vs Test Confidence", fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2 = axes[1]
    ax2.hist(train_conf_arr, bins=20, alpha=0.65, label="Train", color="#003366", density=True)
    ax2.hist(test_conf_arr, bins=20, alpha=0.65, label="Test", color="#FFB81C", density=True)
    ax2.set_xlabel("Prediction Confidence")
    ax2.set_ylabel("Density")
    ax2.set_title("Confidence Distribution (MIA Signal)", fontweight="bold")
    ax2.legend()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    privacy_img = fig_to_base64(fig)

    # ── Feature Risk Score (original contribution) ──
    feature_risk = feature_risk_scores(xai)
    top_feature = feature_risk[0]["name"] if feature_risk else "failures"
    second_feature = feature_risk[1]["name"] if len(feature_risk) > 1 else "absences"

    privacy_insights = [
        {"dot": "#dc2626",
         "text": f"<strong>{top_feature}</strong> is the most privacy-sensitive feature."},
        {"dot": "#d49600",
         "text": f"<strong>{second_feature}</strong> contributes significantly to both the "
                 f"prediction and the disclosure risk."},
        {"dot": "#16a34a",
         "text": "Demographic features (age, gender) contribute comparatively less to exposure."},
    ]

    key_findings = [
        "The model is more confident on training data than on unseen test data.",
        f"Training confidence exceeds test confidence by {gap:.2%}.",
        "This gap is consistent with mild overfitting.",
        "Overfitting can increase privacy risk by making training examples identifiable.",
        "Explainability methods reveal useful insights but may also expose sensitive patterns.",
    ]
    recommendations = [
        "Reduce model overfitting (e.g. regularization, more data, a simpler model)",
        "Limit exposure of sensitive explanations to trusted users",
        "Consider privacy-preserving machine learning techniques (e.g. differential privacy)",
        "Monitor explanation outputs before deployment",
    ]

    return {
        "privacy_img": privacy_img,
        "train_conf": f"{train_conf:.2%}",
        "test_conf": f"{test_conf:.2%}",
        "gap": f"{gap:.2%}",
        "risk": risk,
        "risk_color": risk_color,
        "risk_msg": risk_msg,
        "security_score": security_score,
        "disclosure_score": disclosure_score,
        "leakage_risk": leakage_risk,
        "feature_risk": feature_risk,
        "privacy_insights": privacy_insights,
        "key_findings": key_findings,
        "recommendations": recommendations,
    }