"""
config.py — Central configuration for the XAI Explorer platform.

Every file path, preprocessing parameter and threshold used by the
application lives here, so nothing is hardcoded or duplicated across
ml_pipeline.py / app.py, and so the values can never silently drift
from the values used in notebooks/eda-and-model-training.ipynb.
"""

import os

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "student-mat.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")

# ─────────────────────────────────────────────────────────────
# Preprocessing parameters
# These MUST match notebooks/eda-and-model-training.ipynb exactly,
# since the trained model (rf_model.pkl) was fit on data produced by
# that exact pipeline.
# ─────────────────────────────────────────────────────────────
TARGET_COLUMN = "G3"
PASS_THRESHOLD = 10          # a student passes if G3 >= 10
LEAKAGE_COLUMNS = ["G1", "G2", "G3"]  # excluded from features (grade leakage)
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ─────────────────────────────────────────────────────────────
# Explainability settings
# ─────────────────────────────────────────────────────────────
SHAP_FACTOR_THRESHOLD = 0.03   # minimum |SHAP| value to count as a notable factor
SHAP_TOP_N = 3                 # top positive/negative factors shown per prediction
LIME_NUM_FEATURES = 10
FEATURE_RISK_TOP_N = 10

# ─────────────────────────────────────────────────────────────
# Prediction risk bands (probability of passing, in %)
# ─────────────────────────────────────────────────────────────
RISK_LOW_PROB = 70    # probability >= 70%  -> Low risk
RISK_HIGH_PROB = 40   # probability <  40%  -> High risk (else Medium)

# ─────────────────────────────────────────────────────────────
# Privacy analysis thresholds (train/test confidence gap)
# ─────────────────────────────────────────────────────────────
PRIVACY_GAP_HIGH = 0.10
PRIVACY_GAP_MEDIUM = 0.05
PRIVACY_GAP_SCALE = 0.30   # gap at which security_score hits 0 / disclosure_score hits 100

# ─────────────────────────────────────────────────────────────
# Static content shown on the Results page
# ─────────────────────────────────────────────────────────────
RESULT_FINDINGS = [
    "Failures is the most influential factor identified by both methods.",
    "Absences has a significant impact on predicted student performance.",
    "SHAP provides more consistent, stable global explanations.",
    "LIME offers more intuitive, easy-to-read local explanations.",
    "Model explanations can potentially expose sensitive information about students.",
]

# ─────────────────────────────────────────────────────────────
# Flask
# ─────────────────────────────────────────────────────────────
# Debug mode is OFF by default. Enable locally with:
#   export FLASK_DEBUG=true        (macOS/Linux)
#   set FLASK_DEBUG=true           (Windows)
DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() == "true"