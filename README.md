# XAI Explorer

**Comparing SHAP and LIME explainability methods on a student performance prediction model — with an integrated privacy-risk analysis layer.**

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://xai-explorer-cuwv.onrender.com)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)

**Live demo:** https://xai-explorer-cuwv.onrender.com

---

## Overview

XAI Explorer is a Flask-based web platform built as part of a diploma thesis on explainability and security in Machine Learning ("Shpjegueshmëria dhe Siguria në Machine Learning", University of Tetova, supervised by Prof. Dr. Florim Idrizi). The platform trains a Random Forest classifier on the UCI Student Performance dataset and lets users interactively compare how SHAP and LIME explain the same predictions — while also surfacing what those explanations reveal about privacy risk.

Most XAI tools stop at "here is which features mattered." This project goes a step further and asks: **what does an explanation leak about the data it was trained on, and does it matter which explainability method you trust?**

## Key Highlights

- **Global vs. local explainability compared side by side** — SHAP (TreeExplainer) and LIME (LimeTabularExplainer) are run on the same model and the same predictions, so their outputs can be directly compared rather than evaluated in isolation.
- **SHAP and LIME agree on direction, diverge on magnitude.** Across the top features, both methods point the same way most of the time — but the size of the effect they report can differ by several times over, which has real implications for which method you'd trust in a high-stakes setting.
- **Original contribution — Feature Risk Score:** a metric derived from SHAP's global importance values, used to flag which features carry the most explanatory (and therefore privacy-relevant) weight.
- **Original contribution — Membership Inference Attack simulation:** the platform compares the model's confidence on training vs. test data to simulate how easily an attacker could infer whether a specific record was used to train the model — turning an abstract privacy concept into something visible and interactive.
- **PDF report generation** — users can export a full explainability + privacy analysis report generated on demand via ReportLab.
- Built from scratch as a 7-page Flask application, with all SHAP/LIME plots generated in-memory and served as base64 (no static plot files, faster iteration, no stale images).

## Screenshots

![Home](docs/screenshots/home.png)

![SHAP Summary](docs/screenshots/shap_summary.png)

![SHAP vs LIME Comparison](docs/screenshots/comparison.png)

![Privacy & Security Analysis](docs/screenshots/privacy_analysis.png)

## Methodology

- **Dataset:** UCI Student Performance Dataset (Cortez & Silva), 395 students, 30 features.
- **Task framing:** reformulated from regression (final grade 0–20) to binary classification (pass/fail, threshold ≥10), with an 80/20 stratified split.
- **Model:** Random Forest Classifier (scikit-learn). The gap between training and test confidence was used deliberately as the basis for the Membership Inference Attack simulation, rather than treated purely as a flaw to eliminate.
- **Explainability:** SHAP TreeExplainer for global feature importance; LIME for per-instance local explanations, recomputed per request.

## Tech Stack

**Backend:** Python, Flask, scikit-learn, SHAP, LIME, ReportLab
**Data/Viz:** pandas, NumPy, Matplotlib
**Frontend:** vanilla HTML/CSS/JavaScript (no framework)
**Deployment:** Render

## Running Locally

```bash
git clone https://github.com/remzije-sulejmanoska/xai-explorer.git
cd xai-explorer
pip install -r requirements.txt
python app.py
```

The app will be available at `http://localhost:5000`.

## Author

**Remzije Sulejmanoska**
Computer Science graduate — Machine Learning, Explainable AI & Trustworthy AI
University of Tetova, Faculty of Natural Sciences and Mathematics

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
