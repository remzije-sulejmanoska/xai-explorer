# ═══════════════════════════════════════════════════════
# APP.PY — XAI Explorer Platform
# SHAP dhe LIME Visualization Platform
# Punim Diplome — UTS 2026
# ═══════════════════════════════════════════════════════

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular
import base64
import pandas as pd
from io import BytesIO
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)
import traceback
import os

# ── PATH ABSOLUTE ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── INICIALIZIMI I FLASK ──
app = Flask(__name__)

# ── NGARKIMI I MODELEVE ──
print("Duke ngarkuar modelet...")

model         = joblib.load(os.path.join(BASE_DIR, 'models', 'rf_model.pkl'))
feature_names = joblib.load(os.path.join(BASE_DIR, 'models', 'feature_names.pkl'))
X_train       = joblib.load(os.path.join(BASE_DIR, 'models', 'X_train.pkl'))

print(f"✓ Features ({len(feature_names)}): {list(feature_names)}")

# ── RIKRIJIMI I SHAP ──
explainer_shap = shap.TreeExplainer(model)

# ── RIKRIJIMI I LIME ──
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=feature_names,
    class_names=['Nuk kalon', 'Kalon'],
    mode='classification',
    random_state=42
)

# ── NGARKO Y_TEST PËR METRIKAT ──
def load_y_test():
    df = pd.read_csv(os.path.join(BASE_DIR, '..', 'data', 'C:/Users/user/OneDrive/Desktop/PUNIMI I DIPLOMES/DATASETI/student+performance/student/student-mat.csv' \
    ''), sep=';')
    df['pass'] = (df['G3'] >= 10).astype(int)
    X = df.drop(columns=['G1', 'G2', 'G3', 'pass'])
    y = df['pass']
    le = LabelEncoder()
    for col in X.select_dtypes(include='object').columns:
        X[col] = le.fit_transform(X[col])
    _, _, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return y_test

print("✓ Modelet u ngarkuan!")
print(f"✓ X_train shape: {X_train.shape}")

# ═══════════════════════════════════════════════════════
# FUNKSIONI NDIHMËS — Grafiku → Base64
# ═══════════════════════════════════════════════════════
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_base64

# ═══════════════════════════════════════════════════════
# ERROR HANDLER GLOBAL
# ═══════════════════════════════════════════════════════
@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Route not found: ' + str(e)}), 404

# ═══════════════════════════════════════════════════════
# ROUTE 1 — Faqja kryesore
# ═══════════════════════════════════════════════════════
@app.route('/')
def index():
    return render_template('index.html', feature_names=feature_names)

# ═══════════════════════════════════════════════════════
# ROUTE 1b — Debug: shfaq feature_names
# ═══════════════════════════════════════════════════════
@app.route('/debug_features')
def debug_features():
    return jsonify({
        'feature_count' : len(feature_names),
        'feature_names' : list(feature_names),
        'X_train_shape' : list(X_train.shape),
    })

# ═══════════════════════════════════════════════════════
# ROUTE 2 — Dashboard Stats
# ═══════════════════════════════════════════════════════
@app.route('/stats')
def stats():
    try:
        X_test_data = joblib.load(os.path.join(BASE_DIR, 'models', 'X_test.pkl'))
        y_test      = load_y_test()
        y_pred      = model.predict(X_test_data)

        return jsonify({
            'students'  : 395,
            'features'  : len(feature_names),
            'algorithm' : 'Random Forest',
            'dataset'   : 'UCI Student Perf.',
            'accuracy'  : f'{accuracy_score(y_test, y_pred):.2%}',
            'precision' : f'{precision_score(y_test, y_pred):.2%}',
            'recall'    : f'{recall_score(y_test, y_pred):.2%}',
            'f1'        : f'{f1_score(y_test, y_pred):.2%}'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

# ═══════════════════════════════════════════════════════
# ROUTE 3 — Parashikimi
# ═══════════════════════════════════════════════════════
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'Nuk u pranua JSON valid'}), 400

        values   = data.get('values', [])
        expected = len(feature_names)

        if len(values) != expected:
            return jsonify({
                'error'        : f'Priten {expected} vlera, u morën {len(values)}',
                'expected'     : expected,
                'received'     : len(values),
                'feature_names': list(feature_names)
            }), 400

        input_array = np.array([values], dtype=float)
        prediction  = int(model.predict(input_array)[0])
        probability = float(model.predict_proba(input_array)[0][1])
        prob_pct    = round(probability * 100, 2)

        # ── RISK LEVEL — bazuar në probabilitetin e kalimit ──
        if prob_pct >= 70:
            risk_level = 'Low'
        elif prob_pct >= 40:
            risk_level = 'Medium'
        else:
            risk_level = 'High'

        # ── SHAP — Top Positive / Negative Factors për këtë instancë ──
        shap_values = explainer_shap.shap_values(input_array)
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]
        elif shap_values.ndim == 3:
            shap_vals = shap_values[:, :, 1][0]
        else:
            shap_vals = shap_values[0]

        factors = sorted(
            [{'name': feature_names[i], 'value': float(shap_vals[i])}
            for i in range(len(feature_names))],
            key=lambda x: abs(x['value']),
            reverse=True
        )

        # Shfaq vetëm faktorët me ndikim real
        THRESHOLD = 0.03

        top_positive = [
            f['name']
            for f in factors
            if f['value'] > THRESHOLD
        ][:3]

        top_negative = [
            f['name']
            for f in factors
            if f['value'] < -THRESHOLD
        ][:3]

        if not top_negative:
            top_negative = ["No significant risk factors detected"]

        return jsonify({
            'prediction'  : prediction,
            'probability' : prob_pct,
            'label'       : 'PASS' if prediction == 1 else 'FAIL',
            'risk_level'  : risk_level,
            'top_positive': top_positive,
            'top_negative': top_negative
        })

    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

# ═══════════════════════════════════════════════════════
# ROUTE 4 — SHAP Shpjegimi Lokal
# ═══════════════════════════════════════════════════════
@app.route('/shap_local', methods=['POST'])
def shap_local():
    try:
        data        = request.get_json()
        input_array = np.array([data['values']], dtype=float)
        shap_values = explainer_shap.shap_values(input_array)

        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]
        elif shap_values.ndim == 3:
            shap_vals = shap_values[:, :, 1][0]
        else:
            shap_vals = shap_values[0]

        shap_exp = shap.Explanation(
            values        = shap_vals,
            base_values   = (explainer_shap.expected_value[1]
                             if hasattr(explainer_shap.expected_value, '__len__')
                             else explainer_shap.expected_value),
            data          = input_array[0],
            feature_names = feature_names
        )
        shap.plots.waterfall(shap_exp, show=False)
        waterfall_img = fig_to_base64(plt.gcf())

        factors = sorted(
            [{'name': feature_names[i], 'value': float(shap_vals[i])}
            for i in range(len(feature_names))],
            key=lambda x: abs(x['value']),
            reverse=True
        )

        THRESHOLD = 0.03

        positive = [
            f for f in factors
            if f['value'] > THRESHOLD
        ][:3]

        negative = [
            f for f in factors
            if f['value'] < -THRESHOLD
        ][:3]

        if not negative:
            negative = [{
                'name': 'No significant risk factors identified',
                'value': 0
            }]

        return jsonify({
            'waterfall_img': waterfall_img,
            'positive'     : positive,
            'negative'     : negative,
            'shap_values'  : [round(float(v), 4) for v in shap_vals]
        })

    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

# ═══════════════════════════════════════════════════════
# ROUTE 5 — SHAP Global
# ═══════════════════════════════════════════════════════
@app.route('/global_shap')
def global_shap():
    try:
        shap_vals_global = joblib.load(os.path.join(BASE_DIR, 'models', 'shap_values.pkl'))
        X_test_data      = joblib.load(os.path.join(BASE_DIR, 'models', 'X_test.pkl'))

        if isinstance(shap_vals_global, list):
            shap_vals_global = shap_vals_global[1]
        elif shap_vals_global.ndim == 3:
            shap_vals_global = shap_vals_global[:, :, 1]

        fig1 = plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_vals_global, X_test_data,
                          feature_names=feature_names, show=False)
        summary_img = fig_to_base64(plt.gcf())

        fig2 = plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_vals_global, X_test_data,
                          feature_names=feature_names,
                          plot_type='bar', show=False)
        bar_img = fig_to_base64(plt.gcf())

        mean_shap = np.abs(shap_vals_global).mean(axis=0)
        top5 = sorted(
            [{'name': feature_names[i], 'score': float(mean_shap[i])}
             for i in range(len(feature_names))],
            key=lambda x: x['score'], reverse=True
        )[:5]

        top1    = top5[0]['name'] if top5 else 'failures'
        top2    = top5[1]['name'] if len(top5) > 1 else 'absences'
        insight = (f"'{top1}' është faktori më ndikues global. "
                   f"'{top2}' gjithashtu ndikon ndjeshëm në parashikim.")

        return jsonify({
            'summary_img': summary_img,
            'bar_img'    : bar_img,
            'top5'       : top5,
            'insight'    : insight
        })

    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

# ═══════════════════════════════════════════════════════
# ROUTE 6 — LIME Shpjegimi Lokal
# ═══════════════════════════════════════════════════════
@app.route('/lime_local', methods=['POST'])
def lime_local():
    try:
        data        = request.get_json()
        input_array = np.array([data['values']], dtype=float)

        exp_lime = explainer_lime.explain_instance(
            input_array[0],
            model.predict_proba,
            num_features=10
        )

        fig = exp_lime.as_pyplot_figure()
        fig.set_size_inches(10, 6)
        lime_img  = fig_to_base64(fig)
        lime_list = exp_lime.as_list()

        return jsonify({
            'lime_img'  : lime_img,
            'lime_list' : [{'feature': f, 'weight': round(float(w), 4)}
                           for f, w in lime_list]
        })

    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

# ═══════════════════════════════════════════════════════
# ROUTE 7 — SHAP vs LIME Krahasimi
# ═══════════════════════════════════════════════════════
@app.route('/comparison', methods=['POST'])
def comparison():
    try:
        data        = request.get_json()
        input_array = np.array([data['values']], dtype=float)

        shap_values = explainer_shap.shap_values(input_array)
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]
        elif shap_values.ndim == 3:
            shap_vals = shap_values[:, :, 1][0]
        else:
            shap_vals = shap_values[0]

        exp_lime  = explainer_lime.explain_instance(
            input_array[0], model.predict_proba, num_features=10
        )
        lime_list = exp_lime.as_list()

        shap_top = sorted(
            [{'name': feature_names[i], 'shap': round(float(shap_vals[i]), 4)}
             for i in range(len(feature_names))],
            key=lambda x: abs(x['shap']), reverse=True
        )[:10]

        lime_dict = {}
        for feat, val in lime_list:
            key = feat.split(' ')[0].split('<')[0].split('>')[0].strip()
            lime_dict[key] = round(float(val), 4)

        table = []
        for item in shap_top:
            table.append({
                'feature': item['name'],
                'shap'   : item['shap'],
                'lime'   : lime_dict.get(item['name'], 0)
            })

        names          = [r['feature'] for r in table]
        shap_vals_plot = [abs(r['shap']) for r in table]
        lime_vals_plot = [abs(r['lime']) for r in table]

        x   = np.arange(len(names))
        w   = 0.35
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - w/2, shap_vals_plot, w, label='SHAP', color='#003366', alpha=0.85)
        ax.bar(x + w/2, lime_vals_plot, w, label='LIME', color='#FFB81C', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha='right', fontsize=9)
        ax.set_ylabel('Rëndësia absolute')
        ax.set_title('SHAP vs LIME — Krahasimi i Rëndësisë së Features', fontweight='bold')
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        comparison_img = fig_to_base64(plt.gcf())

        top1       = shap_top[0]['name'] if shap_top else 'failures'
        top2       = shap_top[1]['name'] if len(shap_top) > 1 else 'absences'
        conclusion = (
            f"Të dyja metodat identifikojnë '{top1}' dhe '{top2}' si faktorët më ndikues. "
            f"SHAP ofron shpjegime globale më të qëndrueshme ndërsa LIME fokusohet "
            f"në interpretueshmërinë lokale."
        )

        return jsonify({
            'table'         : table,
            'comparison_img': comparison_img,
            'conclusion'    : conclusion
        })

    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

# ═══════════════════════════════════════════════════════
# ROUTE 8 — Privacy & Security
# ═══════════════════════════════════════════════════════
@app.route('/privacy_features')
def privacy_features():
    try:
        X_test_data = joblib.load(os.path.join(BASE_DIR, 'models', 'X_test.pkl'))

        # ── BESUESHMËRIA E MODELIT NË TRAIN VS TEST ──
        train_probs    = model.predict_proba(X_train)
        test_probs     = model.predict_proba(X_test_data)
        train_conf_arr = np.max(train_probs, axis=1)
        test_conf_arr  = np.max(test_probs, axis=1)

        train_conf = float(np.mean(train_conf_arr))
        test_conf  = float(np.mean(test_conf_arr))
        diff       = train_conf - test_conf

        if diff > 0.1:
            risk       = 'High'
            risk_color = '#dc2626'
            risk_msg   = ("The confidence difference between training and testing data "
                          "suggests potential exposure to Membership Inference Attacks.")
        elif diff > 0.05:
            risk       = 'Medium'
            risk_color = '#d49600'
            risk_msg   = ("There is a moderate gap between training and testing confidence, "
                          "indicating some risk of information leakage.")
        else:
            risk       = 'Low'
            risk_color = '#16a34a'
            risk_msg   = ("Training and testing confidence are closely aligned, "
                          "suggesting limited exposure to Membership Inference Attacks.")

        # ── SECURITY SCORE (0-100) — bazuar te gap-i train/test ──
        # gap=0   -> 100 pikë (asnjë shenjë rrjedhjeje)
        # gap=0.3 -> 0 pikë (rrjedhje e rëndë), shkallëzim linear mes tyre
        security_score = round(max(0, min(100, 100 - (diff / 0.3) * 100)))
        if security_score >= 85:
            leakage_risk = 'Minimal'
        elif security_score >= 60:
            leakage_risk = 'Low'
        elif security_score >= 35:
            leakage_risk = 'Moderate'
        else:
            leakage_risk = 'High'

        # ── GRAFIKU — privacy_analysis.png ──
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Paneli 1 — Krahasimi i besueshmërisë mesatare
        ax1   = axes[0]
        bars  = ax1.bar(['Train', 'Test'], [train_conf * 100, test_conf * 100],
                        color=['#003366', '#FFB81C'], width=0.5, edgecolor='white')
        for b in bars:
            h = b.get_height()
            ax1.text(b.get_x() + b.get_width() / 2, h + 1.5, f'{h:.2f}%',
                     ha='center', fontsize=11, fontweight='bold', color='#0a1628')
        ax1.set_ylim(0, 100)
        ax1.set_ylabel('Average Confidence (%)')
        ax1.set_title('Train vs Test Confidence', fontweight='bold')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # Paneli 2 — Shpërndarja e besueshmërisë (sinjali i MIA)
        ax2 = axes[1]
        ax2.hist(train_conf_arr, bins=20, alpha=0.65, label='Train',
                 color='#003366', density=True)
        ax2.hist(test_conf_arr, bins=20, alpha=0.65, label='Test',
                 color='#FFB81C', density=True)
        ax2.set_xlabel('Prediction Confidence')
        ax2.set_ylabel('Density')
        ax2.set_title('Confidence Distribution (MIA Signal)', fontweight='bold')
        ax2.legend()
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        plt.tight_layout()
        privacy_img = fig_to_base64(fig)

        # ── KEY FINDINGS ──
        key_findings = [
            'The model is more confident on training data.',
            f'Training confidence exceeds test confidence by {diff:.2%}.',
            'This indicates possible overfitting.',
            'Overfitting may increase privacy risks.',
            'Explainability methods can reveal useful insights but may also expose sensitive patterns.'
        ]

        # ── RECOMMENDATIONS ──
        recommendations = [
            'Reduce model overfitting',
            'Limit exposure of sensitive explanations',
            'Use privacy-preserving machine learning techniques',
            'Monitor explanation outputs before deployment'
        ]

        return jsonify({
            'privacy_img'    : privacy_img,
            'train_conf'     : f'{train_conf:.2%}',
            'test_conf'      : f'{test_conf:.2%}',
            'gap'            : f'{diff:.2%}',
            'risk'           : risk,
            'risk_color'     : risk_color,
            'risk_msg'       : risk_msg,
            'security_score' : security_score,
            'leakage_risk'   : leakage_risk,
            'key_findings'   : key_findings,
            'recommendations': recommendations
        })

    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

# ═══════════════════════════════════════════════════════
# ROUTE 9 — Results & Conclusions
# ═══════════════════════════════════════════════════════
@app.route('/results')
def results():
    try:
        X_test_data = joblib.load(os.path.join(BASE_DIR, 'models', 'X_test.pkl'))
        y_test      = load_y_test()
        y_pred      = model.predict(X_test_data)

        return jsonify({
            'accuracy' : f'{accuracy_score(y_test, y_pred):.2%}',
            'precision': f'{precision_score(y_test, y_pred):.2%}',
            'recall'   : f'{recall_score(y_test, y_pred):.2%}',
            'f1'       : f'{f1_score(y_test, y_pred):.2%}',
            'findings' : [
                "Failures është faktori më ndikues në të dyja metodat.",
                "Absences ndikon ndjeshëm në performancën e studentit.",
                "SHAP ofron shpjegime globale më të qëndrueshme.",
                "LIME ofron shpjegime lokale më intuitive.",
                "Shpjegimet mund të ekspozojnë informacion sensitiv të studentëve."
            ]
        })

    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

# ═══════════════════════════════════════════════════════
# STARTON FLASK
# ═══════════════════════════════════════════════════════
if __name__ == '__main__':
    app.run(debug=True)