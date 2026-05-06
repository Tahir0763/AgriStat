"""
Crop Yield Statistical Analysis Web App
Probability & Statistics Semester Project - Spring 2026
"""

from flask import Flask, jsonify, render_template, request
import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ── Load & clean data ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'yield_df.csv'))
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.columns = [c.strip() for c in df.columns]
df = df.loc[:, [c for c in df.columns if c]]
df.rename(columns={
    'hg/ha_yield': 'Yield_hg_ha',
    'average_rain_fall_mm_per_year': 'Rainfall_mm',
    'pesticides_tonnes': 'Pesticides_tonnes',
    'avg_temp': 'Avg_Temp'
}, inplace=True)
df.dropna(inplace=True)

CROPS = sorted(df['Item'].unique().tolist())
AREAS = sorted(df['Area'].unique().tolist())
YEARS = sorted(df['Year'].unique().tolist())
NUM_COLS = ['Yield_hg_ha', 'Rainfall_mm', 'Pesticides_tonnes', 'Avg_Temp']

PRED_NUM_COLS = ['Rainfall_mm', 'Pesticides_tonnes', 'Avg_Temp', 'Year']
PRED_CAT_COLS = ['Item', 'Area']

df_model = pd.get_dummies(df[PRED_NUM_COLS + PRED_CAT_COLS], columns=PRED_CAT_COLS)
MODEL_FEATURES = df_model.columns.tolist()
PRED_MODEL = LinearRegression().fit(df_model, df['Yield_hg_ha'])

# ── helpers ────────────────────────────────────────────────────────────────────
def safe(v):
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    return v

def series_stats(s):
    s = s.dropna()
    n = len(s)
    if n == 0:
        return {}
    mean = float(s.mean())
    std  = float(s.std())
    se   = std / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n-1)
    ci_lo  = mean - t_crit * se
    ci_hi  = mean + t_crit * se
    return {
        "n": n, "mean": safe(mean), "median": safe(float(s.median())),
        "std": safe(std), "var": safe(float(s.var())),
        "min": safe(float(s.min())), "max": safe(float(s.max())),
        "q1": safe(float(s.quantile(0.25))), "q3": safe(float(s.quantile(0.75))),
        "skew": safe(float(s.skew())), "kurt": safe(float(s.kurt())),
        "ci_lo": safe(ci_lo), "ci_hi": safe(ci_hi), "se": safe(se)
    }

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html',
                           crops=CROPS, areas=AREAS, years=YEARS)

# 1. Dataset overview
@app.route('/api/overview')
def overview():
    return jsonify({
        "rows": len(df), "cols": len(df.columns),
        "crops": len(CROPS), "countries": len(AREAS),
        "year_start": int(min(YEARS)), "year_end": int(max(YEARS)),
        "columns": df.columns.tolist()
    })

# 2. Descriptive statistics
@app.route('/api/descriptive')
def descriptive():
    crop = request.args.get('crop', 'All')
    area = request.args.get('area', 'All')
    sub  = df.copy()
    if crop != 'All': sub = sub[sub['Item'] == crop]
    if area != 'All': sub = sub[sub['Area'] == area]
    result = {col: series_stats(sub[col]) for col in NUM_COLS}
    return jsonify(result)

# 4. Crop comparison (bar)
@app.route('/api/crop_compare')
def crop_compare():
    grp = df.groupby('Item')['Yield_hg_ha'].mean().reset_index().sort_values('Yield_hg_ha', ascending=False)
    return jsonify({
        "crops":  grp['Item'].tolist(),
        "yields": [safe(v) for v in grp['Yield_hg_ha'].tolist()]
    })

# 5. Correlation matrix
@app.route('/api/correlation')
def correlation():
    corr = df[NUM_COLS].corr().round(3)
    return jsonify({
        "labels": NUM_COLS,
        "matrix": corr.values.tolist()
    })

# 6. Distribution (histogram data)
@app.route('/api/distribution')
def distribution():
    col  = request.args.get('col', 'Yield_hg_ha')
    crop = request.args.get('crop', 'All')
    sub  = df.copy()
    if crop != 'All': sub = sub[sub['Item'] == crop]
    s    = sub[col].dropna()
    counts, edges = np.histogram(s, bins=30)
    centers = ((edges[:-1] + edges[1:]) / 2).tolist()

    # Normal fit
    mu, sigma = float(s.mean()), float(s.std())
    x_range   = np.linspace(float(s.min()), float(s.max()), 100)
    pdf_vals  = stats.norm.pdf(x_range, mu, sigma) * len(s) * (edges[1]-edges[0])

    # Normality test (Shapiro if n<=5000 else KS)
    if len(s) <= 5000:
        stat, p = stats.shapiro(s.sample(min(len(s), 5000), random_state=42))
        test_name = "Shapiro-Wilk"
    else:
        stat, p = stats.kstest(s, 'norm', args=(mu, sigma))
        test_name = "KS Test"

    return jsonify({
        "centers": [safe(v) for v in centers],
        "counts":  counts.tolist(),
        "pdf_x":   x_range.tolist(),
        "pdf_y":   [safe(v) for v in pdf_vals.tolist()],
        "mu": safe(mu), "sigma": safe(sigma),
        "test": test_name, "stat": safe(float(stat)), "p": safe(float(p))
    })

# 7. Probability (P(X > threshold))
@app.route('/api/probability')
def probability():
    col   = request.args.get('col', 'Yield_hg_ha')
    crop  = request.args.get('crop', 'All')
    thresh = float(request.args.get('threshold', df[col].mean()))
    sub   = df.copy()
    if crop != 'All': sub = sub[sub['Item'] == crop]
    s     = sub[col].dropna()
    mu, sigma = float(s.mean()), float(s.std())
    z     = (thresh - mu) / sigma
    p_gt  = float(1 - stats.norm.cdf(z))
    p_lt  = float(stats.norm.cdf(z))
    emp_gt = float((s > thresh).mean())
    return jsonify({
        "threshold": thresh, "mu": safe(mu), "sigma": safe(sigma),
        "z_score": safe(z),
        "p_greater": safe(p_gt), "p_less": safe(p_lt),
        "empirical_greater": safe(emp_gt)
    })

# 8. Regression
@app.route('/api/regression')
def regression():
    crop   = request.args.get('crop', 'Wheat')
    x_col  = request.args.get('x', 'Rainfall_mm')
    degree = int(request.args.get('degree', 1))
    sub    = df[df['Item'] == crop][[x_col, 'Yield_hg_ha']].dropna()
    if len(sub) < 10:
        return jsonify({"error": "Not enough data"})

    X = sub[[x_col]].values
    y = sub['Yield_hg_ha'].values

    pf   = PolynomialFeatures(degree=degree, include_bias=False)
    Xp   = pf.fit_transform(X)
    reg  = LinearRegression().fit(Xp, y)
    y_pred = reg.predict(Xp)
    r2   = float(r2_score(y, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))

    x_line = np.linspace(float(X.min()), float(X.max()), 200).reshape(-1,1)
    y_line = reg.predict(pf.transform(x_line))

    return jsonify({
        "x_data":  [safe(v) for v in X.flatten().tolist()],
        "y_data":  [safe(v) for v in y.tolist()],
        "x_line":  x_line.flatten().tolist(),
        "y_line":  [safe(v) for v in y_line.tolist()],
        "r2": safe(r2), "rmse": safe(rmse),
        "coef": [safe(float(c)) for c in reg.coef_],
        "intercept": safe(float(reg.intercept_)),
        "x_label": x_col, "crop": crop, "degree": degree
    })

# 9. Prediction using all parameters
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json or {}
    try:
        row = {
            'Item': data['crop'],
            'Area': data['area'],
            'Year': int(data['year']),
            'Rainfall_mm': float(data['rainfall']),
            'Pesticides_tonnes': float(data['pesticides']),
            'Avg_Temp': float(data['avg_temp'])
        }
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({'error': f'Invalid input: {exc}'}), 400

    row_df = pd.DataFrame([row])
    row_df = pd.get_dummies(row_df, columns=PRED_CAT_COLS)
    row_df = row_df.reindex(columns=MODEL_FEATURES, fill_value=0)

    pred = float(PRED_MODEL.predict(row_df)[0])
    return jsonify({'prediction': round(pred, 2)})

# 10. Country ranking
@app.route('/api/country_rank')
def country_rank():
    crop = request.args.get('crop', 'Wheat')
    sub  = df[df['Item'] == crop]
    grp  = sub.groupby('Area')['Yield_hg_ha'].mean().reset_index()
    grp  = grp.sort_values('Yield_hg_ha', ascending=False).head(15)
    return jsonify({
        "areas":  grp['Area'].tolist(),
        "yields": [safe(v) for v in grp['Yield_hg_ha'].tolist()]
    })

# 11. Scatter: two variables
@app.route('/api/scatter')
def scatter():
    x_col = request.args.get('x', 'Avg_Temp')
    y_col = request.args.get('y', 'Yield_hg_ha')
    crop  = request.args.get('crop', 'Wheat')
    sub   = df[df['Item'] == crop][[x_col, y_col]].dropna().sample(min(500, len(df)), random_state=1)
    return jsonify({
        "x": [safe(v) for v in sub[x_col].tolist()],
        "y": [safe(v) for v in sub[y_col].tolist()],
        "x_label": x_col, "y_label": y_col
    })

# 12. Rainfall vs Yield heatmap bins
@app.route('/api/heatmap')
def heatmap():
    sub = df[['Rainfall_mm','Avg_Temp','Yield_hg_ha']].dropna()
    sub['rain_bin'] = pd.cut(sub['Rainfall_mm'], bins=8, labels=False)
    sub['temp_bin'] = pd.cut(sub['Avg_Temp'], bins=8, labels=False)
    piv = sub.groupby(['temp_bin','rain_bin'])['Yield_hg_ha'].mean().unstack(fill_value=0)
    return jsonify({
        "matrix": piv.values.tolist(),
        "rows": piv.index.astype(str).tolist(),
        "cols": piv.columns.astype(str).tolist()
    })

if __name__ == '__main__':
    app.run(debug=True, port=5050)