import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard | Bank Customer Churn Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

* { font-family: 'DM Sans', sans-serif; }
h1, h2, h3, .big-title { font-family: 'Syne', sans-serif; }

/* Background */
.stApp { background: #0a0e1a; }
section[data-testid="stSidebar"] { background: #0f1322 !important; border-right: 1px solid #1e2540; }

/* Main title block */
.hero-block {
    background: linear-gradient(135deg, #1a1f35 0%, #0f1322 100%);
    border: 1px solid #2a3158;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-block::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(99,179,237,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #e2e8f0;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    color: #64748b;
    font-size: 0.95rem;
    font-weight: 300;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,179,237,0.12);
    color: #63b3ed;
    border: 1px solid rgba(99,179,237,0.25);
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

/* Metric cards */
.metric-card {
    background: #111827;
    border: 1px solid #1e2540;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #3a4568; }
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #63b3ed;
}
.metric-label { color: #64748b; font-size: 0.8rem; font-weight: 500; letter-spacing: 0.5px; text-transform: uppercase; }

/* Result block */
.result-churn {
    background: linear-gradient(135deg, #2d1515 0%, #1a0e0e 100%);
    border: 1px solid #7f1d1d;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-safe {
    background: linear-gradient(135deg, #0f2d1a 0%, #0a1f12 100%);
    border: 1px solid #14532d;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    margin: 0.5rem 0;
}
.result-sublabel { color: #94a3b8; font-size: 0.9rem; }

/* Section headers */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #94a3b8;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 1.5rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e2540;
}

/* Sidebar labels */
[data-testid="stSidebar"] label {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.3px;
}
[data-testid="stSidebar"] .stSlider p { color: #63b3ed !important; font-weight: 600; }

/* Hide streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Data loader ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Try multiple public mirrors of the Kaggle Churn_Modelling dataset."""
    urls = [
        "https://raw.githubusercontent.com/selva86/datasets/master/Churn_Modelling.csv",
        "https://raw.githubusercontent.com/2blam/ML/master/deep_learning/Churn_Modelling.csv",
    ]
    required = {'CreditScore', 'Geography', 'Gender', 'Age', 'Exited'}
    for url in urls:
        try:
            tmp = pd.read_csv(url)
            tmp.columns = [c.strip().replace(" ", "") for c in tmp.columns]
            tmp = tmp.drop(columns=[c for c in ['RowNumber','CustomerId','Surname'] if c in tmp.columns], errors='ignore')
            # Validate it is actually the bank churn CSV (not an HTML 404 page)
            if not required.issubset(set(tmp.columns)):
                continue
            return tmp, True   # real data
        except Exception:
            continue
    return None, False            # will fall back to synthetic


# ── Model training (cached) ────────────────────────────────────────────────────
@st.cache_resource
def train_model():
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

    raw_df, is_real = load_data()

    if is_real:
        df = raw_df.copy()
    else:
        # ── Fallback: high-fidelity synthetic data ──────────────────────────
        np.random.seed(42)
        n = 10000
        geography = np.random.choice(['France','Germany','Spain'], n, p=[0.50,0.25,0.25])
        gender    = np.random.choice(['Male','Female'], n, p=[0.55,0.45])
        age       = np.clip(np.random.normal(38,11,n), 18, 92).astype(int)
        credit_score    = np.clip(np.random.normal(650,97,n), 350, 850).astype(int)
        tenure          = np.random.randint(0,11,n)
        balance         = np.where(np.random.rand(n)<0.3, 0, np.random.normal(76000,62000,n))
        balance         = np.clip(balance, 0, 250000)
        num_products    = np.random.choice([1,2,3,4], n, p=[0.50,0.46,0.03,0.01])
        has_cr_card     = np.random.choice([0,1], n, p=[0.30,0.70])
        is_active_member= np.random.choice([0,1], n, p=[0.49,0.51])
        estimated_salary= np.random.uniform(11,199992,n)
        churn_prob = np.clip(
            0.05 + 0.15*(geography=='Germany') + 0.10*(gender=='Female')
            + 0.008*np.maximum(age-40,0) - 0.005*tenure
            + 0.10*(balance>0) + 0.15*(num_products>=3)
            - 0.08*is_active_member - 0.0001*(credit_score-350),
            0.03, 0.90)
        exited = np.random.binomial(1, churn_prob, n)
        df = pd.DataFrame({
            'CreditScore': credit_score, 'Geography': geography, 'Gender': gender,
            'Age': age, 'Tenure': tenure, 'Balance': balance,
            'NumOfProducts': num_products, 'HasCrCard': has_cr_card,
            'IsActiveMember': is_active_member, 'EstimatedSalary': estimated_salary,
            'Exited': exited
        })

    # Encode
    le_geo = LabelEncoder(); le_gen = LabelEncoder()
    df['Geography_enc'] = le_geo.fit_transform(df['Geography'])
    df['Gender_enc']    = le_gen.fit_transform(df['Gender'])

    features = ['CreditScore', 'Geography_enc', 'Gender_enc', 'Age', 'Tenure',
                'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    X = df[features]
    y = df['Exited']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # Train multiple models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }

    results = {}
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        results[name] = {
            'accuracy': round(accuracy_score(y_test, preds) * 100, 2),
            'auc': round(roc_auc_score(y_test, proba) * 100, 2),
            'f1': round(f1_score(y_test, preds) * 100, 2),
        }
        trained[name] = model

    best_model = trained['Random Forest']
    return best_model, scaler, le_geo, le_gen, results, df, features, is_real


# ── Load model ─────────────────────────────────────────────────────────────────
with st.spinner("Loading dataset & training models..."):
    model, scaler, le_geo, le_gen, model_results, df, features, is_real = train_model()


# ── Hero header ────────────────────────────────────────────────────────────────
data_badge = "🟢 Live Kaggle Dataset · 10,000 rows" if is_real else "🟡 Synthetic Data (offline fallback)"
st.markdown(f"""
<div class="hero-block">
    <div class="hero-badge">ML · Classification · v1.0 &nbsp;|&nbsp; {data_badge}</div>
    <div class="hero-title">🏦 ChurnGuard</div>
    <p class="hero-subtitle">Bank Customer Churn Prediction — Real-time inference powered by Random Forest & Gradient Boosting</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='font-family:Syne;font-size:1.1rem;font-weight:700;color:#e2e8f0;margin-bottom:1rem'>Customer Profile</div>", unsafe_allow_html=True)

    st.markdown("**📊 Demographics**")
    geography   = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender      = st.selectbox("Gender", ["Male", "Female"])
    age         = st.slider("Age", 18, 92, 38)

    st.markdown("---")
    st.markdown("**💳 Financial Info**")
    credit_score     = st.slider("Credit Score", 350, 850, 650)
    balance          = st.number_input("Account Balance ($)", min_value=0.0, max_value=250000.0, value=76000.0, step=500.0)
    estimated_salary = st.number_input("Estimated Annual Salary ($)", min_value=0.0, max_value=200000.0, value=75000.0, step=1000.0)

    st.markdown("---")
    st.markdown("**🏦 Banking Relationship**")
    tenure         = st.slider("Tenure (Years)", 0, 10, 5)
    num_products   = st.selectbox("Number of Products", [1, 2, 3, 4])
    has_cr_card    = st.radio("Has Credit Card?", ["Yes", "No"], horizontal=True)
    is_active      = st.radio("Active Member?", ["Yes", "No"], horizontal=True)

    st.markdown("---")
    predict_btn = st.button("🔍  Run Prediction", use_container_width=True, type="primary")


# ── Prediction ─────────────────────────────────────────────────────────────────
def predict(geography, gender, age, credit_score, balance, estimated_salary,
            tenure, num_products, has_cr_card, is_active):
    geo_enc = le_geo.transform([geography])[0]
    gen_enc = le_gen.transform([gender])[0]
    hcc = 1 if has_cr_card == "Yes" else 0
    iam = 1 if is_active == "Yes" else 0

    inp = np.array([[credit_score, geo_enc, gen_enc, age, tenure,
                     balance, num_products, hcc, iam, estimated_salary]])
    inp_scaled = scaler.transform(inp)
    prob = model.predict_proba(inp_scaled)[0][1]
    pred = int(prob >= 0.5)
    return pred, prob


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Model Performance", "🔬 Data Insights"])

# ─── TAB 1: Prediction ────────────────────────────────────────────────────────
with tab1:
    if predict_btn:
        pred, prob = predict(geography, gender, age, credit_score, balance,
                             estimated_salary, tenure, num_products, has_cr_card, is_active)
        churn_pct  = prob * 100
        retain_pct = (1 - prob) * 100

        col1, col2 = st.columns([1, 1])

        with col1:
            if pred == 1:
                st.markdown(f"""
                <div class="result-churn">
                    <div style="font-size:3rem">⚠️</div>
                    <div class="result-label" style="color:#f87171">HIGH CHURN RISK</div>
                    <div style="font-size:2.5rem;font-family:Syne;font-weight:800;color:#fca5a5;margin:0.5rem 0">
                        {churn_pct:.1f}%
                    </div>
                    <div class="result-sublabel">Probability of churning</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-safe">
                    <div style="font-size:3rem">✅</div>
                    <div class="result-label" style="color:#4ade80">LOW CHURN RISK</div>
                    <div style="font-size:2.5rem;font-family:Syne;font-weight:800;color:#86efac;margin:0.5rem 0">
                        {retain_pct:.1f}%
                    </div>
                    <div class="result-sublabel">Probability of staying</div>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_pct,
                number={'suffix': '%', 'font': {'size': 32, 'color': '#e2e8f0', 'family': 'Syne'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#64748b', 'tickfont': {'color': '#64748b'}},
                    'bar': {'color': '#f87171' if pred == 1 else '#4ade80', 'thickness': 0.25},
                    'bgcolor': '#1a1f35',
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 30], 'color': '#0f2d1a'},
                        {'range': [30, 60], 'color': '#2d2a0f'},
                        {'range': [60, 100], 'color': '#2d1515'},
                    ],
                    'threshold': {
                        'line': {'color': '#fbbf24', 'width': 3},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#e2e8f0'}, height=260, margin=dict(t=30, b=10, l=20, r=20),
                title={'text': 'Churn Risk Score', 'font': {'family': 'Syne', 'size': 14, 'color': '#94a3b8'}, 'x': 0.5}
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Key factors
        st.markdown("<div class='section-header'>Key Risk Factors</div>", unsafe_allow_html=True)
        factors = {
            'Geography (Germany)': 1.0 if geography == 'Germany' else 0.0,
            'Age > 40':            min((max(age - 40, 0)) / 50, 1.0),
            'Inactive Member':     1.0 if is_active == 'No' else 0.0,
            'Multiple Products':   1.0 if num_products >= 3 else 0.0,
            'Low Credit Score':    max(0, (500 - credit_score) / 150),
            'High Balance':        min(balance / 200000, 1.0),
            'Short Tenure':        max(0, (3 - tenure) / 3),
        }
        fig_factors = go.Figure(go.Bar(
            x=list(factors.values()),
            y=list(factors.keys()),
            orientation='h',
            marker=dict(
                color=[f'rgba(248,113,113,{v*0.9+0.1})' for v in factors.values()],
                line=dict(width=0)
            ),
            text=[f'{v:.0%}' for v in factors.values()],
            textposition='outside',
            textfont=dict(color='#94a3b8', size=12)
        ))
        fig_factors.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(range=[0, 1.15], showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(tickfont=dict(color='#94a3b8', size=12), gridcolor='#1e2540'),
            height=280, margin=dict(t=10, b=10, l=10, r=60),
            font=dict(color='#94a3b8')
        )
        st.plotly_chart(fig_factors, use_container_width=True)

        # Retention actions
        if pred == 1:
            st.markdown("<div class='section-header'>💡 Recommended Retention Actions</div>", unsafe_allow_html=True)
            actions = []
            if is_active == 'No':     actions.append("🎯 Launch re-engagement campaign with personalised offers")
            if num_products <= 1:     actions.append("📦 Cross-sell a second product (insurance, savings account)")
            if balance > 100000:      actions.append("💎 Offer premium/priority banking tier")
            if credit_score < 600:    actions.append("📈 Provide credit score improvement program")
            if tenure < 3:            actions.append("🤝 Assign dedicated relationship manager")
            if not actions:           actions.append("📞 Schedule proactive check-in call within 7 days")
            for a in actions:
                st.markdown(f"<div style='background:#111827;border:1px solid #1e2540;border-radius:8px;padding:0.7rem 1rem;margin:0.4rem 0;color:#cbd5e1;font-size:0.9rem'>{a}</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align:center;padding:4rem 2rem;color:#475569'>
            <div style='font-size:3rem;margin-bottom:1rem'>🎯</div>
            <div style='font-family:Syne;font-size:1.1rem;font-weight:600;color:#64748b'>Fill in the customer profile on the left</div>
            <div style='font-size:0.85rem;margin-top:0.5rem'>Then click <strong style='color:#63b3ed'>Run Prediction</strong> to see churn risk analysis</div>
        </div>
        """, unsafe_allow_html=True)


# ─── TAB 2: Model Performance ─────────────────────────────────────────────────
with tab2:
    st.markdown("<div class='section-header'>Model Comparison</div>", unsafe_allow_html=True)

    # Metric cards
    cols = st.columns(3)
    best = model_results['Random Forest']
    metrics = [('Accuracy', f"{best['accuracy']}%"), ('ROC-AUC', f"{best['auc']}%"), ('F1 Score', f"{best['f1']}%")]
    for col, (label, val) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Bar chart: model comparison
    model_names = list(model_results.keys())
    acc_vals  = [model_results[m]['accuracy'] for m in model_names]
    auc_vals  = [model_results[m]['auc']      for m in model_names]
    f1_vals   = [model_results[m]['f1']       for m in model_names]

    fig_cmp = go.Figure()
    for vals, name, color in [(acc_vals,'Accuracy','#63b3ed'), (auc_vals,'ROC-AUC','#68d391'), (f1_vals,'F1 Score','#f6ad55')]:
        fig_cmp.add_trace(go.Bar(name=name, x=model_names, y=vals, marker_color=color,
                                  marker_line_width=0, text=[f'{v}%' for v in vals],
                                  textposition='outside', textfont=dict(size=11, color='#94a3b8')))
    fig_cmp.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        barmode='group', height=350,
        xaxis=dict(tickfont=dict(color='#94a3b8'), gridcolor='#1e2540'),
        yaxis=dict(range=[0, 115], tickfont=dict(color='#94a3b8'), gridcolor='#1e2540', ticksuffix='%'),
        legend=dict(font=dict(color='#94a3b8'), bgcolor='rgba(0,0,0,0)'),
        margin=dict(t=20, b=20), font=dict(color='#94a3b8')
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # Feature importances
    st.markdown("<div class='section-header'>Feature Importances (Random Forest)</div>", unsafe_allow_html=True)
    feat_labels = ['Credit Score', 'Geography', 'Gender', 'Age', 'Tenure',
                   'Balance', 'Num Products', 'Has Credit Card', 'Active Member', 'Est. Salary']
    importances = model.feature_importances_
    sorted_idx  = np.argsort(importances)
    fig_imp = go.Figure(go.Bar(
        x=importances[sorted_idx],
        y=[feat_labels[i] for i in sorted_idx],
        orientation='h',
        marker=dict(color=[f'rgba(99,179,237,{0.3 + 0.7 * v / importances.max()})' for v in importances[sorted_idx]], line_width=0),
        text=[f'{v:.3f}' for v in importances[sorted_idx]],
        textposition='outside', textfont=dict(color='#94a3b8', size=11)
    ))
    fig_imp.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(tickfont=dict(color='#94a3b8', size=12), gridcolor='#1e2540'),
        height=360, margin=dict(t=10, b=10, l=10, r=60), font=dict(color='#94a3b8')
    )
    st.plotly_chart(fig_imp, use_container_width=True)


# ─── TAB 3: Data Insights ─────────────────────────────────────────────────────
with tab3:
    st.markdown("<div class='section-header'>Churn Distribution by Segment</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        churn_geo = df.groupby('Geography')['Exited'].mean().reset_index()
        churn_geo['Churn %'] = (churn_geo['Exited'] * 100).round(1)
        fig1 = px.bar(churn_geo, x='Geography', y='Churn %',
                      color='Churn %', color_continuous_scale=['#1a365d', '#63b3ed', '#f87171'],
                      title='Churn Rate by Geography')
        fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           font=dict(color='#94a3b8'), title_font=dict(family='Syne', color='#e2e8f0'),
                           showlegend=False, height=300, margin=dict(t=40, b=20))
        fig1.update_xaxes(tickfont=dict(color='#94a3b8'), gridcolor='#1e2540')
        fig1.update_yaxes(tickfont=dict(color='#94a3b8'), gridcolor='#1e2540', ticksuffix='%')
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        churn_prod = df.groupby('NumOfProducts')['Exited'].mean().reset_index()
        churn_prod['Churn %'] = (churn_prod['Exited'] * 100).round(1)
        fig2 = px.bar(churn_prod, x='NumOfProducts', y='Churn %',
                      color='Churn %', color_continuous_scale=['#1a365d', '#63b3ed', '#f87171'],
                      title='Churn Rate by Number of Products')
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           font=dict(color='#94a3b8'), title_font=dict(family='Syne', color='#e2e8f0'),
                           showlegend=False, height=300, margin=dict(t=40, b=20))
        fig2.update_xaxes(tickfont=dict(color='#94a3b8'), gridcolor='#1e2540', title_text='# Products')
        fig2.update_yaxes(tickfont=dict(color='#94a3b8'), gridcolor='#1e2540', ticksuffix='%')
        st.plotly_chart(fig2, use_container_width=True)

    # Age distribution
    st.markdown("<div class='section-header'>Age Distribution: Churned vs. Retained</div>", unsafe_allow_html=True)
    churned    = df[df['Exited'] == 1]['Age']
    retained   = df[df['Exited'] == 0]['Age']
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(x=retained, name='Retained', nbinsx=30,
                                 marker_color='rgba(99,179,237,0.6)', marker_line_width=0))
    fig3.add_trace(go.Histogram(x=churned,  name='Churned',  nbinsx=30,
                                 marker_color='rgba(248,113,113,0.6)', marker_line_width=0))
    fig3.update_layout(
        barmode='overlay', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8'), height=300, margin=dict(t=10, b=20),
        legend=dict(font=dict(color='#94a3b8'), bgcolor='rgba(0,0,0,0)'),
        xaxis=dict(tickfont=dict(color='#94a3b8'), gridcolor='#1e2540', title_text='Age', title_font=dict(color='#64748b')),
        yaxis=dict(tickfont=dict(color='#94a3b8'), gridcolor='#1e2540', title_text='Count', title_font=dict(color='#64748b'))
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Summary stats
    st.markdown("<div class='section-header'>Dataset Overview</div>", unsafe_allow_html=True)
    total     = len(df)
    churned_n = df['Exited'].sum()
    c1, c2, c3, c4 = st.columns(4)
    for col_w, val, label in [
        (c1, f"{total:,}", "Total Customers"),
        (c2, f"{churned_n:,}", "Churned"),
        (c3, f"{total-churned_n:,}", "Retained"),
        (c4, f"{churned_n/total*100:.1f}%", "Churn Rate"),
    ]:
        with col_w:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)
