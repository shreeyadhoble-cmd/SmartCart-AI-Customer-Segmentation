import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="SmartCart",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

ARTIFACT_PATH = "smartcart_model_artifacts.pkl"
DATA_PATH = "smartcart_customers.csv"

# ---------- Theme / UI ----------
st.markdown(
    """
<style>
/* App background */
.stApp {
    background: #070d1a;
    color: #f5f7ff;
}
[data-testid="stHeader"] {
    background: rgba(7,13,26,0.92);
}
[data-testid="stAppViewContainer"] > .main {
    background: #070d1a;
}
.block-container {
    max-width: 1500px;
    padding-top: 2.0rem;
    padding-bottom: 2rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101a31 0%, #0b1325 55%, #081020 100%);
    border-right: 1px solid rgba(120,145,210,0.16);
}
section[data-testid="stSidebar"] > div {
    padding-top: 1.4rem;
}
.sidebar-brand {
    padding: 8px 8px 22px 8px;
}
.brand-row {
    display: flex;
    align-items: center;
    gap: 11px;
}
.brand-icon {
    width: 48px;
    height: 48px;
    border-radius: 15px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
    background: linear-gradient(135deg, #2563eb, #a855f7);
    box-shadow: 0 10px 28px rgba(80,70,220,.32);
}
.brand-name {
    font-size: 28px;
    font-weight: 850;
    letter-spacing: -1px;
    color: #f7f8ff;
}
.brand-name span {
    color: #a855f7;
}
.brand-sub {
    margin-top: 12px;
    color: #9fb0d7;
    line-height: 1.45;
    font-size: 14px;
}
.sidebar-divider {
    height: 1px;
    background: rgba(148,163,184,.18);
    margin: 10px 0 18px 0;
}

/* Radio navigation */
section[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 8px;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    border-radius: 12px;
    padding: 11px 12px;
    transition: .2s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(79, 105, 220, .16);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {
    background: linear-gradient(90deg, #3148c8, #3848b9);
    box-shadow: 0 8px 24px rgba(44,68,196,.24);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label p {
    color: #e9edff !important;
    font-size: 15px;
    font-weight: 650;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
    display: none;
}

.quick-box {
    margin-top: 26px;
    padding: 15px;
    border-radius: 14px;
    background: rgba(17,30,57,.72);
    border: 1px solid rgba(130,151,210,.14);
}
.quick-title {
    color: #aebeff;
    font-weight: 750;
    font-size: 15px;
    margin-bottom: 8px;
}
.quick-text {
    color: #8f9fbe;
    font-size: 13px;
    line-height: 1.5;
}
.sidebar-footer {
    margin-top: 80px;
    text-align: center;
    color: #8998b7;
    font-size: 13px;
}
.sidebar-footer strong {
    color: #d66cff;
}

/* Page heading */
.page-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 18px;
}
.page-title {
    font-size: 38px;
    font-weight: 850;
    letter-spacing: -1.5px;
    color: #f8f9ff;
    margin: 0;
}
.page-subtitle {
    color: #96a5c5;
    font-size: 15px;
    margin-top: 3px;
}

/* Hero */
.hero {
    min-height: 155px;
    border-radius: 20px;
    padding: 28px 34px;
    background: linear-gradient(105deg, #0756d9 0%, #2230a8 50%, #3b176b 100%);
    border: 1px solid rgba(115,151,255,.42);
    box-shadow: 0 18px 50px rgba(16,45,150,.22);
    position: relative;
    overflow: hidden;
}
.hero:after {
    content: "";
    position: absolute;
    width: 420px;
    height: 420px;
    right: -100px;
    top: -160px;
    border-radius: 50%;
    background: rgba(255,255,255,.055);
}
.hero-row {
    display: flex;
    align-items: center;
    gap: 18px;
    position: relative;
    z-index: 1;
}
.hero-cart {
    width: 65px;
    height: 65px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 39px;
}
.hero-title {
    font-size: 35px;
    font-weight: 850;
    line-height: 1;
}
.hero-title span {
    color: #d77aff;
}
.hero-sub {
    margin-top: 9px;
    font-size: 16px;
    color: #dce5ff;
}
.hero-right {
    margin-left: auto;
    min-width: 210px;
    position: relative;
    z-index: 1;
}
.hero-right-title {
    font-size: 18px;
    font-weight: 750;
    line-height: 1.25;
}
.hero-line {
    width: 50px;
    height: 4px;
    margin-top: 14px;
    border-radius: 5px;
    background: #70a0ff;
}

/* KPI cards */
.kpi-card {
    border-radius: 17px;
    padding: 22px;
    min-height: 140px;
    background: #0c172c;
    border: 1px solid rgba(102,131,205,.25);
    box-shadow: 0 12px 35px rgba(0,0,0,.16);
}
.kpi-blue { border-color: rgba(62,137,255,.42); }
.kpi-green { border-color: rgba(48,207,178,.36); background: linear-gradient(135deg,#0b2028,#0c182a); }
.kpi-purple { border-color: rgba(185,80,255,.36); background: linear-gradient(135deg,#17132d,#0e172c); }
.kpi-top { display:flex; align-items:center; gap:13px; }
.kpi-icon {
    width: 45px; height:45px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:21px;
    background: rgba(38,96,220,.30);
}
.kpi-green .kpi-icon { background: rgba(25,178,145,.25); }
.kpi-purple .kpi-icon { background: rgba(155,65,235,.23); }
.kpi-label { color:#aab7d4; font-weight:650; font-size:14px; }
.kpi-value { color:#f7f9ff; font-size:31px; font-weight:850; margin-top:9px; letter-spacing:-.8px; }
.kpi-note { color:#59e2bd; font-size:12px; margin-top:7px; }

/* Chart containers */
.chart-card {
    background: #0a1529;
    border: 1px solid rgba(88,119,194,.22);
    border-radius: 17px;
    padding: 12px 14px 5px 14px;
}
.section-title {
    color: #eef2ff;
    font-size: 21px;
    font-weight: 780;
    margin: 22px 0 12px;
}

/* Streamlit widgets */
.stButton > button, .stFormSubmitButton > button {
    border-radius: 10px;
    border: 0;
    background: linear-gradient(90deg,#3154e7,#7b42e8);
    color: white;
    font-weight: 700;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    background: linear-gradient(90deg,#4164f0,#8b51f0);
}
[data-testid="stMetric"] {
    background: #0c172c;
    border: 1px solid rgba(102,131,205,.24);
    padding: 14px;
    border-radius: 14px;
}

/* Hide Streamlit chrome that is not part of the design */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    return joblib.load(ARTIFACT_PATH)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Income"] = df["Income"].fillna(df["Income"].median())
    df["Age"] = 2026 - df["Year_Birth"]
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
    reference_date = df["Dt_Customer"].max()
    df["Customer_Tenure_Days"] = (reference_date - df["Dt_Customer"]).dt.days
    df["Total_Spending"] = (
        df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"]
        + df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
    )
    df["Total_Children"] = df["Kidhome"] + df["Teenhome"]
    df["Education"] = df["Education"].replace({
        "Basic": "Undergraduate", "2n Cycle": "Undergraduate",
        "Graduation": "Graduate", "Master": "Postgraduate", "PhD": "Postgraduate"
    })
    df["Living_With"] = df["Marital_Status"].replace({
        "Married": "Partner", "Together": "Partner", "Single": "Alone",
        "Divorced": "Alone", "Widow": "Alone", "Absurd": "Alone", "YOLO": "Alone"
    })
    return df


def _prepare_prediction_frame(data, encoder):
    features = [
        "Income", "Recency", "NumDealsPurchases", "NumWebPurchases",
        "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth",
        "Complain", "Response", "Age", "Customer_Tenure_Days",
        "Total_Spending", "Total_Children", "Education", "Living_With"
    ]
    x = data[features].copy()
    encoded = pd.DataFrame(
        encoder.transform(x[["Education", "Living_With"]]).toarray(),
        columns=encoder.get_feature_names_out(["Education", "Living_With"]),
        index=x.index,
    )
    x = x.drop(columns=["Education", "Living_With"])
    return pd.concat([x, encoded], axis=1)


def calculate_rfm_score(recency, frequency, monetary, bins):
    if not bins:
        r = 5 if recency <= 20 else 4 if recency <= 40 else 3 if recency <= 60 else 2 if recency <= 80 else 1
        f = 1 if frequency <= 3 else 2 if frequency <= 6 else 3 if frequency <= 10 else 4 if frequency <= 15 else 5
        m = 1 if monetary <= 50 else 2 if monetary <= 200 else 3 if monetary <= 500 else 4 if monetary <= 1000 else 5
        return r + f + m

    def score(value, q):
        return int(np.digitize(value, q, right=True) + 1)

    return int((6 - score(recency, bins["recency"])) + score(frequency, bins["frequency"]) + score(monetary, bins["monetary"]))


def rfm_segment_name(score):
    if score >= 13:
        return "Champions"
    if score >= 10:
        return "Loyal Customers"
    if score >= 8:
        return "Potential Loyalists"
    if score >= 6:
        return "At Risk"
    return "Lost Customers"


def default_recommendation(segment):
    return {
        "Champions": "Reward with exclusive offers, early access, and loyalty benefits.",
        "Loyal Customers": "Encourage repeat purchases with personalized offers and loyalty rewards.",
        "Potential Loyalists": "Use targeted promotions and recommendations to increase purchase frequency.",
        "At Risk": "Send re-engagement offers, reminders, and personalized discounts.",
        "Lost Customers": "Use win-back campaigns and special incentives to encourage a return.",
    }[segment]


def money(value):
    return f"₹{value:,.0f}"


def plotly_dark(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dce5ff", family="Inter, Arial"),
        margin=dict(l=20, r=20, t=42, b=20),
        title=dict(font=dict(size=18, color="#eef2ff")),
        xaxis=dict(gridcolor="rgba(91,119,177,.14)", zerolinecolor="rgba(91,119,177,.14)"),
        yaxis=dict(gridcolor="rgba(91,119,177,.14)", zerolinecolor="rgba(91,119,177,.14)"),
    )
    return fig


# ---------- Required model files ----------
if not os.path.exists(ARTIFACT_PATH):
    st.error("smartcart_model_artifacts.pkl is missing. Run the final deployment cell in the notebook first.")
    st.stop()

artifacts = load_artifacts()
ohe = artifacts["encoder"]
scaler = artifacts["scaler"]
model = artifacts["prediction_model"]
cluster_info = artifacts["cluster_info"]
recommendations = artifacts.get("recommendations", {})
rfm_bins = artifacts.get("rfm_bins")

df = load_data() if os.path.exists(DATA_PATH) else None

# ---------- Sidebar ----------
st.sidebar.markdown(
    """
<div class="sidebar-brand">
  <div class="brand-row">
    <div class="brand-icon">🛒</div>
    <div class="brand-name">Smart<span>Cart</span></div>
  </div>
  <div class="brand-sub">AI-powered customer segmentation<br>and RFM analysis</div>
</div>
<div class="sidebar-divider"></div>
""",
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigate",
    ["🏠  Dashboard", "👤  Customer Prediction", "ⓘ  About"],
    label_visibility="collapsed",
)

st.sidebar.markdown(
    """
<div class="quick-box">
  <div class="quick-title">✣ &nbsp;Quick Insights</div>
  <div class="quick-text">Understand your customers.<br>Build better strategies.</div>
</div>
<div class="sidebar-footer">Smarter <strong>Customers.</strong><br>Better Opportunities.<br><br>🛒</div>
""",
    unsafe_allow_html=True,
)

page = page.split("  ", 1)[1]

# ---------- Dashboard ----------
if page == "Dashboard":
    if df is None:
        st.warning("smartcart_customers.csv is missing.")
        st.stop()

    clean = df[(df["Age"] < 90) & (df["Income"] < 600_000)].copy()

    st.markdown(
        """
<div class="page-head">
  <div>
    <div class="page-title">⌂ &nbsp;Dashboard</div>
    <div class="page-subtitle">Customer intelligence at a glance</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="hero">
  <div class="hero-row">
    <div class="hero-cart">🛒</div>
    <div>
      <div class="hero-title">Smart<span>Cart</span></div>
      <div class="hero-sub">AI-powered customer segmentation and RFM analysis</div>
    </div>
    <div class="hero-right">
      <div class="hero-right-title">Better Insights<br>for Smarter<br>Business Decisions</div>
      <div class="hero-line"></div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")
    k1, k2, k3 = st.columns(3, gap="medium")
    with k1:
        st.markdown(
            f'''<div class="kpi-card kpi-blue"><div class="kpi-top"><div class="kpi-icon">👥</div><div class="kpi-label">Customers</div></div><div class="kpi-value">{len(df):,}</div><div class="kpi-note">↑ Total customers in dataset</div></div>''',
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f'''<div class="kpi-card kpi-green"><div class="kpi-top"><div class="kpi-icon">◉</div><div class="kpi-label">Average Income</div></div><div class="kpi-value">{money(df["Income"].mean())}</div><div class="kpi-note">↑ Per customer (approx.)</div></div>''',
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f'''<div class="kpi-card kpi-purple"><div class="kpi-top"><div class="kpi-icon">🛍</div><div class="kpi-label">Average Spending</div></div><div class="kpi-value">{money(df["Total_Spending"].mean())}</div><div class="kpi-note">↑ Per customer (approx.)</div></div>''',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Customer Overview</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        fig = px.histogram(clean, x="Income", nbins=30, title="Income Distribution")
        fig.update_traces(marker_line_width=0, opacity=.9)
        fig.update_xaxes(title="Income (₹)")
        fig.update_yaxes(title="count")
        fig = plotly_dark(fig)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        fig = px.scatter(
            clean,
            x="Income",
            y="Total_Spending",
            hover_data=["Age", "Recency"],
            title="Income vs Total Spending",
        )
        fig.update_traces(marker=dict(size=7, opacity=.72))
        fig.update_xaxes(title="Income (₹)")
        fig.update_yaxes(title="Total Spending (₹)")
        fig = plotly_dark(fig)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Recent Customers</div>', unsafe_allow_html=True)
    st.dataframe(
        df[["ID", "Income", "Age", "Recency", "Total_Spending", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]].head(10),
        use_container_width=True,
        hide_index=True,
    )

# ---------- Prediction ----------
elif page == "Customer Prediction":
    st.markdown(
        """
<div class="page-head"><div><div class="page-title">👤 &nbsp;Customer Prediction</div>
<div class="page-subtitle">Predict a customer segment and generate an RFM-based business recommendation</div></div></div>
""",
        unsafe_allow_html=True,
    )

    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            income = st.number_input("Income", min_value=0.0, value=50000.0, step=1000.0)
            age = st.number_input("Age", min_value=18, max_value=89, value=35)
            recency = st.number_input("Recency (days)", min_value=0, max_value=120, value=30)
            total_children = st.number_input("Total Children", min_value=0, max_value=10, value=1)
            tenure = st.number_input("Customer Tenure (days)", min_value=0, max_value=5000, value=400)
        with col2:
            deals = st.number_input("Deal Purchases", min_value=0, max_value=30, value=2)
            web = st.number_input("Web Purchases", min_value=0, max_value=30, value=5)
            catalog = st.number_input("Catalog Purchases", min_value=0, max_value=30, value=4)
            store = st.number_input("Store Purchases", min_value=0, max_value=30, value=6)
            web_visits = st.number_input("Web Visits / Month", min_value=0, max_value=30, value=4)
        with col3:
            spending = st.number_input("Total Spending", min_value=0.0, value=1000.0, step=50.0)
            education = st.selectbox("Education", ["Graduate", "Postgraduate", "Undergraduate"])
            living = st.selectbox("Living With", ["Partner", "Alone"])
            complain = st.selectbox("Complaint", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            response = st.selectbox("Previous Campaign Response", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        submitted = st.form_submit_button("Predict Customer Segment", type="primary", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([{
            "Income": income, "Recency": recency, "NumDealsPurchases": deals,
            "NumWebPurchases": web, "NumCatalogPurchases": catalog, "NumStorePurchases": store,
            "NumWebVisitsMonth": web_visits, "Complain": complain, "Response": response,
            "Age": age, "Customer_Tenure_Days": tenure, "Total_Spending": spending,
            "Total_Children": total_children, "Education": education, "Living_With": living,
        }])
        cat = pd.DataFrame(
            ohe.transform(input_df[["Education", "Living_With"]]).toarray(),
            columns=ohe.get_feature_names_out(["Education", "Living_With"]),
        )
        model_input = scaler.transform(pd.concat([input_df.drop(columns=["Education", "Living_With"]), cat], axis=1))
        predicted_cluster = int(model.predict(model_input)[0])

        frequency = web + catalog + store
        rfm_score = calculate_rfm_score(recency, frequency, spending, rfm_bins)
        rfm_segment = rfm_segment_name(rfm_score)
        recommendation = recommendations.get(rfm_segment, default_recommendation(rfm_segment))

        st.success(f"Predicted Cluster: {predicted_cluster}")
        a, b, c = st.columns(3)
        a.metric("Cluster", predicted_cluster)
        b.metric("RFM Score", rfm_score)
        c.metric("RFM Segment", rfm_segment)
        st.markdown('<div class="section-title">Business Recommendation</div>', unsafe_allow_html=True)
        st.info(recommendation)
        if predicted_cluster in cluster_info.index:
            st.markdown('<div class="section-title">Cluster Profile</div>', unsafe_allow_html=True)
            st.dataframe(cluster_info.loc[[predicted_cluster]], use_container_width=True)

# ---------- About ----------
else:
    st.markdown(
        """
<div class="page-head"><div><div class="page-title">ⓘ &nbsp;About SmartCart</div>
<div class="page-subtitle">Customer segmentation, prediction and RFM analysis</div></div></div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="hero">
  <div class="hero-row">
    <div class="hero-cart">🛒</div>
    <div><div class="hero-title">Smart<span>Cart</span></div>
    <div class="hero-sub">Turn customer behavior into actionable business insights.</div></div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-title">How it works</div>', unsafe_allow_html=True)
    st.write(
        "SmartCart uses customer-level demographic and behavioral features to compare K-Means and Agglomerative Clustering. Agglomerative Clustering provides the final discovered labels, while a Random Forest classifier learns those labels so new customers can be assigned to a segment in Streamlit."
    )
    st.markdown('<div class="section-title">RFM Analysis</div>', unsafe_allow_html=True)
    st.write("Recency, Frequency and Monetary value are combined into an RFM score and business segment such as Champions, Loyal Customers, Potential Loyalists, At Risk and Lost Customers.")
