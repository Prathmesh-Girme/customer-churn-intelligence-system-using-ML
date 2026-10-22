"""
Customer Churn Intelligence System using ML
Designed and developed by Prathmesh Girme
Run locally:  streamlit run app.py
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
MODEL_PATH = Path(__file__).parent / "customer_churn_model.pkl"
FEATURES = ["Age", "Tenure", "MonthlyCharges", "SupportCalls"]
MAX_BATCH_ROWS = 50_000

# ---- Branding (change here, updates everywhere) ----
APP_NAME = "Customer Churn Intelligence System using ML"
TAGLINE = "Machine learning based churn risk scoring to support customer retention decisions"
DEVELOPER = "Prathmesh Girme"
LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("churn-dashboard")

st.set_page_config(
    page_title=APP_NAME,
    page_icon=Image.open(LOGO_PATH) if LOGO_PATH.exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1280px;}
      footer {visibility: hidden;}
      h2, h3 {font-weight: 600; letter-spacing: -.2px;}

      /* Header */
      .app-header {
          display: flex; align-items: stretch; justify-content: space-between; gap: 1.5rem;
          background: #0f172a; color: #f8fafc; border-radius: 8px;
          padding: 1.5rem 1.8rem; margin-bottom: 1.2rem;
          border-top: 4px solid #2563eb; border: 1px solid #1e293b; border-top: 4px solid #2563eb;
      }
      .app-eyebrow {
          font-size: .75rem; letter-spacing: 1.6px; text-transform: uppercase;
          color: #93c5fd; font-weight: 600; margin-bottom: .35rem;
      }
      .app-header h1 {margin: 0; padding: 0; font-size: 1.85rem; line-height: 1.25; color: #f8fafc; font-weight: 700;}
      .app-header p {margin: .45rem 0 0 0; font-size: .98rem; color: #cbd5e1; max-width: 640px;}
      .app-credit {
          flex-shrink: 0; align-self: center; text-align: right; padding-left: 1.4rem;
          border-left: 2px solid #2563eb; min-width: 190px;
      }
      .app-credit span {display: block; font-size: .72rem; letter-spacing: 1.2px;
          text-transform: uppercase; color: #94a3b8;}
      .app-credit b {display: block; margin-top: .2rem; font-size: 1.1rem; color: #f8fafc; font-weight: 600;}
      @media (max-width: 800px) {
          .app-header {flex-direction: column;}
          .app-credit {text-align: left; border-left: none; padding-left: 0;
              border-top: 1px solid #334155; padding-top: .8rem; align-self: stretch;}
      }

      /* Sidebar credit */
      .dev-card {
          border-left: 3px solid #2563eb; padding: .55rem .85rem; border-radius: 4px;
          background: rgba(37,99,235,.10);
      }
      .dev-card small {display: block; font-size: .68rem; letter-spacing: 1px;
          text-transform: uppercase; opacity: .75;}
      .dev-card b {font-size: .98rem;}

      /* Risk card */
      .risk-card {
          border-radius: 6px; padding: 1rem 1.2rem; margin-bottom: .6rem;
          border: 1px solid rgba(128,128,128,.25);
      }
      .risk-low    {background: rgba(22,163,74,.10);  border-left: 5px solid #16a34a;}
      .risk-medium {background: rgba(217,119,6,.10);  border-left: 5px solid #d97706;}
      .risk-high   {background: rgba(220,38,38,.10);  border-left: 5px solid #dc2626;}
      .risk-card h3 {margin: 0 0 .2rem 0; font-size: 1.1rem; letter-spacing: .6px; text-transform: uppercase;}
      .risk-card p  {margin: 0; opacity: .85;}

      .footer-dev {text-align: center; padding: .6rem; font-size: .85rem; opacity: .75;}
      .footer-dev b {font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Model helpers
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model...")
def load_model(path: Path):
    """Load the trained model once and cache it across reruns/sessions."""
    if not path.exists():
        raise FileNotFoundError(
            f"Model file '{path.name}' not found. Place it next to app.py "
            "or run `python train_model.py` to create one."
        )
    return joblib.load(path)


def predict_proba(model, df: pd.DataFrame) -> pd.Series:
    """Return churn probability (class 1) for every row."""
    df = df[FEATURES]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)
        classes = list(getattr(model, "classes_", [0, 1]))
        idx = classes.index(1) if 1 in classes else -1
        return pd.Series(proba[:, idx], index=df.index)
    # Fallback for models without probabilities
    return pd.Series(model.predict(df), index=df.index).astype(float)


def risk_level(p: float, threshold: float) -> str:
    if p >= max(threshold, 0.66):
        return "high"
    if p >= threshold * 0.6:
        return "medium"
    return "low"


RISK_TEXT = {
    "low": ("Low risk", "This customer is unlikely to leave. Keep engagement steady."),
    "medium": ("Medium risk", "Some warning signs. Consider a proactive check-in."),
    "high": ("High risk", "High chance of churn. Prioritise retention outreach."),
}


def recommendations(row: dict) -> list[str]:
    """Simple rule-based retention hints (business heuristics, not model output)."""
    tips = []
    if row["SupportCalls"] >= 4:
        tips.append("Many support calls: assign a dedicated agent and review open issues.")
    if row["Tenure"] < 12:
        tips.append("New customer (<12 months): run an onboarding / loyalty programme.")
    if row["MonthlyCharges"] > 100:
        tips.append("High monthly bill: offer a plan review or a discount bundle.")
    if row["Age"] < 25:
        tips.append("Younger customer: promote flexible or digital-first plans.")
    if not tips:
        tips.append("No specific risk drivers detected. Continue standard engagement.")
    return tips


def gauge(prob: float, threshold: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 46}},
            title={"text": "Churn probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1f2937"},
                "steps": [
                    {"range": [0, 33], "color": "#bbf7d0"},
                    {"range": [33, 66], "color": "#fde68a"},
                    {"range": [66, 100], "color": "#fecaca"},
                ],
                "threshold": {
                    "line": {"color": "#dc2626", "width": 4},
                    "thickness": 0.8,
                    "value": threshold * 100,
                },
            },
        )
    )
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=10))
    return fig


# --------------------------------------------------------------------------- #
# Load model (fail gracefully)
# --------------------------------------------------------------------------- #
try:
    model = load_model(MODEL_PATH)
except Exception as exc:  # noqa: BLE001
    logger.exception("Model load failed")
    st.error(f"Could not load the model: {exc}")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=56)
    st.markdown(f"**{APP_NAME}**")
    st.markdown(
        f'<div class="dev-card"><small>Designed and developed by</small><b>{DEVELOPER}</b></div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.subheader("Configuration")
    threshold = st.slider(
        "Decision threshold",
        0.05, 0.95, 0.50, 0.05,
        help="Customers with churn probability at or above this value are flagged as 'likely to churn'.",
    )
    st.caption("Lower threshold = catches more churners but raises false alarms.")
    st.divider()
    st.markdown("**Model status**")
    st.success("Model loaded successfully")
    st.caption(f"Type: `{type(model).__name__}`")
    st.caption(f"Features: {', '.join(FEATURES)}")
    st.divider()
    if st.button("Clear prediction history", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown(
    f"""<div class="app-header">
<div>
<div class="app-eyebrow">Customer Analytics &middot; Retention Intelligence</div>
<h1>{APP_NAME}</h1>
<p>{TAGLINE}</p>
</div>
<div class="app-credit"><span>Designed and developed by</span><b>{DEVELOPER}</b></div>
</div>""",
    unsafe_allow_html=True,
)

tab_single, tab_batch, tab_insights, tab_about = st.tabs(
    ["Single Prediction", "Batch Prediction", "Model Insights", "About"]
)

# --------------------------------------------------------------------------- #
# Tab 1 - Single prediction
# --------------------------------------------------------------------------- #
with tab_single:
    left, right = st.columns([1, 1.2], gap="large")

    with left:
        st.subheader("Customer details")
        with st.form("single_form"):
            age = st.number_input("Customer age", 18, 100, 35)
            tenure = st.number_input("Tenure (months)", 0, 600, 12)
            charges = st.number_input("Monthly charges", 0.0, 10_000.0, 70.0, step=5.0)
            calls = st.number_input("Support calls", 0, 100, 1)
            submitted = st.form_submit_button(
                "Run prediction", type="primary", use_container_width=True
            )

    with right:
        st.subheader("Prediction")
        if submitted:
            try:
                row = {
                    "Age": age,
                    "Tenure": tenure,
                    "MonthlyCharges": charges,
                    "SupportCalls": calls,
                }
                prob = float(predict_proba(model, pd.DataFrame([row])).iloc[0])
                will_churn = prob >= threshold
                level = risk_level(prob, threshold)
                title, text = RISK_TEXT[level]

                st.plotly_chart(gauge(prob, threshold), use_container_width=True)
                st.markdown(
                    f'<div class="risk-card risk-{level}"><h3>{title}</h3>'
                    f"<p>{text}</p></div>",
                    unsafe_allow_html=True,
                )
                if will_churn:
                    st.error("Customer is **likely to churn**")
                else:
                    st.success("Customer is **unlikely to churn**")

                st.markdown("**Suggested actions**")
                for tip in recommendations(row):
                    st.markdown(f"- {tip}")

                st.session_state.history.append(
                    {
                        "Time": datetime.now().strftime("%H:%M:%S"),
                        **row,
                        "Churn probability": round(prob, 4),
                        "Prediction": "Churn" if will_churn else "Stay",
                    }
                )
                logger.info("Prediction made: %s -> %.3f", row, prob)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Prediction failed")
                st.error(f"Prediction failed: {exc}")
        else:
            st.info("Fill in the customer details and click **Run prediction**.")

    if st.session_state.history:
        st.divider()
        st.subheader("Session history")
        hist = pd.DataFrame(st.session_state.history)
        st.dataframe(hist, use_container_width=True, hide_index=True)
        st.download_button(
            "Download history (CSV)",
            hist.to_csv(index=False).encode(),
            "prediction_history.csv",
            "text/csv",
        )

# --------------------------------------------------------------------------- #
# Tab 2 - Batch prediction
# --------------------------------------------------------------------------- #
with tab_batch:
    st.subheader("Bulk customer scoring")
    st.write(f"Upload a CSV containing these columns: `{', '.join(FEATURES)}`")

    template = pd.DataFrame(
        [[35, 12, 70.0, 1], [52, 48, 95.5, 0], [23, 3, 110.0, 6]], columns=FEATURES
    )
    st.download_button(
        "Download CSV template",
        template.to_csv(index=False).encode(),
        "churn_template.csv",
        "text/csv",
    )

    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        try:
            data = pd.read_csv(file)
            missing = [c for c in FEATURES if c not in data.columns]
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            elif len(data) > MAX_BATCH_ROWS:
                st.error(f"File too large. Maximum {MAX_BATCH_ROWS:,} rows.")
            else:
                feats = data[FEATURES].apply(pd.to_numeric, errors="coerce")
                bad = feats.isna().any(axis=1)
                if bad.any():
                    st.warning(
                        f"{int(bad.sum())} row(s) with missing/non-numeric values were skipped."
                    )
                scored = data.loc[~bad].copy()
                scored["Churn_Probability"] = predict_proba(model, feats.loc[~bad]).round(4)
                scored["Prediction"] = (
                    scored["Churn_Probability"] >= threshold
                ).map({True: "Churn", False: "Stay"})
                scored["Risk"] = scored["Churn_Probability"].apply(
                    lambda p: risk_level(p, threshold).title()
                )

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Customers scored", f"{len(scored):,}")
                n_churn = int((scored["Prediction"] == "Churn").sum())
                c2.metric("Predicted to churn", f"{n_churn:,}")
                c3.metric("Churn rate", f"{n_churn / max(len(scored), 1):.1%}")
                c4.metric("Avg. probability", f"{scored['Churn_Probability'].mean():.1%}")

                cA, cB = st.columns(2)
                with cA:
                    counts = scored["Risk"].value_counts().reindex(
                        ["Low", "Medium", "High"], fill_value=0
                    )
                    fig = go.Figure(
                        go.Bar(
                            x=counts.index,
                            y=counts.values,
                            marker_color=["#16a34a", "#d97706", "#dc2626"],
                        )
                    )
                    fig.update_layout(
                        title="Customers by risk level", height=320,
                        margin=dict(l=10, r=10, t=50, b=10),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                with cB:
                    fig = go.Figure(
                        go.Histogram(x=scored["Churn_Probability"], nbinsx=20,
                                     marker_color="#1d4ed8")
                    )
                    fig.update_layout(
                        title="Churn probability distribution", height=320,
                        margin=dict(l=10, r=10, t=50, b=10),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("**Top 20 highest-risk customers**")
                st.dataframe(
                    scored.sort_values("Churn_Probability", ascending=False).head(20),
                    use_container_width=True, hide_index=True,
                )
                st.download_button(
                    "Download full results (CSV)",
                    scored.to_csv(index=False).encode(),
                    "churn_predictions.csv",
                    "text/csv",
                    type="primary",
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Batch scoring failed")
            st.error(f"Could not process file: {exc}")

# --------------------------------------------------------------------------- #
# Tab 3 - Model insights
# --------------------------------------------------------------------------- #
with tab_insights:
    st.subheader("What drives churn in this model?")
    est = model.steps[-1][1] if hasattr(model, "steps") else model

    importance = None
    if hasattr(est, "feature_importances_"):
        importance = pd.Series(est.feature_importances_, index=FEATURES)
    elif hasattr(est, "coef_"):
        importance = pd.Series(abs(est.coef_).ravel(), index=FEATURES)

    if importance is not None and len(importance) == len(FEATURES):
        importance = importance.sort_values()
        fig = go.Figure(go.Bar(x=importance.values, y=importance.index,
                               orientation="h", marker_color="#1d4ed8"))
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10),
                          xaxis_title="Relative importance")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("This model type does not expose feature importances.")

    st.markdown("**What-if analysis** - see how one feature changes churn risk")
    base = {"Age": 35, "Tenure": 12, "MonthlyCharges": 70.0, "SupportCalls": 1}
    feat = st.selectbox("Feature to vary", FEATURES, index=3)
    ranges = {
        "Age": range(18, 81, 2),
        "Tenure": range(0, 73, 3),
        "MonthlyCharges": range(10, 201, 10),
        "SupportCalls": range(0, 11),
    }
    sweep = pd.DataFrame([{**base, feat: v} for v in ranges[feat]])
    sweep["Churn probability"] = predict_proba(model, sweep).values
    fig = go.Figure(go.Scatter(x=sweep[feat], y=sweep["Churn probability"],
                               mode="lines+markers", line=dict(color="#ef4444")))
    fig.update_layout(height=320, xaxis_title=feat, yaxis_title="Churn probability",
                      yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Other features held at: {', '.join(f'{k}={v}' for k, v in base.items() if k != feat)}")

# --------------------------------------------------------------------------- #
# Tab 4 - About
# --------------------------------------------------------------------------- #
with tab_about:
    st.subheader("About this project")
    st.markdown(
        """
        **{APP_NAME}** is an end-to-end machine learning application that predicts whether a telecom/subscription
        customer is likely to leave, so retention teams can act early.

        **Features**
        - Real-time single-customer prediction with probability gauge and risk level
        - Adjustable decision threshold to balance precision vs. recall
        - Batch scoring from CSV with validation, charts and downloadable results
        - Model insights: feature importance and what-if analysis
        - Cached model loading, input validation, structured logging and error handling

        **Technology stack:** Python, scikit-learn, pandas, Streamlit, Plotly, Docker
        """.replace("{APP_NAME}", APP_NAME)
    )
    st.markdown(
        f'<div class="dev-card"><small>Designed and developed by</small>'
        f"<b>{DEVELOPER}</b></div>",
        unsafe_allow_html=True,
    )
    st.caption("Predictions are probabilistic and should support, not replace, business judgement.")

st.divider()
st.markdown(
    f'<div class="footer-dev">{APP_NAME} &nbsp;|&nbsp; Designed and developed by <b>{DEVELOPER}</b> &nbsp;|&nbsp; '
    f"© {datetime.now().year}</div>",
    unsafe_allow_html=True,
)
