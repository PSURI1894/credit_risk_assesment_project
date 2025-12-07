# ============================================================
# Streamlit App - Credit Risk Assessment
# ============================================================

import numpy as np
import pandas as pd
import streamlit as st
import joblib

from pathlib import Path

# ------------------------------
# 1. Load artifacts
# ------------------------------

@st.cache_resource
def load_artifacts():
    """Load clean dataset, model, scaler, and label encoders."""
    data_path = Path("clean_credit_risk_dataset.csv")
    model_path = Path("credit_risk_rf_model.pkl")
    scaler_path = Path("credit_risk_scaler.pkl")
    encoders_path = Path("credit_risk_label_encoders.pkl")

    if not data_path.exists():
        st.error("clean_credit_risk_dataset.csv not found. Run the notebook cell that saves it.")
        st.stop()

    if not model_path.exists():
        st.error("credit_risk_rf_model.pkl not found. Run the notebook training cells.")
        st.stop()

    df = pd.read_csv(data_path)
    model = joblib.load(model_path)

    # Scaler is not used for RandomForest (trained on unscaled features),
    # but we load it anyway in case we extend the app later.
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    encoders = joblib.load(encoders_path) if encoders_path.exists() else None

    return df, model, scaler, encoders


df, model, scaler, encoders = load_artifacts()

# ------------------------------
# 2. Basic config
# ------------------------------

st.set_page_config(
    page_title="Credit Risk Assessment",
    page_icon="💳",
    layout="wide"
)

st.title("💳 AI-Based Credit Risk Assessment Dashboard")

st.markdown(
    """
This app uses a **Random Forest credit risk model** trained on application and credit history data.

You can:

- 🔍 Explore **existing customers** from the dataset and see their predicted default risk  
- 🧪 Run a **What-if Simulation** by tweaking key features  
- ℹ️ View **model info** and feature importance
"""
)

TARGET_COL = "default_flag"

# Identify feature columns (must match training pipeline)
cols_to_drop = [TARGET_COL]
if "ID" in df.columns:
    cols_to_drop.append("ID")
if "AGE_BIN" in df.columns:
    cols_to_drop.append("AGE_BIN")

feature_cols = [c for c in df.columns if c not in cols_to_drop]

# For the simulator we will use median template
median_row = df[feature_cols].median(numeric_only=True)
mode_row = df[feature_cols].mode().iloc[0]
template_row = median_row.copy()
template_row.update(mode_row)


# ------------------------------
# 3. Utility functions
# ------------------------------

def predict_proba_from_row(row: pd.Series) -> float:
    """
    Take a row with all feature columns, reshape to 2D and predict default probability.
    """
    X = row[feature_cols].to_frame().T  # 1 x n_features
    prob = model.predict_proba(X)[:, 1][0]
    return float(prob)


def get_risk_bucket(prob: float) -> str:
    """
    Map probability to human-readable risk bucket.
    """
    if prob >= 0.6:
        return "🔥 High Risk"
    elif prob >= 0.3:
        return "⚠️ Medium Risk"
    else:
        return "✅ Low Risk"


# ------------------------------
# 4. Sidebar - navigation
# ------------------------------

st.sidebar.header("Navigation")
mode = st.sidebar.radio(
    "Choose a mode:",
    ["Explore Existing Customers", "What-if Simulation", "Model Info"]
)


# ============================================================
# Mode 1: Explore Existing Customers
# ============================================================

if mode == "Explore Existing Customers":
    st.subheader("🔍 Explore Existing Customers")

    if "ID" in df.columns:
        id_list = df["ID"].unique()
        selected_id = st.selectbox(
            "Select customer ID",
            options=sorted(id_list)
        )
        customer_df = df[df["ID"] == selected_id].copy()
        if customer_df.empty:
            st.warning("No rows found for this ID.")
            st.stop()
        row = customer_df.iloc[0]
    else:
        # Fallback: choose by index
        st.info("No ID column detected. Selecting by row index.")
        max_idx = len(df) - 1
        idx = st.slider("Choose row index", 0, max_idx, 0)
        row = df.iloc[idx]

    # Show raw info
    st.markdown("### Customer Snapshot")
    # Show some human-readable subset
    cols_to_show = [
        col for col in df.columns
        if col not in [TARGET_COL, "credit_history_oldest_month",
                       "credit_history_newest_month", "AGE_BIN"]
    ]
    st.dataframe(row[cols_to_show].to_frame("value"))

    # Prediction
    prob = predict_proba_from_row(row)
    bucket = get_risk_bucket(prob)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Estimated Default Probability", f"{prob:.2%}")
    with col2:
        st.metric("Risk Category", bucket)
    with col3:
        if TARGET_COL in row:
            actual = "Defaulted (1)" if row[TARGET_COL] == 1 else "No default (0)"
            st.metric("Actual Label (dataset)", actual)

    # Show the most important features for this model globally
    st.markdown("### Top Global Features (from training)")
    # Compute feature importances once and cache in session
    if "feature_importances" not in st.session_state:
        importances = pd.Series(
            model.feature_importances_,
            index=feature_cols
        ).sort_values(ascending=False)
        st.session_state["feature_importances"] = importances
    else:
        importances = st.session_state["feature_importances"]

    st.bar_chart(importances.head(10))


# ============================================================
# Mode 2: What-if Simulation
# ============================================================

elif mode == "What-if Simulation":
    st.subheader("🧪 What-if Simulation")

    st.markdown(
        """
    Here we simulate a **hypothetical applicant**.

    We use the dataset medians/modes as a base template and let you tweak key features:
    """
    )

    # We'll start from the template_row and let user override some important fields
    sim = template_row.copy()

    # Layout: two columns for inputs
    left, right = st.columns(2)

    # Some important numeric features if they exist
    with left:
        if "AGE_YEARS" in feature_cols:
            sim["AGE_YEARS"] = st.slider("Age (years)", 18, 80, int(sim["AGE_YEARS"]))

        if "AMT_INCOME_TOTAL_CLIPPED" in feature_cols:
            sim["AMT_INCOME_TOTAL_CLIPPED"] = st.number_input(
                "Annual Income (clipped)",
                min_value=1000.0,
                max_value=5_000_000.0,
                value=float(sim["AMT_INCOME_TOTAL_CLIPPED"]),
                step=1000.0
            )

        if "CNT_CHILDREN" in feature_cols:
            sim["CNT_CHILDREN"] = st.number_input(
                "Number of Children",
                min_value=0,
                max_value=10,
                value=int(sim["CNT_CHILDREN"])
            )

        if "CNT_FAM_MEMBERS" in feature_cols:
            sim["CNT_FAM_MEMBERS"] = st.number_input(
                "Family Members",
                min_value=1,
                max_value=15,
                value=int(sim["CNT_FAM_MEMBERS"])
            )

    with right:
        # Credit behavior knobs
        if "max_delay_severity" in feature_cols:
            sim["max_delay_severity"] = st.slider(
                "Max Delay Severity (0–5)",
                0, 5, int(sim["max_delay_severity"])
            )

        if "delay_ratio" in feature_cols:
            sim["delay_ratio"] = st.slider(
                "Fraction of months with delay",
                0.0, 1.0, float(sim["delay_ratio"]),
                step=0.01
            )

        if "credit_history_length" in feature_cols:
            sim["credit_history_length"] = st.slider(
                "Credit History Length (months)",
                0, 120, int(sim["credit_history_length"])
            )

        if "months_with_records" in feature_cols:
            sim["months_with_records"] = st.slider(
                "Months with any credit record",
                0, 120, int(sim["months_with_records"])
            )

    # Derived fields: recompute ratios if possible
    if {"AMT_INCOME_TOTAL_CLIPPED", "CNT_CHILDREN"}.issubset(feature_cols):
        sim["INCOME_PER_CHILD"] = (
            sim["AMT_INCOME_TOTAL_CLIPPED"] / (1 + sim.get("CNT_CHILDREN", 0))
        )

    if {"AMT_INCOME_TOTAL_CLIPPED", "CNT_FAM_MEMBERS"}.issubset(feature_cols):
        sim["INCOME_PER_FAM_MEMBER"] = (
            sim["AMT_INCOME_TOTAL_CLIPPED"] / max(sim.get("CNT_FAM_MEMBERS", 1), 1)
        )

    st.markdown("### Simulated Applicant Feature Vector (subset)")
    st.dataframe(sim.to_frame("value").loc[
        [c for c in sim.index if c in [
            "AGE_YEARS",
            "AMT_INCOME_TOTAL_CLIPPED",
            "CNT_CHILDREN",
            "CNT_FAM_MEMBERS",
            "INCOME_PER_CHILD",
            "INCOME_PER_FAM_MEMBER",
            "max_delay_severity",
            "delay_ratio",
            "credit_history_length",
            "months_with_records"
        ] if c in sim.index]
    ])

    if st.button("Predict Risk for Simulated Applicant"):
        prob = predict_proba_from_row(sim)
        bucket = get_risk_bucket(prob)

        st.success(f"Estimated default probability: **{prob:.2%}**")
        st.info(f"Risk bucket: **{bucket}**")

        st.caption(
            "Higher values of `max_delay_severity` and `delay_ratio` generally increase the risk."
        )


# ============================================================
# Mode 3: Model Info
# ============================================================

elif mode == "Model Info":
    st.subheader("ℹ️ Model & Dataset Info")

    st.markdown(
        f"""
        - Model type: **RandomForestClassifier**  
        - Number of features: **{len(feature_cols)}**  
        - Number of records in clean dataset: **{len(df)}**  
        - Target column: `{TARGET_COL}`  
        """
    )

    st.markdown("### Target Distribution")
    target_counts = df[TARGET_COL].value_counts()
    st.bar_chart(target_counts)

    st.markdown("### Example Feature Columns")
    st.write(feature_cols[:30])

    st.markdown(
        """
        **Training pipeline (high-level):**
        1. Built `default_flag` from monthly `STATUS` in `credit_record`  
        2. Engineered application features like `AGE_YEARS`, `YEARS_EMPLOYED`, `INCOME_PER_*`  
        3. Aggregated credit behavior: `num_delay_months`, `delay_ratio`, `max_delay_severity`, etc.  
        4. Handled class imbalance using SMOTE  
        5. Trained Random Forest with class weights and evaluated on a held-out test set  
        6. Used SHAP in the notebook for global & local interpretability  
        """
    )
