"""
Customer Lifetime Value (CLV) Prediction — Streamlit App

Run from the project root with:
    streamlit run app.py

This app is a thin UI layer over src/predict.py. It does not
reimplement any prediction or segmentation logic — every number
shown here comes from the same model and the same segment
thresholds used in training, so this app can never silently
disagree with the notebooks or the model card.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config
from src.predict import ModelNotTrainedError, load_segment_thresholds, predict_clv

# -----------------------------
# App Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Lifetime Value Predictor",
    page_icon="📈",
    layout="wide",
)

st.title("📊 Customer Lifetime Value (CLV) Prediction")
st.caption(
    "Predicts 6-month forward customer revenue (£ GBP) from behavioral purchase history. "
    "All figures on this page use the same model and thresholds documented in the model card."
)


# -----------------------------
# Cached Resources
# -----------------------------
@st.cache_resource
def get_model_card() -> dict:
    import json
    path = config.MODEL_DIR / config.MODEL_CARD_FILE
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_resource
def get_segment_thresholds() -> dict:
    try:
        return load_segment_thresholds()
    except ModelNotTrainedError:
        return {}


@st.cache_data
def get_scored_customers() -> pd.DataFrame:
    """Load the full-population scored customer file produced by train/scoring, for lookup + context."""
    path = config.DATASET_DIR / config.PREDICTIONS_FILE
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def check_model_ready() -> bool:
    model_file = config.MODEL_DIR / config.MODEL_FILE
    if not model_file.exists():
        st.error("🚨 No trained model found. Please run `python -m src.train` first.")
        st.code("python -m src.train")
        return False
    return True


def show_prediction_result(predicted_clv: float, segment: str, scored_customers: pd.DataFrame):
    color_map = {"Low Value": "🔴", "Medium Value": "🟠", "High Value": "🟢"}
    color = color_map.get(segment, "⚪")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Predicted 6-Month CLV", value=f"£{predicted_clv:,.2f}")
    with col2:
        st.markdown(f"### {color} Segment: **{segment}**")

    if not scored_customers.empty:
        percentile = (scored_customers["Predicted_CLV"] < predicted_clv).mean() * 100
        st.info(f"This prediction is higher than **{percentile:.0f}%** of the {len(scored_customers):,} scored customers in the training population.")


# -----------------------------
# Tabs
# -----------------------------
tab_lookup, tab_whatif, tab_batch, tab_model = st.tabs(
    ["🔍 Customer Lookup", "🧪 What-If Scenario", "📤 Batch Upload", "ℹ️ Model Info"]
)

# ============================================================
# TAB 1 — Customer Lookup
# ============================================================
with tab_lookup:
    st.subheader("Look up an existing customer")
    st.write("Select a real `CustomerID` from the trained population to see their actual behavioral profile and predicted CLV.")

    scored_customers = get_scored_customers()

    if scored_customers.empty:
        st.warning("No scored customer file found. Run `python -m src.train` (Notebook 3 also produces this).")
    else:
        customer_id = st.selectbox(
            "Customer ID",
            options=scored_customers["CustomerID"].sort_values().tolist(),
        )
        row = scored_customers[scored_customers["CustomerID"] == customer_id].iloc[0]

        st.markdown("**Behavioral profile (from purchase history):**")
        profile_cols = st.columns(4)
        profile_cols[0].metric("Recency (days)", f"{row['Recency']:.0f}")
        profile_cols[1].metric("Frequency (orders)", f"{row['Frequency']:.0f}")
        profile_cols[2].metric("Monetary (£, historical)", f"£{row['Monetary']:,.2f}")
        profile_cols[3].metric("Avg Order Value", f"£{row['AvgOrderValue']:,.2f}")

        profile_cols2 = st.columns(4)
        profile_cols2[0].metric("Avg Basket Size", f"{row['AvgBasketSize']:.1f}")
        profile_cols2[1].metric("Avg Purchase Interval", f"{row['AvgPurchaseIntervalDays']:.0f} days")
        profile_cols2[2].metric("Tenure", f"{row['TenureDays']:.0f} days")
        profile_cols2[3].metric("Was in test set?", "Yes" if row["WasInTestSet"] else "No")

        st.divider()
        show_prediction_result(row["Predicted_CLV"], row["CLV_Segment"], scored_customers)

        if row["WasInTestSet"]:
            st.caption(
                f"This customer was held out during training. Actual 6-month CLV in the historical "
                f"data was £{row['CLV_6M']:,.2f}, so you can directly compare it to the prediction above."
            )
        else:
            st.caption(
                "This customer's data was used during training, so this prediction is not a fair "
                "test of accuracy — see the Model Info tab for honest, held-out performance figures."
            )

# ============================================================
# TAB 2 — What-If Scenario
# ============================================================
with tab_whatif:
    st.subheader("Manual what-if scenario")
    st.write("Enter a hypothetical customer's behavior to see a predicted CLV. Useful for testing thresholds, not for scoring real customers (use Customer Lookup for that).")

    if check_model_ready():
        c1, c2 = st.columns(2)
        with c1:
            recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=1000, value=30)
            frequency = st.number_input("Frequency (number of orders)", min_value=1, max_value=500, value=5)
            monetary = st.number_input("Monetary (£, total historical spend)", min_value=0.0, max_value=1_000_000.0, value=500.0)
            total_quantity = st.number_input("Total Quantity Purchased", min_value=1, max_value=100_000, value=50)
        with c2:
            avg_basket_size = st.number_input("Avg items per basket", min_value=1.0, max_value=500.0, value=8.0)
            avg_purchase_interval = st.number_input("Avg days between purchases", min_value=1.0, max_value=1000.0, value=30.0)
            tenure_days = st.number_input("Tenure (days since first purchase)", min_value=0, max_value=2000, value=180)

        # AvgOrderValue is a derived field — computing it here instead of asking the
        # user to enter it prevents inconsistent inputs (e.g. AvgOrderValue not
        # matching Monetary / Frequency).
        avg_order_value = monetary / frequency
        st.caption(f"Avg Order Value is auto-computed as Monetary ÷ Frequency = **£{avg_order_value:,.2f}**")

        if st.button("🔮 Predict CLV", type="primary"):
            input_data = pd.DataFrame([{
                "Recency": recency,
                "Frequency": frequency,
                "Monetary": monetary,
                "TotalQuantity": total_quantity,
                "AvgOrderValue": avg_order_value,
                "AvgBasketSize": avg_basket_size,
                "AvgPurchaseIntervalDays": avg_purchase_interval,
                "TenureDays": tenure_days,
            }])

            try:
                result = predict_clv(input_data)
                st.divider()
                show_prediction_result(
                    result["Predicted_CLV"].iloc[0],
                    result["CLV_Segment"].iloc[0],
                    get_scored_customers(),
                )
            except (ValueError, TypeError) as e:
                st.error(f"🚨 Invalid input: {e}")
            except ModelNotTrainedError as e:
                st.error(f"🚨 {e}")

# ============================================================
# TAB 3 — Batch Upload
# ============================================================
with tab_batch:
    st.subheader("Batch scoring")
    st.write(f"Upload a CSV with columns: `{', '.join(config.FEATURE_COLUMNS)}`")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(batch_df):,} rows.")
            st.dataframe(batch_df.head())

            if st.button("🔮 Score All Customers", type="primary"):
                with st.spinner("Scoring..."):
                    results = predict_clv(batch_df)
                st.success(f"Scored {len(results):,} customers.")
                st.dataframe(results)

                csv_bytes = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Results as CSV",
                    data=csv_bytes,
                    file_name="clv_batch_predictions.csv",
                    mime="text/csv",
                )
        except (ValueError, TypeError) as e:
            st.error(f"🚨 Invalid file: {e}")
        except ModelNotTrainedError as e:
            st.error(f"🚨 {e}")
        except Exception as e:
            st.error(f"🚨 Could not read file: {e}")

# ============================================================
# TAB 4 — Model Info
# ============================================================
with tab_model:
    st.subheader("About this model")

    model_card = get_model_card()

    if not model_card:
        st.warning("No model card found. Run `python -m src.train` first.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Model type", model_card.get("model_name", "—"))
        c2.metric("Trained on", model_card.get("training_date", "—"))
        c3.metric("Training customers", f"{model_card.get('n_training_customers', 0):,}")

        st.markdown("#### Held-out test performance")
        metrics = model_card.get("test_set_metrics", {})
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE", f"£{metrics.get('MAE_GBP', 0):,.2f}")
        m2.metric("RMSE", f"£{metrics.get('RMSE_GBP', 0):,.2f}")
        m3.metric("R²", f"{metrics.get('R2', 0):.3f}")
        m4.metric("MAE as % of mean CLV", f"{metrics.get('MAE_as_pct_of_mean_CLV', 0):.1f}%")

        st.markdown("#### Baseline comparison — honest disclosure")
        baseline = model_card.get("baseline_metrics", {})
        beats_baseline = baseline.get("beats_naive_baseline", None)

        b1, b2 = st.columns(2)
        b1.metric("Naive 'past = future' baseline MAE", f"£{baseline.get('naive_past_equals_future_MAE_GBP', 0):,.2f}")
        b2.metric("This model's MAE", f"£{metrics.get('MAE_GBP', 0):,.2f}")

        if beats_baseline is False:
            st.warning(
                f"⚠️ This model does **not** currently outperform the naive baseline of simply "
                f"assuming a customer's next 6 months will match their historical spend "
                f"({baseline.get('improvement_over_naive_baseline_pct', 0):.1f}% vs. baseline). "
                "This is disclosed here deliberately rather than hidden — see Known Limitations below."
            )
        elif beats_baseline is True:
            st.success(
                f"✅ This model outperforms the naive baseline by "
                f"{baseline.get('improvement_over_naive_baseline_pct', 0):.1f}%."
            )

        st.markdown("#### Known limitations")
        for limitation in model_card.get("known_limitations", []):
            st.markdown(f"- {limitation}")

        st.markdown(f"**% of customers with £0 six-month CLV in training data:** {model_card.get('pct_customers_with_zero_target', 0):.1f}%")

        with st.expander("Full model card (raw)"):
            st.json(model_card)
