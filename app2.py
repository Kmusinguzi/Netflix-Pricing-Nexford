# app2.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Netflix Revenue Predictor", page_icon="📈")

st.title("📊 Netflix Revenue Prediction Tool")
st.write(
    """
    This app predicts expected revenue for Netflix based on:
    - Subscribers
    - Average Watch Hours
    - Monthly Fee
    - Marketing Spend
    """
)

# -----------------------------
# 1️⃣ Load model
# -----------------------------
MODEL_PATH = os.path.join("model", "revenue_optimization_model.pkl")

if not os.path.exists(MODEL_PATH):
    st.warning("Model not found! Generating a demo model for predictions...")
    # Create demo model if missing
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    np.random.seed(42)
    demo_data = pd.DataFrame({
        "subscribers": np.random.randint(100_000, 10_000_000, 100),
        "avg_watch_hours": np.random.uniform(1, 10, 100),
        "monthly_fee": np.random.uniform(5, 20, 100),
        "marketing_spend": np.random.uniform(10_000, 500_000, 100),
    })
    demo_data["revenue"] = demo_data["subscribers"] * demo_data["monthly_fee"] * np.random.uniform(0.95, 1.05, 100)

    X = demo_data[["subscribers", "avg_watch_hours", "monthly_fee", "marketing_spend"]]
    y = demo_data["revenue"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
else:
    model = joblib.load(MODEL_PATH)

# -----------------------------
# 2️⃣ User Input
# -----------------------------
st.sidebar.header("Enter Netflix Parameters")
subscribers = st.sidebar.number_input("Subscribers", min_value=0, value=5_000_000, step=100_000)
avg_watch_hours = st.sidebar.number_input("Average Watch Hours per Week", min_value=0.0, value=5.5, step=0.1)
monthly_fee = st.sidebar.number_input("Monthly Fee ($)", min_value=0.0, value=12.99, step=0.1)
marketing_spend = st.sidebar.number_input("Marketing Spend ($)", min_value=0.0, value=200_000.0, step=1_000.0)

input_df = pd.DataFrame({
    "subscribers": [subscribers],
    "avg_watch_hours": [avg_watch_hours],
    "monthly_fee": [monthly_fee],
    "marketing_spend": [marketing_spend]
})

# -----------------------------
# 3️⃣ Prediction
# -----------------------------
if st.button("Predict Revenue"):
    revenue_pred = model.predict(input_df)[0]
    st.success(f"💰 Predicted Revenue: ${revenue_pred:,.2f}")

# -----------------------------
# 4️⃣ Option to upload CSV for batch prediction
# -----------------------------
st.subheader("Batch Prediction from CSV")
uploaded_file = st.file_uploader("Upload CSV with columns: subscribers, avg_watch_hours, monthly_fee, marketing_spend", type="csv")

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)
        if all(col in batch_df.columns for col in ["subscribers", "avg_watch_hours", "monthly_fee", "marketing_spend"]):
            batch_preds = model.predict(batch_df)
            batch_df["predicted_revenue"] = batch_preds
            st.dataframe(batch_df)
            st.download_button("Download Predictions CSV", batch_df.to_csv(index=False), "predictions.csv")
        else:
            st.error("CSV missing required columns.")
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
