# =============================================================================
# Netflix Revenue Prediction Tool - app2.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os

# -----------------------------
# 1️⃣ App Header
# -----------------------------
st.title("📈 Netflix Revenue Prediction Tool")
st.write("""
Predict expected Netflix revenue based on:
- Subscribers
- Average Watch Hours
- Monthly Fee
- Marketing Spend
""")

# -----------------------------
# 2️⃣ Load or Generate Model
# -----------------------------
MODEL_PATH = "revenue_optimization_model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    st.success("✅ Loaded existing revenue model.")
else:
    st.warning("⚠️ Model not found! Generating a demo model for predictions...")
    # Generate demo dataset
    np.random.seed(42)
    demo_data = pd.DataFrame({
        "subscribers": np.random.randint(100_000, 10_000_000, 200),
        "avg_watch_hours": np.random.uniform(1, 10, 200),
        "monthly_fee": np.random.uniform(5, 20, 200),
        "marketing_spend": np.random.uniform(10_000, 500_000, 200),
    })
    demo_data["revenue"] = demo_data["subscribers"] * demo_data["monthly_fee"] * np.random.uniform(0.95, 1.05, 200)
    
    X = demo_data[["subscribers", "avg_watch_hours", "monthly_fee", "marketing_spend"]]
    y = demo_data["revenue"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    st.info(f"Demo model trained with R² score: {r2:.4f}")
    joblib.dump(model, MODEL_PATH)
    st.success("✅ Demo model saved for future use.")

# -----------------------------
# 3️⃣ Single Prediction Form
# -----------------------------
st.header("💻 Single Prediction")

subscribers = st.number_input("Subscribers", min_value=0, value=1_000_000, step=1000)
avg_watch_hours = st.number_input("Average Watch Hours", min_value=0.0, value=5.0, step=0.1)
monthly_fee = st.number_input("Monthly Fee ($)", min_value=0.0, value=12.99, step=0.01)
marketing_spend = st.number_input("Marketing Spend ($)", min_value=0.0, value=100_000.0, step=1000.0)

if st.button("Predict Revenue"):
    input_df = pd.DataFrame([{
        "subscribers": subscribers,
        "avg_watch_hours": avg_watch_hours,
        "monthly_fee": monthly_fee,
        "marketing_spend": marketing_spend
    }])
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted Revenue: ${prediction:,.2f}")

# -----------------------------
# 4️⃣ Batch Prediction from CSV
# -----------------------------
st.header("📁 Batch Prediction from CSV")
st.write("Upload CSV with columns: subscribers, avg_watch_hours, monthly_fee, marketing_spend")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    required_cols = ["subscribers", "avg_watch_hours", "monthly_fee", "marketing_spend"]
    if all(col in df.columns for col in required_cols):
        predictions = model.predict(df[required_cols])
        df["predicted_revenue"] = predictions
        st.success("✅ Predictions generated!")
        st.dataframe(df.head(10))
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions CSV", data=csv, file_name="predicted_revenue.csv", mime="text/csv")
    else:
        st.error(f"❌ CSV must contain these columns: {required_cols}")

# -----------------------------
# 5️⃣ Footer
# -----------------------------
st.markdown("---")
st.markdown("Developed for Netflix Revenue Prediction Demo")
