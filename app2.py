# app.py
import streamlit as st
import boto3
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ================================
# CONFIGURATION
# ================================
REGION = "us-east-1"  # Must match your deployment
ENDPOINT_NAME = "netflix-revenue-endpoint-20251026-213645"  # Replace with your unique endpoint

# ================================
# Helper function to invoke SageMaker endpoint
# ================================
def predict_revenue(subscribers, avg_watch_hours, monthly_fee, marketing_spend):
    runtime = boto3.client("sagemaker-runtime", region_name=REGION)

    payload = [{
        "subscribers": subscribers,
        "avg_watch_hours": avg_watch_hours,
        "monthly_fee": monthly_fee,
        "marketing_spend": marketing_spend
    }]

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(payload)
    )

    result = json.loads(response['Body'].read().decode())
    return result[0] if isinstance(result, list) else result

# ================================
# Streamlit App UI
# ================================
st.set_page_config(page_title="Netflix Revenue Optimizer", layout="wide")
st.title("📊 Netflix Revenue Optimization")

st.markdown("""
Enter the scenario parameters to predict **expected revenue** or explore the **revenue surface**.
""")

# --- Input sliders ---
subscribers = st.number_input("Number of Subscribers", min_value=1000, max_value=100_000_000, value=5_000_000, step=1000)
avg_watch_hours = st.number_input("Average Watch Hours per Week", min_value=0.0, max_value=50.0, value=5.0, step=0.1)
monthly_fee = st.number_input("Monthly Fee ($)", min_value=1.0, max_value=50.0, value=12.99, step=0.1)
marketing_spend = st.number_input("Marketing Spend ($)", min_value=0.0, max_value=1_000_000.0, value=200_000.0, step=1000.0)

# --- Predict single scenario ---
if st.button("Predict Revenue"):
    with st.spinner("Invoking SageMaker endpoint..."):
        try:
            revenue = predict_revenue(subscribers, avg_watch_hours, monthly_fee, marketing_spend)
            st.success(f"💰 Predicted Revenue: ${revenue:,.2f}")
        except Exception as e:
            st.error(f"❌ Error invoking endpoint: {e}")
            st.info("Make sure your AWS credentials are configured and endpoint name is correct.")

# --- Revenue Surface Plot ---
st.markdown("---")
st.subheader("3D Revenue Surface: Your Price vs Competitor Price")

price_range = st.slider("Your Price Range ($)", min_value=5.0, max_value=30.0, value=(10.0, 20.0), step=0.5)
comp_price_range = st.slider("Competitor Price Range ($)", min_value=5.0, max_value=30.0, value=(8.0, 25.0), step=0.5)

# Create grid
Price_Grid, CompPrice_Grid = np.meshgrid(np.linspace(*price_range, 20), np.linspace(*comp_price_range, 20))
Revenue_Grid = np.zeros_like(Price_Grid)

# Base input (use current slider values)
base_input = {
    "subscribers": subscribers,
    "avg_watch_hours": avg_watch_hours,
    "marketing_spend": marketing_spend
}

# Compute revenue surface
for i in range(Price_Grid.shape[0]):
    for j in range(Price_Grid.shape[1]):
        test_monthly_fee = Price_Grid[i, j]
        test_comp_price = CompPrice_Grid[i, j]

        try:
            revenue_val = predict_revenue(
                subscribers=base_input["subscribers"],
                avg_watch_hours=base_input["avg_watch_hours"],
                monthly_fee=test_monthly_fee,
                marketing_spend=base_input["marketing_spend"]
            )
        except:
            revenue_val = 0  # fallback in case of endpoint error
        Revenue_Grid[i, j] = revenue_val

# Plotting
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(Price_Grid, CompPrice_Grid, Revenue_Grid, cmap='viridis', edgecolor='none')
ax.set_xlabel('Your Price ($)')
ax.set_ylabel('Competitor Price ($)')
ax.set_zlabel('Expected Revenue ($)')
ax.set_title('Revenue Surface')
fig.colorbar(surf, shrink=0.5, aspect=5, label='Revenue')

st.pyplot(fig)
