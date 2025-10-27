# app.py
import streamlit as st
import boto3
import json

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
st.set_page_config(page_title="Netflix Revenue Optimizer", layout="centered")
st.title("📊 Netflix Revenue Optimization")

st.markdown("""
Enter the scenario parameters to predict **expected revenue**.
""")

subscribers = st.number_input("Number of Subscribers", min_value=1000, max_value=100_000_000, value=5_000_000, step=1000)
avg_watch_hours = st.number_input("Average Watch Hours per Week", min_value=0.0, max_value=50.0, value=5.0, step=0.1)
monthly_fee = st.number_input("Monthly Fee ($)", min_value=1.0, max_value=50.0, value=12.99, step=0.1)
marketing_spend = st.number_input("Marketing Spend ($)", min_value=0.0, max_value=1_000_000.0, value=200_000.0, step=1000.0)

if st.button("Predict Revenue"):
    with st.spinner("Invoking SageMaker endpoint..."):
        try:
            revenue = predict_revenue(subscribers, avg_watch_hours, monthly_fee, marketing_spend)
            st.success(f"💰 Predicted Revenue: ${revenue:,.2f}")
        except Exception as e:
            st.error(f"❌ Error invoking endpoint: {e}")
            st.info("Make sure your AWS credentials are configured and endpoint name is correct.")
