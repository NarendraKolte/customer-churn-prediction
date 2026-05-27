import streamlit as st
import requests

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("📉 Customer Churn Prediction Dashboard")
st.write("Enter customer details to predict churn risk")

frequency = st.number_input("Purchase Frequency", min_value=1, max_value=100, value=2)
monetary = st.number_input("Total Monetary Value", min_value=0.0, value=80.5)
segment = st.selectbox(
    "Customer Segment",
    ["Champions", "Loyal Customers", "Potential Loyalist", "At Risk", "Lost"]
)

if st.button("Predict Churn"):
    payload = {
        "frequency": frequency,
        "monetary": monetary,
        "segment": segment
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict_churn",
            json=payload
        )

        if response.status_code == 200:
            result = response.json()
            st.success("Prediction Successful")
            st.metric("Churn Probability", result["churn_probability"])
            st.metric("Churn Risk", result["churn_risk"])
            st.write("### Recommended Action")
            st.info(result["recommended_action"])
        else:
            st.error("API error. Make sure FastAPI is running.")

    except Exception as e:
        st.error("Cannot connect to FastAPI backend.")
