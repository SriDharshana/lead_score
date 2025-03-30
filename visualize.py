import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from src import predict_model

st.title("Lead Conversion Prediction Dashboard")
try:
    df = pd.read_csv("training_progress.csv")
except FileNotFoundError:
    st.error("Error: 'training_progress.csv' not found!")
    df = None  

try:
    model = joblib.load("models/lead_conversion_model.pkl")
except FileNotFoundError:
    st.error("Error: Model file not found!")
    model = None

# User Input Form
st.subheader("Enter Customer Details")
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Monthly Income", min_value=0, value=50000)
purchases = st.number_input("Past Purchases", min_value=0, value=5)
interaction = st.number_input("Days Since Last Interaction", min_value=0, value=10)
lead_source = st.selectbox("Lead Source", ["Email", "Referral", "Website"])

# Prepare user data
user_data = {
    'age': age,
    'monthly_income': income,
    'past_purchases': purchases,
    'last_interaction_days': interaction,
    'lead_source': lead_source
}

# Prediction Button
if st.button("Predict"):
    if model:
        pred, prob = predict_model.predict_lead_conversion(user_data)
        st.write(f"### Prediction: {'Converted' if pred == 1 else 'Not Converted'}")
        st.write(f"### Probability: {prob:.2f}")
    else:
        st.error("Prediction Model not loaded. Check the model file.")

# Model Performance Graph
st.subheader("Model Accuracy Over Time")

# Load performance CSV
try:
    df_perf = pd.read_csv("training_progress.csv")
    if "epoch" in df_perf.columns and "accuracy" in df_perf.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(df_perf['epoch'], df_perf['accuracy'], marker='o', linestyle='-', color='b', label="Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Model Accuracy Over Time")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)
    else:
        st.error("Error: 'epoch' column missing in 'training_progress.csv'. Check your training script.")
except FileNotFoundError:
    st.error("Error: 'training_progress.csv' not found!")

