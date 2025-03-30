import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from src import predict_model

st.title("Lead Conversion Prediction Dashboard")

try:
    df_training = pd.read_csv("training_progress.csv")
except FileNotFoundError:
    st.error("Error: 'training_progress.csv' not found!")
    df_training = None  

try:
    model = joblib.load("models/lead_conversion_model.pkl")
except FileNotFoundError:
    st.error("Error: Model file not found!")
    model = None

st.subheader("Enter Customer Details")
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Monthly Income", min_value=0, value=50000)
purchases = st.number_input("Past Purchases", min_value=0, value=5)
interaction = st.number_input("Days Since Last Interaction", min_value=0, value=10)
lead_source = st.selectbox("Lead Source", ["Email", "Referral", "Website"])

user_data = {
    'age': age,
    'monthly_income': income,
    'past_purchases': purchases,
    'last_interaction_days': interaction,
    'lead_source': lead_source
}

if st.button("Predict"):
    if model:
        pred, prob = predict_model.predict_lead_conversion(user_data)
        st.write(f"### Prediction: {'Converted' if pred == 1 else 'Not Converted'}")
        st.write(f"### Probability: {prob:.2f}")
    else:
        st.error("Prediction Model not loaded. Check the model file.")


st.subheader("Model Accuracy Over Time")
if df_training is not None:
    if "epoch" in df_training.columns and "accuracy" in df_training.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df_training['epoch'], df_training['accuracy'], marker='o', linestyle='-', color='b', label="Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Model Accuracy Over Time")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.error("Error: 'epoch' or 'accuracy' column missing in 'training_progress.csv'. Check your training script.")


st.title("Data Visualization")

st.markdown("### Upload your dataset CSV file for analysis")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    try:
        df_viz = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        df_viz = None

    if df_viz is not None:
        # Univariate Analysis
        st.subheader("Univariate Analysis")
        numeric_cols = df_viz.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            st.write(f"#### Distribution of {col}")
            fig, ax = plt.subplots()
            sns.histplot(df_viz[col], bins=20, kde=True, ax=ax)
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        # Bivariate Analysis
        st.subheader("Bivariate Analysis")
        if len(numeric_cols) >= 2:
            x_var = st.selectbox("Select X variable", numeric_cols)
            default_y = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
            y_var = st.selectbox("Select Y variable", numeric_cols, index=numeric_cols.get_loc(default_y))
            if x_var and y_var:
                st.write(f"#### Scatter Plot: {x_var} vs {y_var}")
                fig, ax = plt.subplots()
                sns.scatterplot(x=df_viz[x_var], y=df_viz[y_var], ax=ax)
                ax.set_xlabel(x_var)
                ax.set_ylabel(y_var)
                st.pyplot(fig)
        else:
            st.info("Not enough numeric columns for bivariate analysis.")

st.subheader("Model Performance Table")
try:
    df_model_perf = pd.read_csv("model_performance.csv")
    st.dataframe(df_model_perf)
except FileNotFoundError:
    st.error("Error: 'model_performance.csv' not found!")
