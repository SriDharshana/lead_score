import pickle
import numpy as np
import pandas as pd
import joblib

# Load model & scaler
model = joblib.load("models/lead_conversion_model.pkl")
scaler = joblib.load("models/scaler.pkl")

EXPECTED_FEATURES = ['age', 'monthly_income', 'past_purchases', 'last_interaction_days',
                     'lead_source_Email', 'lead_source_Referral', 'lead_source_Website']

def preprocess_input(user_input):
    df = pd.DataFrame([user_input])

    # Handle missing 'lead_source' column
    if 'lead_source' not in df.columns:
        df['lead_source'] = 'Unknown'

    # One-hot encode 'lead_source'
    df = pd.get_dummies(df, columns=['lead_source'])

    # Ensure expected features exist
    for col in EXPECTED_FEATURES:
        if col not in df:
            df[col] = 0

    return df[EXPECTED_FEATURES].values.flatten()


def predict_lead_conversion(user_input):
    features = preprocess_input(user_input)
    features_scaled = scaler.transform([features])
    
    probability = model.predict_proba(features_scaled)[:, 1][0] 
    
    threshold = 0.3  
    prediction = 1 if probability >= threshold else 0
    return prediction, probability

# Example prediction
# user_input = {'age': 30, 'monthly_income': 80000, 'past_purchases': 100, 'last_interaction_days': 0, 'lead_source': 'Website'}
# pred, prob = predict_lead_conversion(user_input)

# print(f"Prediction: {'Converted' if pred == 1 else 'Not Converted'} (Probability: {prob:.2f})")

# # Store incorrect predictions
# feedback = input("Is this prediction correct? (yes/no): ").strip().lower()
# if feedback == "no":
#     df_feedback = pd.DataFrame([user_input])
#     df_feedback['actual_conversion'] = int(not pred)
#     df_feedback.to_csv("feedback_data.csv", mode='a', index=False, header=False)
#     print("Incorrect prediction saved for retraining.")
