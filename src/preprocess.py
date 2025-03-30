import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    
    # Handle missing values
    df.fillna(0, inplace=True)

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=['lead_source'], drop_first=True)

    # Drop unnecessary columns
    df.drop(columns=['CustomerID'], inplace=True)

    # Define features and target variable
    X = df.drop(columns=['Converted'])
    y = df['Converted']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler
