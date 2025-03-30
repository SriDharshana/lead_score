import pickle
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocess import preprocess_data
training_progress = []
# Load and preprocess data
X_train, X_test, y_train, y_test, scaler = preprocess_data("data/customer_data.csv")

# Define the parameter grid
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}

# Initialize RandomForestClassifier
rf = RandomForestClassifier()

# GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Train final model with best parameters
best_rf = grid_search.best_estimator_

# Predictions
y_pred = best_rf.predict(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="binary")  # Change if multi-class
recall = recall_score(y_test, y_pred, average="binary")  # Change if multi-class
f1 = f1_score(y_test, y_pred, average="binary")  # Change if multi-class

# Save model performance metrics to CSV
df_metrics = pd.DataFrame([{
    "Model": "RandomForest (Tuned)",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1,
    "Best Params": str(grid_search.best_params_)  # Convert to string for CSV
}])
for i, n_estimators in enumerate([10, 50, 100, 200]):  
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
    acc = rf.score(X_test, y_test)
    training_progress.append({"epoch": i+1, "n_estimators": n_estimators, "accuracy": acc})

df_epochs = pd.DataFrame(training_progress)
df_epochs.to_csv("training_progress.csv", index=False)
print("Training progress saved")
df_metrics.to_csv("model_performance.csv", index=False)
print("Model Performance Saved to CSV ")

# Ensure 'models/' directory exists
os.makedirs("models", exist_ok=True)

# Save the trained model
with open("models/lead_conversion_model.pkl", "wb") as f:
    pickle.dump(best_rf, f)

# Save the scaler
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model and Scaler saved successfully!")
