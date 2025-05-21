# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load dataset
df = pd.read_csv("loan_a.csv")

# Basic preprocessing
df = df.dropna()
df.replace({'Loan_Status': {'Y': 1, 'N': 0}}, inplace=True)

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Features and Target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train using ID3 (Entropy-based Decision Tree)
model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(X_train, y_train)

# Save model and feature columns
joblib.dump(model, "loan_model.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")
