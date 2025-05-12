import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import os

# Load the CSV manually
df = pd.read_csv("submission_example.csv")  # Make sure this file is committed

# Basic preprocessing
X = df.drop(columns=["medv"])  # MEDV is the target
y = df["medv"]

print("result")
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
os.makedirs("models", exist_ok=True)
with open("models/trained_model.pkl", "wb") as f:
    pickle.dump(model, f)
