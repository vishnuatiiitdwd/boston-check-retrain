import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import argparse

# Accept CSV file as a command-line argument
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', required=True)
args = parser.parse_args()

# Load the dataset
df = pd.read_csv(args.data_path, sep='\t')  # if tab-separated # if using semicolon

print("Available columns:", df.columns.tolist())
# Basic preprocessing
X = df.drop(columns=["medv"])  # Assuming "medv" is the target column
y = df["medv"]



# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
with open("models/trained_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Print model performance (if you need to evaluate it)
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")
