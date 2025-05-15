# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# import pickle
# import argparse

# # Accept CSV file as a command-line argument
# parser = argparse.ArgumentParser()
# parser.add_argument('--data_path', required=True)
# args = parser.parse_args()

# # Load the dataset
# df = pd.read_csv(args.data_path, sep='\t')  # if tab-separated # if using semicolon

# print("Available columns:", df.columns.tolist())
# # Basic preprocessing
# X = df.drop(columns=["medv"])  # Assuming "medv" is the target column
# y = df["medv"]



# # Train/test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Model training
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Save the trained model
# with open("models/trained_model.pkl", "wb") as f:
#     pickle.dump(model, f)

# import pandas as pd
# import requests
# from io import StringIO
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# import pickle
# import argparse
# import os

# def download_csv_from_drive(url):
#     """Download CSV from Google Drive (public link or using gdown)."""
#     if "drive.google.com" in url:
#         # If using a direct download link (requires file to be publicly accessible)
#         file_id = url.split("/d/")[1].split("/")[0]
#         download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
#         response = requests.get(download_url)
#         response.raise_for_status()  # Check for errors
#         return StringIO(response.text)
#     else:
#         raise ValueError("Unsupported URL. Use a Google Drive public link.")

# # Accept CSV URL as a command-line argument
# parser = argparse.ArgumentParser()
# parser.add_argument('--data_url', required=True, help="Google Drive CSV URL")
# args = parser.parse_args()

# # Download and load the dataset
# try:
#     csv_data = download_csv_from_drive(args.data_url)
#     df = pd.read_csv(csv_data)
# except Exception as e:
#     print(f"Error loading CSV: {e}")
#     raise

# # Rest of the code remains the same...
# X = df.drop(columns=["medv"])  # MEDV is the target
# y = df["medv"]

# # Train/test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Model training
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Save the trained model
# os.makedirs("models", exist_ok=True)  # Ensure dir exists
# with open("models/trained_model.pkl", "wb") as f:
#     pickle.dump(model, f)

# # Print model performance
# accuracy = model.score(X_test, y_test)
# print(f"Model accuracy: {accuracy:.2f}")
# # # Print model performance (if you need to evaluate it)
# # accuracy = model.score(X_test, y_test)
# # print(f"Model accuracy: {accuracy:.2f}")


# train1.py
import argparse
import pandas as pd
import os
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', required=True, help='Path or URL to CSV file')
args = parser.parse_args()

# Download the file if it's a URL
if args.data_path.startswith('http'):
    print(f"Downloading CSV from {args.data_path}...")
    csv_path = "downloaded_dataset.csv"
    urllib.request.urlretrieve(args.data_path, csv_path)
else:
    csv_path = args.data_path

# Load dataset
df = pd.read_csv(csv_path)

# Example preprocessing
print("Columns:", df.columns)
X = df.drop(columns=["medv"])  # Replace with your actual feature columns
y = df["medv"]                 # Replace with your actual target column

# Dummy model training
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)

# Save model
import joblib
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/trained_model.pkl")
print("Model saved to models/trained_model.pkl")

print("finished")
