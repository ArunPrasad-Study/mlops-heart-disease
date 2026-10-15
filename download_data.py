import urllib.request
import pandas as pd
import os

# Create data directory
os.makedirs('data', exist_ok=True)

# Download UCI Heart Disease (Cleveland) dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
file_path = "data/heart.csv"

print("Downloading dataset...")
urllib.request.urlretrieve(url, file_path)

# Define column names
columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", 
           "exang", "oldpeak", "slope", "ca", "thal", "target"]

# Load and clean data
df = pd.read_csv(file_path, names=columns, na_values="?")

# The UCI dataset has missing values represented as '?'. We will drop them for simplicity.
df = df.dropna()

# Save cleaned dataset
df.to_csv("data/heart_cleaned.csv", index=False)
print(f"Dataset downloaded and cleaned. Shape: {df.shape}")