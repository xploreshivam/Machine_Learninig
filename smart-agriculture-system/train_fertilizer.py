import pandas as pd
import numpy as np
import urllib.request
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

print("1. Downloading Dataset...")
dataset_url = "https://raw.githubusercontent.com/Lanchavi/AgroTechh/main/Fertilizer%20Prediction.csv"
dataset_path = "datasets/fertilizer.csv"

if not os.path.exists("datasets"):
    os.makedirs("datasets")
if not os.path.exists("models"):
    os.makedirs("models")

urllib.request.urlretrieve(dataset_url, dataset_path)
print("Dataset downloaded successfully!")

print("2. Loading and Preprocessing Data...")
# Read dataset
df = pd.read_csv(dataset_path)

print(f"Dataset columns: {df.columns.tolist()}")


df.columns = df.columns.str.strip()

try:
    X = df[['Nitrogen', 'Phosphorous', 'Potassium', 'Crop Type']]
    y = df['Fertilizer Name']
except KeyError as e:
    print(f"Error, missing columns: {e}")
    print("Printing sample to debug...")
    print(df.head())
    exit()

# We need to encode the 'Crop Type' because it's a categorical string
crop_encoder = LabelEncoder()
X_copy = X.copy()
X_copy['Crop Type'] = crop_encoder.fit_transform(X['Crop Type'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_copy, y, test_size=0.2, random_state=42)

print("3. Training Random Forest Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Data: {acc * 100:.2f}%")

print("4. Saving Model and Encoders...")
# Save the model
pickle.dump(model, open('models/fertilizer_model.pkl', 'wb'))

# Save the label encoder so we can transform user input (like 'rice') into the same numbers used in training
pickle.dump(crop_encoder, open('models/crop_encoder.pkl', 'wb'))

print("Completed successfully! Models saved in 'models/' directory.")
