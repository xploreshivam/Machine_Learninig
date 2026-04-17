import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("1. Generating Synthetic Dataset for Weather-Based Crop Suggestion...")
np.random.seed(42)
n_samples = 4000

# Features: Temperature, Humidity, Rainfall
# Target: Crop Name

crops_profile = {
    'Rice': {'temp': 25, 'humidity': 80, 'rain': 200},
    'Wheat': {'temp': 15, 'humidity': 50, 'rain': 50},
    'Maize': {'temp': 25, 'humidity': 60, 'rain': 80},
    'Cotton': {'temp': 30, 'humidity': 70, 'rain': 100},
    'Jute': {'temp': 30, 'humidity': 85, 'rain': 180},
    'Apple': {'temp': 10, 'humidity': 60, 'rain': 100},
    'Orange': {'temp': 20, 'humidity': 70, 'rain': 120},
    'Coconut': {'temp': 28, 'humidity': 80, 'rain': 220}
}

data = []
for _ in range(n_samples):
    crop = np.random.choice(list(crops_profile.keys()))
    prof = crops_profile[crop]
    
    # Add random noise
    temp = np.random.normal(prof['temp'], 3)
    humidity = np.random.normal(prof['humidity'], 8)
    rain = np.random.normal(prof['rain'], 25)
    
    # Ensure no negative values
    temp = max(5, temp)
    humidity = max(10, min(100, humidity))
    rain = max(10, rain)
    
    data.append([temp, humidity, rain, crop])

df = pd.DataFrame(data, columns=['Temperature', 'Humidity', 'Rainfall', 'Crop'])

if not os.path.exists("datasets"):
    os.makedirs("datasets")
df.to_csv('datasets/weather_crop.csv', index=False)
print("Dataset generated and saved to datasets/weather_crop.csv")

print("2. Preprocessing Data...")
X = df[['Temperature', 'Humidity', 'Rainfall']]
y = df['Crop']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("3. Training Random Forest Classifier Model...")
model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Data: {acc * 100:.2f}%")

print("4. Saving Model...")
if not os.path.exists("models"):
    os.makedirs("models")
pickle.dump(model, open('models/weather_crop_model.pkl', 'wb'))

print("Completed successfully! Model saved in 'models/weather_crop_model.pkl'.")
