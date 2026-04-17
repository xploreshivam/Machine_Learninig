import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error

print("1. Generating Realistic Synthetic Dataset for Crop Yield...")
# We use a synthetic dataset based on agricultural facts to ensure perfect matching 
# with the frontend fields (Crop, Area, Rainfall, Temperature).
np.random.seed(42)
n_samples = 5000

crops = ['Wheat', 'Rice', 'Maize', 'Sugarcane', 'Cotton', 'Soybean']

# Base stats (optimal temp, optimal rain, base yield per hectare in tonnes)
crop_stats = {
    'Wheat': {'opt_temp': 20, 'opt_rain': 500, 'base_yield': 3.5},
    'Rice': {'opt_temp': 25, 'opt_rain': 1200, 'base_yield': 4.5},
    'Maize': {'opt_temp': 22, 'opt_rain': 800, 'base_yield': 5.0},
    'Sugarcane': {'opt_temp': 30, 'opt_rain': 1500, 'base_yield': 60.0},
    'Cotton': {'opt_temp': 28, 'opt_rain': 600, 'base_yield': 1.5},
    'Soybean': {'opt_temp': 26, 'opt_rain': 700, 'base_yield': 2.5}
}

data = []
for _ in range(n_samples):
    crop = np.random.choice(crops)
    stats = crop_stats[crop]
    
    # Generate random features around optimal values
    area = np.random.uniform(1.0, 50.0) # 1 to 50 hectares
    temp = np.random.normal(stats['opt_temp'], 5.0) 
    rain = np.random.normal(stats['opt_rain'], 200.0)
    
    # Calculate yield with penalties for deviating from optimal weather
    temp_penalty = abs(temp - stats['opt_temp']) / stats['opt_temp']
    rain_penalty = abs(rain - stats['opt_rain']) / stats['opt_rain']
    
    # Yield per hectare
    yield_per_hectare = stats['base_yield'] * (1 - 0.2 * temp_penalty - 0.1 * rain_penalty)
    
    # Add some random noise
    yield_per_hectare += np.random.normal(0, stats['base_yield'] * 0.1)
    
    # Ensure yield is not negative
    yield_per_hectare = max(0.1, yield_per_hectare)
    
    # Total production (yield) in tonnes
    total_yield = yield_per_hectare * area
    
    data.append([crop, area, rain, temp, total_yield])

df = pd.DataFrame(data, columns=['Crop', 'Area', 'Rainfall', 'Temperature', 'Yield'])

if not os.path.exists("datasets"):
    os.makedirs("datasets")
df.to_csv('datasets/crop_yield.csv', index=False)
print("Dataset generated and saved to datasets/crop_yield.csv")

print("2. Preprocessing Data...")
X = df[['Crop', 'Area', 'Rainfall', 'Temperature']]
y = df['Yield']

# Encode the Crop feature
le_crop = LabelEncoder()
X_encoded = X.copy()
X_encoded['Crop'] = le_crop.fit_transform(X['Crop'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

print("3. Training Random Forest Regressor Model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Model R2 Score: {r2:.4f} (Closer to 1 is better)")
print(f"Model Mean Absolute Error: {mae:.2f} Tonnes")

print("4. Saving Model and Encoders...")
if not os.path.exists("models"):
    os.makedirs("models")
pickle.dump(model, open('models/yield_model.pkl', 'wb'))
pickle.dump(le_crop, open('models/yield_crop_encoder.pkl', 'wb'))

print("Completed successfully! Models saved in 'models/' directory.")
