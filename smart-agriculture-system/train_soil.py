import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("1. Generating Synthetic Dataset for Soil Quality Classification...")
np.random.seed(42)
n_samples = 4000

# We need features: Nitrogen, Phosphorus, Potassium, pH
# Target: Quality ("Fertile", "Medium", "Poor")

data = []
for _ in range(n_samples):
    n = np.random.uniform(0, 150)
    p = np.random.uniform(0, 150)
    k = np.random.uniform(0, 150)
    ph = np.random.uniform(3.0, 10.0)
    
    # Calculate a base score
    score = (n + p + k) / 3.0
    
    # Determine the strict logic, but we'll add some noise so the model actually learns boundaries
    if ph < 5.0 or ph > 8.5:
        # Extreme pH usually means poor soil availability regardless of nutrients
        quality = "Poor"
    else:
        if score > 80 and 6.0 <= ph <= 7.5:
            quality = "Fertile"
        elif score > 40:
            quality = "Medium"
        else:
            quality = "Poor"
            
    # Occasionally insert some noise to make it realistic (approx 5% chance)
    if np.random.rand() < 0.05:
        quality = np.random.choice(["Fertile", "Medium", "Poor"])
        
    data.append([n, p, k, ph, quality])

df = pd.DataFrame(data, columns=['Nitrogen', 'Phosphorus', 'Potassium', 'ph', 'Quality'])

if not os.path.exists("datasets"):
    os.makedirs("datasets")
df.to_csv('datasets/soil_quality.csv', index=False)
print("Dataset generated and saved to datasets/soil_quality.csv")

print("2. Preprocessing Data...")
X = df[['Nitrogen', 'Phosphorus', 'Potassium', 'ph']]
y = df['Quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("3. Training Random Forest Classifier Model...")
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Data: {acc * 100:.2f}%")

print("4. Saving Model...")
if not os.path.exists("models"):
    os.makedirs("models")
pickle.dump(model, open('models/soil_model.pkl', 'wb'))

print("Completed successfully! Model saved in 'models/soil_model.pkl'.")
