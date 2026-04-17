import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from PIL import Image

print("1. Preparing Synthetic Image Dataset for Plant Disease CNN...")

# We'll define 4 classes
classes = ["Apple Scab", "Corn Common Rust", "Potato Early Blight", "Healthy Crop"]
num_classes = len(classes)
img_size = 64

# Generate 400 random "images" (tensors)
n_samples = 400
X = []
y = []

# Generate synthetic arrays simulating image pixels (0-255)
np.random.seed(42)
for i in range(n_samples):
    class_idx = i % num_classes
    # Base color variation per class to give CNN something to learn
    base_color = [np.random.randint(50, 150), np.random.randint(100, 200), np.random.randint(50, 100)]
    noise = np.random.randint(0, 50, (img_size, img_size, 3), dtype=np.uint8)
    img_array = np.clip(base_color + noise, 0, 255).astype(np.uint8)
    
    X.append(img_array)
    y.append(class_idx)

X = np.array(X)
y = np.array(y)

# Normalize pixels (Standard Deep Learning Practice)
X = X / 255.0

# Mix and split
indices = np.arange(n_samples)
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

y_cat = to_categorical(y, num_classes=num_classes)

split = int(n_samples * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y_cat[:split], y_cat[split:]

print("2. Building Convolutional Neural Network (CNN)...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

print("3. Training the Deep Learning Model...")
# A few epochs are enough for synthetic colored noise
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), verbose=1)

# Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Model Accuracy on Validation Data: {acc * 100:.2f}%")

print("4. Saving the H5 Model & Classes...")
if not os.path.exists("models"):
    os.makedirs("models")

# Save Keras Model
model.save("models/disease_keras_model.keras")

# Save corresponding class tags
pickle.dump(classes, open('models/disease_classes.pkl', 'wb'))

print("Completed successfully! Model saved in 'models/disease_keras_model.keras'.")
