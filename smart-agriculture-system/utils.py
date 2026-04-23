import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image

# Setup Model Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')

# Global variables for models
models = {
    'fertilizer': None,
    'crop_encoder': None,
    'yield': None,
    'yield_crop_encoder': None,
    'soil': None,
    'weather': None,
    'disease_interpreter': None,
    'disease_input_details': None,
    'disease_output_details': None,
    'disease_classes': None
}

def load_models(tflite_module):
    """Load all machine learning models from the models directory."""
    try:
        # 1. Fertilizer Models
        models['fertilizer'] = pickle.load(open(os.path.join(MODELS_DIR, 'fertilizer_model.pkl'), 'rb'))
        models['crop_encoder'] = pickle.load(open(os.path.join(MODELS_DIR, 'crop_encoder.pkl'), 'rb'))
        
        # 2. Yield Models
        models['yield'] = pickle.load(open(os.path.join(MODELS_DIR, 'yield_model.pkl'), 'rb'))
        models['yield_crop_encoder'] = pickle.load(open(os.path.join(MODELS_DIR, 'yield_crop_encoder.pkl'), 'rb'))
        
        # 3. Soil Model
        models['soil'] = pickle.load(open(os.path.join(MODELS_DIR, 'soil_model.pkl'), 'rb'))
        
        # 4. Weather Model
        models['weather'] = pickle.load(open(os.path.join(MODELS_DIR, 'weather_crop_model.pkl'), 'rb'))
        
        # 5. Disease Model (TFLite)
        if tflite_module:
            interpreter = tflite_module.Interpreter(model_path=os.path.join(MODELS_DIR, 'disease_model.tflite'))
            interpreter.allocate_tensors()
            models['disease_interpreter'] = interpreter
            models['disease_input_details'] = interpreter.get_input_details()
            models['disease_output_details'] = interpreter.get_output_details()
            models['disease_classes'] = pickle.load(open(os.path.join(MODELS_DIR, 'disease_classes.pkl'), 'rb'))
            
        print("All models loaded successfully!")
    except Exception as e:
        print(f"Warning: Model loading failed. {e}")

# --- LOGIC FUNCTIONS ---

def recommend_fertilizer(n, p, k, crop):
    if models['fertilizer'] is None or models['crop_encoder'] is None:
        return "Model Error", "Fertilizer model not loaded."
    
    if crop not in models['crop_encoder'].classes_:
        return "Unknown Crop", f"Crop '{crop}' not supported."
        
    encoded_crop = models['crop_encoder'].transform([crop])[0]
    prediction = models['fertilizer'].predict([[n, p, k, encoded_crop]])
    
    return prediction[0], "Based on N-P-K levels and crop type."

def predict_yield(crop, area, rain, temp):
    if models['yield'] is None or models['yield_crop_encoder'] is None:
        return "Model Error", "Yield model not loaded."
        
    try:
        if crop not in models['yield_crop_encoder'].classes_:
            return "Unknown Crop", f"Crop '{crop}' not supported."
            
        encoded_crop = models['yield_crop_encoder'].transform([crop])[0]
        prediction = models['yield'].predict([[encoded_crop, float(area), float(rain), float(temp)]])
        
        return round(prediction[0], 2), f"Estimated yield for {area} hectares."
    except Exception as e:
        return "Error", str(e)

def predict_soil(n, p, k, ph):
    if models['soil'] is None:
        return "Model Error", "Soil model not loaded."
        
    try:
        prediction = models['soil'].predict([[float(n), float(p), float(k), float(ph)]])
        predicted_class = prediction[0]
        
        messages = {
            "Fertile": "Soil is in excellent condition.",
            "Medium": "Moderate health. Minor fertilization suggested.",
            "Poor": "Soil needs significant treatment."
        }
        return predicted_class, messages.get(predicted_class, "Check nutrients.")
    except Exception as e:
        return "Error", str(e)

def predict_weather_crop(temp, humid, rain):
    if models['weather'] is None:
        return "Model Error", "Weather model not loaded."
        
    try:
        prediction = models['weather'].predict([[float(temp), float(humid), float(rain)]])
        return prediction[0], f"AI recommendation for given climate."
    except Exception as e:
        return "Error", str(e)

def predict_disease(image_file):
    if models['disease_interpreter'] is None:
        return "Model Error", "Disease model not loaded."
        
    try:
        # Preprocess image
        img = Image.open(image_file).convert('RGB').resize((64, 64))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
        
        # Inference
        interpreter = models['disease_interpreter']
        interpreter.set_tensor(models['disease_input_details'][0]['index'], img_array)
        interpreter.invoke()
        
        preds = interpreter.get_tensor(models['disease_output_details'][0]['index'])
        class_idx = np.argmax(preds[0])
        confidence = float(np.max(preds[0]))
        
        result = models['disease_classes'][class_idx]
        
        tips = {
            "Apple Scab": "Use fungicide and prune branches.",
            "Corn Common Rust": "Improve air circulation.",
            "Potato Early Blight": "Remove infected leaves.",
            "Healthy Crop": "Maintain current care."
        }
        
        return result, f"{tips.get(result, 'Monitor closely.')} (Confidence: {confidence*100:.1f}%)"
    except Exception as e:
        return "Error", f"Image processing failed: {e}"
