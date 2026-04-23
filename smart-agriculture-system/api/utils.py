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
    print("Starting model loading process...")
    try:
        # Helper to load pickle files
        def load_pkl(filename):
            path = os.path.join(MODELS_DIR, filename)
            if not os.path.exists(path):
                print(f"Error: {filename} not found at {path}")
                return None
            with open(path, 'rb') as f:
                return pickle.load(f)

        # 1. Fertilizer Models
        models['fertilizer'] = load_pkl('fertilizer_model.pkl')
        models['crop_encoder'] = load_pkl('crop_encoder.pkl')
        
        # 2. Yield Models
        models['yield'] = load_pkl('yield_model.pkl')
        models['yield_crop_encoder'] = load_pkl('yield_crop_encoder.pkl')
        
        # 3. Soil Model
        models['soil'] = load_pkl('soil_model.pkl')
        
        # 4. Weather Model
        models['weather'] = load_pkl('weather_crop_model.pkl')
        
        # 5. Disease Model (TFLite)
        if tflite_module:
            tflite_path = os.path.join(MODELS_DIR, 'disease_model.tflite')
            if os.path.exists(tflite_path):
                interpreter = tflite_module.Interpreter(model_path=tflite_path)
                interpreter.allocate_tensors()
                models['disease_interpreter'] = interpreter
                models['disease_input_details'] = interpreter.get_input_details()
                models['disease_output_details'] = interpreter.get_output_details()
                models['disease_classes'] = load_pkl('disease_classes.pkl')
            else:
                print("Warning: disease_model.tflite not found.")
        else:
            print("Warning: TFLite module not provided. Disease detection disabled.")
            
        print("Model loading check complete.")
    except Exception as e:
        print(f"Critical Error during model loading: {e}")

def validate_numeric(*args):
    """Ensure all arguments can be converted to float and are non-negative."""
    try:
        nums = [float(arg) for arg in args]
        if any(n < 0 for n in nums):
            return False, "Values cannot be negative."
        return True, nums
    except (ValueError, TypeError):
        return False, "Please enter valid numeric values."


# --- LOGIC FUNCTIONS ---

def recommend_fertilizer(n, p, k, crop):
    # Validation
    valid, values = validate_numeric(n, p, k)
    if not valid:
        return "Validation Error", values

    if models['fertilizer'] is None or models['crop_encoder'] is None:
        return "Model Error", "Fertilizer model not loaded on server."
    
    if crop not in models['crop_encoder'].classes_:
        return "Unknown Crop", f"Crop '{crop}' not supported by this model."
        
    try:
        encoded_crop = models['crop_encoder'].transform([crop])[0]
        prediction = models['fertilizer'].predict([[n, p, k, encoded_crop]])
        return prediction[0], "Recommendation based on chemical balance."
    except Exception as e:
        return "Prediction Error", str(e)


def predict_yield(crop, area, rain, temp):
    # Validation
    valid, values = validate_numeric(area, rain, temp)
    if not valid:
        return "Validation Error", values

    if models['yield'] is None or models['yield_crop_encoder'] is None:
        return "Model Error", "Yield model not loaded on server."
        
    try:
        if crop not in models['yield_crop_encoder'].classes_:
            return "Unknown Crop", f"Crop '{crop}' not supported."
            
        encoded_crop = models['yield_crop_encoder'].transform([crop])[0]
        prediction = models['yield'].predict([[encoded_crop, float(area), float(rain), float(temp)]])
        
        return round(prediction[0], 2), f"Estimated yield for {area} hectares."
    except Exception as e:
        return "Prediction Error", str(e)


def predict_soil(n, p, k, ph):
    # Validation
    valid, values = validate_numeric(n, p, k, ph)
    if not valid:
        return "Validation Error", values
    
    # Specific ph check
    if float(ph) < 0 or float(ph) > 14:
        return "Validation Error", "pH level must be between 0 and 14."

    if models['soil'] is None:
        return "Model Error", "Soil model not loaded on server."
        
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
        return "Prediction Error", str(e)


def predict_weather_crop(temp, humid, rain):
    # Validation
    valid, values = validate_numeric(temp, humid, rain)
    if not valid:
        return "Validation Error", values

    if models['weather'] is None:
        return "Model Error", "Weather model not loaded on server."
        
    try:
        prediction = models['weather'].predict([[float(temp), float(humid), float(rain)]])
        return prediction[0], f"AI recommendation for given climate."
    except Exception as e:
        return "Prediction Error", str(e)


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
