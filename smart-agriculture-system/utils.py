import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image

# Setup Model Paths for student project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Yeh dictionaries hamare models ko store karengi
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
    """Sare machine learning models ko load karta hai server start hone par."""
    print("Models load ho rahe hain...")
    try:
        def load_pkl(filename):
            path = os.path.join(MODELS_DIR, filename)
            if not os.path.exists(path):
                print(f"File nahi mili: {filename}")
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
        
        # 5. Disease Model (Image Processing)
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
                print("Warning: disease_model.tflite nahi mili.")
                
        print("Sare models successfully load ho gaye!")
    except Exception as e:
        print(f"Error aagaya model loading me: {e}")

def validate_numeric(*args):
    """Check karta hai ki user ne numbers hi daale hain na."""
    try:
        nums = [float(arg) for arg in args]
        if any(n < 0 for n in nums):
            return False, "Negative values allowed nahi hain."
        return True, nums
    except (ValueError, TypeError):
        return False, "Kripya sirf numbers daalein."


# --- CORE LOGIC FUNCTIONS ---

def recommend_fertilizer(n, p, k, crop):
    valid, values = validate_numeric(n, p, k)
    if not valid:
        return "Validation Error", values

    if models['fertilizer'] is None:
        return "Model Error", "Model load nahi hua."
    
    try:
        # User input ko encode karo ML ke format me
        encoded_crop = models['crop_encoder'].transform([crop])[0]
        prediction = models['fertilizer'].predict([[n, p, k, encoded_crop]])
        return prediction[0], "Ye fertilizer aapke crop ke liye best rahega."
    except Exception as e:
        return "Error", str(e)


def predict_yield(crop, area, rain, temp):
    valid, values = validate_numeric(area, rain, temp)
    if not valid:
        return "Validation Error", values

    if models['yield'] is None:
        return "Model Error", "Yield model load nahi hua."
        
    try:
        encoded_crop = models['yield_crop_encoder'].transform([crop])[0]
        prediction = models['yield'].predict([[encoded_crop, float(area), float(rain), float(temp)]])
        return round(prediction[0], 2), f"Estimated yield for {area} hectares."
    except Exception as e:
        return "Error", str(e)


def predict_soil(n, p, k, ph):
    valid, values = validate_numeric(n, p, k, ph)
    if not valid:
        return "Validation Error", values
    
    if float(ph) < 0 or float(ph) > 14:
        return "Error", "pH level 0 se 14 ke beech hona chahiye."

    if models['soil'] is None:
        return "Model Error", "Soil model load nahi hua."
        
    try:
        prediction = models['soil'].predict([[float(n), float(p), float(k), float(ph)]])
        predicted_class = prediction[0]
        
        messages = {
            "Fertile": "Aapki mitti bahut achi (Fertile) condition me hai.",
            "Medium": "Mitti theek hai, thoda khad (fertilizer) daalna padega.",
            "Poor": "Mitti ki condition kharab hai, treatment ki zaroorat hai."
        }
        return predicted_class, messages.get(predicted_class, "Check nutrients.")
    except Exception as e:
        return "Error", str(e)


def predict_weather_crop(temp, humid, rain):
    valid, values = validate_numeric(temp, humid, rain)
    if not valid:
        return "Validation Error", values

    if models['weather'] is None:
        return "Model Error", "Weather model load nahi hua."
        
    try:
        prediction = models['weather'].predict([[float(temp), float(humid), float(rain)]])
        return prediction[0], "Mausam ke hisab se ye fasal sabse best rahegi."
    except Exception as e:
        return "Error", str(e)


def predict_disease(image_file):
    if models['disease_interpreter'] is None:
        return "Model Error", "Disease model load nahi hua."
        
    try:
        # Dynamic sizing for student project fix
        input_details = models['disease_input_details'][0]
        req_height, req_width = input_details['shape'][1], input_details['shape'][2]

        # Image Process karna
        img = Image.open(image_file).convert('RGB').resize((req_width, req_height))
        img_array = np.array(img, dtype=np.float32)
        
        # OpenCV wala BGR fix
        img_array = img_array[..., ::-1]
        
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Prediction
        interpreter = models['disease_interpreter']
        input_idx = input_details['index']
        output_idx = models['disease_output_details'][0]['index']
        
        interpreter.set_tensor(input_idx, img_array)
        interpreter.invoke()
        
        # Results nikalna
        preds = interpreter.get_tensor(output_idx)
        class_idx = np.argmax(preds[0])
        confidence = float(preds[0][class_idx])
        
        classes = models['disease_classes']
        result = classes[class_idx]
        
        tips = {
            "Apple Scab": "Fungicide ka spray karein.",
            "Corn Common Rust": "Fasal me hawa lagne ki jagah banayein.",
            "Potato Early Blight": "Kharab patton ko turant kaat de.",
            "Healthy Crop": "Sab kuch theek hai, aise hi dhyan rakhein."
        }
        
        if confidence < 0.35:
            msg = f"Sure nahi hu ({confidence*100:.1f}% confidence). Shayad ye hai: {result}"
        else:
            msg = f"{tips.get(result, 'Kripya dhyan rakhein.')} (Confidence: {confidence*100:.1f}%)"
            
        return result, msg
        
    except Exception as e:
        print(f"Image scan karte waqt error: {e}")
        return "Error", "Scanning fail ho gayi."
