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
        return "Model Error", "Disease model not loaded on server."
        
    try:
        # 1. Get model's expected input shape
        input_details = models['disease_input_details'][0]
        input_shape = input_details['shape']  # e.g., [1, 224, 224, 3]
        target_h, target_w = input_shape[1], input_shape[2]

        # 2. Preprocess image
        img = Image.open(image_file).convert('RGB').resize((target_w, target_h))
        img_array = np.array(img)
        
        # 3. Convert RGB to BGR (Many ML models trained on BGR)
        img_array = img_array[:, :, ::-1]
        
        # 4. Dynamic Normalization based on model's dtype
        dtype = input_details['dtype']
        if dtype == np.float32:
            # Try 0-1 normalization first (standard)
            img_array = img_array.astype(np.float32) / 255.0

        elif dtype == np.uint8:
            # Quantized Uint8 models: 0 to 255
            img_array = img_array.astype(np.uint8)
        elif dtype == np.int8:
            # Quantized Int8 models: -128 to 127
            img_array = (img_array.astype(np.int32) - 128).astype(np.int8)
            
        img_array = np.expand_dims(img_array, axis=0)
        
        # 4. Inference
        interpreter = models['disease_interpreter']
        interpreter.set_tensor(input_details['index'], img_array)
        interpreter.invoke()
        
        # 5. Process Results
        output_details = models['disease_output_details'][0]
        preds = interpreter.get_tensor(output_details['index'])
        
        # Dequantize if needed
        if output_details.get('quantization'):
            scale, zero_point = output_details['quantization']
            if scale > 0:
                preds = scale * (preds.astype(np.float32) - zero_point)

        # DEEP DIAGNOSTIC LOGGING
        print(f"--- DISEASE DIAGNOSTIC ---")
        print(f"Input Shape: {input_shape}")
        print(f"Output Raw: {preds[0]}")
        
        class_idx = np.argmax(preds[0])
        confidence = float(preds[0][class_idx])
        
        classes = models['disease_classes']
        print(f"Available Classes: {classes}")
        print(f"Predicted Index: {class_idx} (Confidence: {confidence})")
        
        if classes is None or not isinstance(classes, (list, np.ndarray)):
            return "Configuration Error", "Disease classes not loaded correctly."
            
        if class_idx >= len(classes):
            return "Index Error", f"Model predicted class {class_idx} but only {len(classes)} names found."
            
        result = classes[class_idx]
        print(f"Final Result: {result}")
        print(f"--------------------------")
        
        tips = {
            "Apple Scab": "Use fungicide and prune branches.",
            "Corn Common Rust": "Improve air circulation.",
            "Potato Early Blight": "Remove infected leaves.",
            "Healthy Crop": "Maintain current care."
        }
        
        if confidence < 0.35:
            msg = f"Low confidence ({confidence*100:.1f}%). Model is unsure. ({result})"
        else:
            msg = f"{tips.get(result, 'Monitor closely.')} (Confidence: {confidence*100:.1f}%)"
            
        return result, msg


        
    except Exception as e:
        print(f"Disease Prediction Error: {e}")
        return "Error", f"Inference failed: {str(e)}"

