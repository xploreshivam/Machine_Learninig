import os
from flask import Flask, render_template, request

# Simple TFLite Import for student project
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        tflite = None

import utils  # Import our machine learning logic from utils.py

# Initialize our Flask app
app = Flask(__name__)

# Load all machine learning models when the server starts
utils.load_models(tflite)

# --- Main Home Page Route ---
@app.route('/')
def home():
    # Ye hamara main page hai
    return render_template('index.html')


# --- 1. Fertilizer Recommendation Module ---
@app.route('/fertilizer')
def fertilizer():
    return render_template('fertilizer.html', result=None)

@app.route('/predict-fertilizer', methods=['POST'])
def predict_fertilizer():
    try:
        # Get data from HTML form
        n = request.form['nitrogen']
        p = request.form['phosphorus']
        k = request.form['potassium']
        crop = request.form['crop']
        
        # Send to ML model
        res, msg = utils.recommend_fertilizer(n, p, k, crop)
        return render_template('fertilizer.html', result=res, message=msg, input_data=request.form)
    except Exception as e:
        return render_template('fertilizer.html', result="Error", message="Kuch gadbad hui. Please input check karein.", input_data=request.form)


# --- 2. Crop Yield Prediction Module ---
@app.route('/crop-yield')
def crop_yield():
    return render_template('crop_yield.html', result=None)

@app.route('/predict-yield', methods=['POST'])
def predict_yield():
    try:
        crop = request.form['crop']
        area = request.form['area']
        rainfall = request.form['rainfall']
        temperature = request.form['temperature']
        
        res, msg = utils.predict_yield(crop, area, rainfall, temperature)
        return render_template('crop_yield.html', result=res, message=msg, input_data=request.form)
    except Exception as e:
        return render_template('crop_yield.html', result="Error", message="Prediction fail ho gayi.", input_data=request.form)


# --- 3. Soil Quality Analysis Module ---
@app.route('/soil-quality')
def soil_quality():
    return render_template('soil_quality.html', result=None)

@app.route('/predict-soil', methods=['POST'])
def predict_soil():
    try:
        n = request.form['nitrogen']
        p = request.form['phosphorus']
        k = request.form['potassium']
        ph = request.form['ph']
        
        res, msg = utils.predict_soil(n, p, k, ph)
        return render_template('soil_quality.html', result=res, message=msg, input_data=request.form)
    except Exception as e:
        return render_template('soil_quality.html', result="Error", message="Soil analysis fail ho gaya.", input_data=request.form)


# --- 4. Plant Disease Detection Module ---
@app.route('/disease')
def disease():
    return render_template('disease.html', result=None)

@app.route('/predict-disease', methods=['POST'])
def predict_disease():
    try:
        # Check if user uploaded a file
        if 'image' not in request.files:
            return render_template('disease.html', result=None, message="Koi photo upload nahi ki gayi.")
            
        file = request.files['image']
        if file.filename == '':
            return render_template('disease.html', result=None, message="File select nahi hui.")
        
        # Send image to ML model
        res, msg = utils.predict_disease(file)
        return render_template('disease.html', result=res, message=msg)
    except Exception as e:
        print(f"Error in disease route: {e}")
        return render_template('disease.html', result="Error", message="Photo scan karte waqt error aaya.")


# --- 5. Weather Based Crop Suggestion ---
@app.route('/weather')
def weather():
    return render_template('weather.html', result=None)

@app.route('/predict-weather-crop', methods=['POST'])
def predict_weather_crop():
    try:
        temp = request.form['temperature']
        humidity = request.form['humidity']
        rainfall = request.form['rainfall']
        
        res, msg = utils.predict_weather_crop(temp, humidity, rainfall)
        return render_template('weather.html', result=res, message=msg, input_data=request.form)
    except Exception as e:
        return render_template('weather.html', result="Error", message="Suggestion process fail ho gaya.", input_data=request.form)


# Start the application
if __name__ == '__main__':
    # Localhost par 5001 port me run hoga
    app.run(host='127.0.0.1', port=5001, debug=True)
