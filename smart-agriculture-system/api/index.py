import os
from flask import Flask, render_template, request
try:
    from . import utils
except ImportError:
    import utils

# Environment setup for TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Import TFLite (handle both standard and runtime versions)
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        tflite = None

# Initialize Flask app (paths relative to api/index.py)
app = Flask(__name__, 
            template_folder='../templates', 
            static_folder='../static')

# Load models on startup
utils.load_models(tflite)

@app.route('/')
def home():
    return render_template('index.html')

# --- 1. Fertilizer Module ---
@app.route('/fertilizer')
def fertilizer():
    return render_template('fertilizer.html', result=None)

@app.route('/predict-fertilizer', methods=['POST'])
def predict_fertilizer():
    try:
        n, p, k = int(request.form['nitrogen']), int(request.form['phosphorus']), int(request.form['potassium'])
        crop = request.form['crop']
        res, msg = utils.recommend_fertilizer(n, p, k, crop)
        return render_template('fertilizer.html', result=res, message=msg)
    except Exception as e:
        return render_template('fertilizer.html', result="Error", message=str(e))

# --- 2. Crop Yield Module ---
@app.route('/crop-yield')
def crop_yield():
    return render_template('crop_yield.html', result=None)

@app.route('/predict-yield', methods=['POST'])
def predict_yield():
    try:
        crop = request.form['crop']
        a, r, t = request.form['area'], request.form['rainfall'], request.form['temperature']
        res, msg = utils.predict_yield(crop, a, r, t)
        return render_template('crop_yield.html', result=res, message=msg)
    except Exception as e:
        return render_template('crop_yield.html', result="Error", message=str(e))

# --- 3. Soil Quality Module ---
@app.route('/soil-quality')
def soil_quality():
    return render_template('soil_quality.html', result=None)

@app.route('/predict-soil', methods=['POST'])
def predict_soil():
    try:
        n, p, k, ph = request.form['nitrogen'], request.form['phosphorus'], request.form['potassium'], request.form['ph']
        res, msg = utils.predict_soil(n, p, k, ph)
        return render_template('soil_quality.html', result=res, message=msg)
    except Exception as e:
        return render_template('soil_quality.html', result="Error", message=str(e))

# --- 4. Disease Detection Module ---
@app.route('/disease')
def disease():
    return render_template('disease.html', result=None)

@app.route('/predict-disease', methods=['POST'])
def predict_disease():
    try:
        if 'image' not in request.files or request.files['image'].filename == '':
            return render_template('disease.html', result="Error", message="No image selected")
        
        res, msg = utils.predict_disease(request.files['image'])
        return render_template('disease.html', result=res, message=msg)
    except Exception as e:
        return render_template('disease.html', result="Error", message=str(e))

# --- 5. Weather Crop Module ---
@app.route('/weather')
def weather():
    return render_template('weather.html', result=None)

@app.route('/predict-weather-crop', methods=['POST'])
def predict_weather_crop():
    try:
        t, h, r = request.form['temperature'], request.form['humidity'], request.form['rainfall']
        res, msg = utils.predict_weather_crop(t, h, r)
        return render_template('weather.html', result=res, message=msg)
    except Exception as e:
        return render_template('weather.html', result="Error", message=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
