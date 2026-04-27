# 🌾 Smart Agriculture Analytic System

The **Smart Agriculture Analytic System** is a comprehensive, machine learning-powered web application built to modernize farming practices. By analyzing various environmental, chemical, and visual data inputs, this system empowers farmers and agricultural experts to make highly accurate, data-driven decisions that maximize yield and prevent crop loss.

---

## 🎯 Project Objective
Agriculture is the backbone of the economy, but unpredictable weather, soil degradation, and plant diseases often lead to massive losses. This project aims to bridge the gap between traditional farming and modern AI technology by providing an all-in-one predictive dashboard that acts as a digital assistant for farmers.

---

## 🚀 Detailed Features & Modules

### 1. 🧪 Fertilizer Recommendation System
* **Input:** Soil nutrient levels (Nitrogen, Phosphorus, Potassium) and the specific crop you wish to plant.
* **Process:** Uses a classification model to understand nutrient deficiencies in the soil context.
* **Output:** Recommends the exact type of fertilizer required to balance the soil for that specific crop.

### 2. 📈 Crop Yield Prediction
* **Input:** Total farm area (hectares), average rainfall (mm), and temperature (°C).
* **Process:** Uses regression ML techniques trained on historical agricultural data.
* **Output:** Estimates the exact crop yield (in tons) so farmers can plan their finances, logistics, and storage in advance.

### 3. 🌍 Soil Quality Prediction
* **Input:** Nitrogen, Phosphorus, Potassium, and pH levels.
* **Process:** Analyzes the chemical balance and acidity of the soil.
* **Output:** Categorizes soil health (e.g., Fertile, Medium, Poor) and provides brief actionable advice to improve soil longevity.

### 4. 🔍 Plant Disease Detection (Computer Vision)
* **Input:** An image of a plant leaf.
* **Process:** Uses a highly optimized TensorFlow Lite (TFLite) Convolutional Neural Network (CNN) to scan the leaf for visual symptoms.
* **Output:** Identifies the disease (e.g., Apple Scab, Potato Early Blight) and provides confidence percentages along with instant treatment tips (like using fungicides or improving air circulation).

### 5. ⛅ Weather-based Crop Suggestion
* **Input:** Current temperature, humidity, and rainfall.
* **Process:** Matches climate conditions with crop survival parameters.
* **Output:** Recommends the most suitable crop to plant in the current weather to ensure maximum survivability.

---

## 🛠️ Technology Stack & Architecture

* **Core Backend Framework:** Python 3.11, Flask
* **Machine Learning & AI:** 
  * `Scikit-Learn` (for classification and regression models)
  * `TensorFlow Lite` (for lightweight, fast image processing)
  * `Pandas` & `NumPy` (for data manipulation and array processing)
  * `Pillow` (for image preprocessing)
* **Frontend Interface:** HTML5, Vanilla CSS, Jinja2 Templating
* **Deployment Ready:** Gunicorn and Vercel compatibility configured

---

## 📂 Project Structure
```text
smart-agriculture-system/
├── api/
│   ├── index.py       # Main Flask application routing & entry point
│   └── utils.py       # ML logic, data validation, and inference functions
├── models/            # Pre-trained ML models (.pkl and .tflite)
├── static/            # CSS styles and application images
├── templates/         # HTML user interface files
├── run_server.bat     # Automated Windows startup script
└── requirements.txt   # Python dependency list
```

---

## ⚙️ How to Setup & Run the Project

### Prerequisites
Make sure you have **Python 3.11** installed on your system.

### The Easiest Way to Run (Windows)
We have created an automated batch script that handles environment activation and server startup.

1. **Open File Explorer** and navigate to your project folder.
2. **Run the Script:** Find and double-click the `run_server.bat` file. 
   *(Note: Do not click "Run" in VS Code, physically go to the folder or use the terminal).*
3. **Access the Application:** Once the black console window says `Running on http://127.0.0.1:5001`, open your web browser and navigate to:
   👉 **[http://127.0.0.1:5001](http://127.0.0.1:5001)**

### Manual Setup (Command Line)
If you prefer to run the commands yourself, open your terminal (PowerShell or CMD) in the project directory and run:

```powershell
# 1. Activate the virtual environment
.\venv\Scripts\activate

# 2. Start the Flask server
python api/index.py
```

---

## 👥 Project Team

This project was developed collectively as a group effort for academic evaluation.

**🏆 Group Leader:**
* **Shivam Chauhan** - Project Management & Core Architecture

**👨‍💻 Group Members (9):**
* **Sanjay Kumar** 
* **Shubham Singh** 
* **Ginesh Singh** 
* **Haseen Babu** 
* **Mohit Babu** 
* **Shivam Sharma** 
* **Pushker** 
* **Shashank Singh** 
* **Saleem Khan** 

---
*Developed in 4th Semester ML Curriculum.*
