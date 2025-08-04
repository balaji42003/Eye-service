import os
import logging
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)
CORS(app)  # Enable CORS for React Native frontend

# Global variables for lazy loading
model = None
MODEL_PATH = "models/eye_disease_model.h5"
CATEGORIES = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"]

def load_model_lazy():
    """Lazy load the model when first needed"""
    global model
    if model is None:
        try:
            logging.info(f"Loading model from {MODEL_PATH}...")
            model = load_model(MODEL_PATH)
            logging.info("Model loaded successfully!")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise e
    return model

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "Eye Disease Detection API",
        "status": "Active",
        "message": "API is running - use /predict endpoint for predictions",
        "supported_conditions": CATEGORIES,
        "endpoints": {
            "/predict": "POST - Eye disease prediction",
            "/status": "GET - Detailed API status"
        }
    })

@app.route("/status", methods=["GET"])
def status():
    model_status = "Not Loaded"
    try:
        if model is not None:
            model_status = "Loaded"
        elif os.path.exists(MODEL_PATH):
            model_status = "Available (Not Loaded)"
        else:
            model_status = "Model File Missing"
    except:
        model_status = "Error"
    
    return jsonify({
        "service": "Eye Disease Detection API",
        "status": "Active",
        "model_status": model_status,
        "model_loaded": model is not None,
        "supported_conditions": CATEGORIES,
        "endpoints": {
            "/": "GET - API info",
            "/predict": "POST - Eye disease prediction",
            "/status": "GET - Detailed API status"
        },
        "usage": {
            "method": "POST",
            "endpoint": "/predict",
            "content_type": "multipart/form-data",
            "field_name": "file",
            "supported_formats": ["jpg", "jpeg", "png"]
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Load model lazily on first prediction request
        current_model = load_model_lazy()

        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "No file uploaded",
                "message": "Please upload an image file"
            }), 400

        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "No file selected",
                "message": "Please select an image file"
            }), 400

        # Process the image using PIL
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((224, 224))
        
        # Convert to array
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32') / 255.0

        # Make Prediction
        prediction = current_model.predict(img_array)
        predicted_index = np.argmax(prediction)
        result = CATEGORIES[predicted_index]
        confidence = float(prediction[0][predicted_index])

        logging.info(f"Prediction: {result} with confidence: {confidence}")

        return jsonify({
            "success": True,
            "prediction": result,
            "confidence": confidence,
            "confidence_percentage": f"{confidence:.2%}",
            "all_predictions": {
                CATEGORIES[i]: float(prediction[0][i]) 
                for i in range(len(CATEGORIES))
            },
            "message": f"Detected: {result} with {confidence:.2%} confidence"
        })

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "An error occurred while processing the image"
        }), 500

if __name__ == "__main__":
    logging.info("Starting Flask app...")
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
