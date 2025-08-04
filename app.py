import os
import logging
import numpy as np
import requests
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
# Google Drive direct download URL (converted from your shared link)
MODEL_URL = "https://drive.google.com/uc?export=download&id=1xUTbvzCI13cKQu0FEddv-AYtn_vOwgdg"
CATEGORIES = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"]

def download_model_from_drive():
    """Download model from Google Drive if not exists locally"""
    if os.path.exists(MODEL_PATH):
        logging.info(f"Model already exists at {MODEL_PATH}")
        return True
    
    try:
        logging.info("Model not found locally. Downloading from Google Drive...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        # Use session to handle Google Drive's virus scan warning
        session = requests.Session()
        
        # First request to get the file
        response = session.get(MODEL_URL, stream=True)
        
        # Handle Google Drive's virus scan warning for large files
        if 'virus scan warning' in response.text.lower():
            # Look for the download confirmation link
            for line in response.text.split('\n'):
                if 'export=download&amp;confirm=' in line:
                    # Extract the confirm token
                    import re
                    confirm_token = re.search(r'confirm=([^&"]*)', line)
                    if confirm_token:
                        confirm_url = f"{MODEL_URL}&confirm={confirm_token.group(1)}"
                        response = session.get(confirm_url, stream=True)
                        break
        
        # Check if we got a valid response
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        # Log progress for large files
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            if downloaded_size % (1024 * 1024 * 10) == 0:  # Log every 10MB
                                logging.info(f"Download progress: {progress:.1f}%")
            
            file_size = os.path.getsize(MODEL_PATH)
            logging.info(f"Model downloaded successfully! File size: {file_size / (1024*1024):.1f} MB")
            return True
            
        else:
            logging.error(f"Failed to download model. Status code: {response.status_code}")
            return False
            
    except Exception as e:
        logging.error(f"Error downloading model: {e}")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)  # Remove incomplete file
        return False

def load_model_lazy():
    """Lazy load the model when first needed"""
    global model
    if model is None:
        try:
            # Try to download model if not exists
            if not os.path.exists(MODEL_PATH):
                if not download_model_from_drive():
                    raise Exception("Failed to download model from Google Drive")
            
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
    model_source = "Local"
    
    try:
        if model is not None:
            model_status = "Loaded"
            model_source = "Local (Downloaded from Google Drive)" if not os.path.exists(MODEL_PATH) else "Local"
        elif os.path.exists(MODEL_PATH):
            model_status = "Available (Not Loaded)"
            file_size = os.path.getsize(MODEL_PATH)
            model_source = f"Local ({file_size / (1024*1024):.1f} MB)"
        else:
            model_status = "Will Download from Google Drive"
            model_source = "Google Drive (Auto-download)"
    except:
        model_status = "Error"
        model_source = "Unknown"
    
    return jsonify({
        "service": "Eye Disease Detection API",
        "status": "Active",
        "model_status": model_status,
        "model_source": model_source,
        "model_loaded": model is not None,
        "model_url": MODEL_URL,
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
