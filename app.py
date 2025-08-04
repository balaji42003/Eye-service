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
        file_size = os.path.getsize(MODEL_PATH)
        if file_size > 100 * 1024 * 1024:  # Check if file is larger than 100MB
            logging.info(f"Model already exists at {MODEL_PATH} ({file_size / (1024*1024):.1f} MB)")
            return True
        else:
            logging.warning(f"Model file exists but is too small ({file_size} bytes). Re-downloading...")
            os.remove(MODEL_PATH)
    
    try:
        logging.info("Model not found locally. Downloading from Google Drive...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        # Use session with headers to mimic browser request
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # First, get the file ID from the URL
        file_id = "1xUTbvzCI13cKQu0FEddv-AYtn_vOwgdg"
        
        # Try direct download first
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        logging.info("Attempting to download model...")
        response = session.get(download_url, stream=True)
        
        # Check if we got the virus scan warning page
        if response.status_code == 200 and 'text/html' in response.headers.get('content-type', ''):
            # This is likely the virus scan warning page
            logging.info("Got virus scan warning, looking for download confirmation...")
            
            # Look for the confirm token in the response
            import re
            confirm_pattern = r'confirm=([^&"]*)'
            match = re.search(confirm_pattern, response.text)
            
            if match:
                confirm_token = match.group(1)
                download_url_with_confirm = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={confirm_token}"
                logging.info(f"Found confirm token, downloading with confirmation...")
                response = session.get(download_url_with_confirm, stream=True)
        
        # Check if we have a valid binary response
        if response.status_code == 200:
            content_type = response.headers.get('content-type', '')
            
            # Make sure we're not getting an HTML error page
            if 'text/html' in content_type:
                logging.error("Received HTML response instead of binary file. Check the Google Drive link permissions.")
                return False
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            logging.info(f"Starting download... Expected size: {total_size / (1024*1024):.1f} MB")
            
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        # Log progress every 10MB
                        if downloaded_size % (10 * 1024 * 1024) == 0:
                            if total_size > 0:
                                progress = (downloaded_size / total_size) * 100
                                logging.info(f"Download progress: {progress:.1f}% ({downloaded_size / (1024*1024):.1f} MB)")
                            else:
                                logging.info(f"Downloaded: {downloaded_size / (1024*1024):.1f} MB")
            
            # Verify the downloaded file
            actual_size = os.path.getsize(MODEL_PATH)
            if actual_size > 100 * 1024 * 1024:  # Should be around 113MB
                logging.info(f"Model downloaded successfully! File size: {actual_size / (1024*1024):.1f} MB")
                return True
            else:
                logging.error(f"Downloaded file is too small: {actual_size} bytes. Download may have failed.")
                if os.path.exists(MODEL_PATH):
                    os.remove(MODEL_PATH)
                return False
            
        else:
            logging.error(f"Failed to download model. Status code: {response.status_code}")
            logging.error(f"Response content: {response.text[:500]}")
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
            # Check if model exists and is valid
            if os.path.exists(MODEL_PATH):
                file_size = os.path.getsize(MODEL_PATH)
                if file_size < 100 * 1024 * 1024:  # Less than 100MB is suspicious
                    logging.warning(f"Model file exists but seems corrupted (size: {file_size} bytes). Re-downloading...")
                    os.remove(MODEL_PATH)
                    if not download_model_from_drive():
                        raise Exception("Failed to download model from Google Drive")
                else:
                    logging.info(f"Using existing model file ({file_size / (1024*1024):.1f} MB)")
            else:
                # Download model if not exists
                if not download_model_from_drive():
                    raise Exception("Failed to download model from Google Drive")
            
            logging.info(f"Loading model from {MODEL_PATH}...")
            model = load_model(MODEL_PATH)
            logging.info("Model loaded successfully!")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            # If loading fails, try to re-download once
            if os.path.exists(MODEL_PATH):
                logging.info("Attempting to re-download model due to loading error...")
                os.remove(MODEL_PATH)
                try:
                    if download_model_from_drive():
                        model = load_model(MODEL_PATH)
                        logging.info("Model re-downloaded and loaded successfully!")
                    else:
                        raise Exception("Failed to re-download model")
                except Exception as retry_error:
                    logging.error(f"Error during model re-download: {retry_error}")
                    raise retry_error
            else:
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

@app.route("/health", methods=["GET"])
def health():
    """Simple health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Eye Disease Detection API",
        "timestamp": logging.Formatter().formatTime(logging.LogRecord("", 0, "", 0, "", (), None))
    })

@app.route("/model-status", methods=["GET"])
def model_status():
    """Detailed model status endpoint"""
    status_info = {
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "model_url": MODEL_URL
    }
    
    if os.path.exists(MODEL_PATH):
        file_size = os.path.getsize(MODEL_PATH)
        status_info.update({
            "model_file_exists": True,
            "model_file_size_mb": round(file_size / (1024*1024), 2),
            "model_file_valid": file_size > 100 * 1024 * 1024
        })
    else:
        status_info.update({
            "model_file_exists": False,
            "model_file_size_mb": 0,
            "model_file_valid": False
        })
    
    return jsonify(status_info)
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
    
@app.route("/status", methods=["GET"])
def status():
    model_status = "Not Loaded"
    model_source = "Local"
    
    try:
        if model is not None:
            model_status = "Loaded"
            model_source = "Local (Downloaded from Google Drive)"
        elif os.path.exists(MODEL_PATH):
            file_size = os.path.getsize(MODEL_PATH)
            if file_size > 100 * 1024 * 1024:
                model_status = "Available (Not Loaded)"
                model_source = f"Local ({file_size / (1024*1024):.1f} MB)"
            else:
                model_status = "File Corrupted - Will Re-download"
                model_source = "Google Drive (Auto-download)"
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
            "/status": "GET - Detailed API status",
            "/health": "GET - Simple health check",
            "/model-status": "GET - Detailed model status"
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
