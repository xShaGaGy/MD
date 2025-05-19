import os
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, request, render_template, jsonify

# Set this to avoid GPU memory issues
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'best_malware_detection_model.h5'  # Update this with your model path
model = None

def load_model():
    global model
    if model is None:
        print("Loading malware detection model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
    return model

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess the uploaded image for the model"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    
    # Apply preprocessing specific to ResNet (as used in your model)
    img = tf.keras.applications.resnet_v2.preprocess_input(img)
    return np.expand_dims(img, axis=0)  # Add batch dimension

def predict_malware(image_path):
    """Predict whether the image is malware or benign"""
    # Load model if not already loaded
    model = load_model()
    
    # Preprocess image
    processed_img = preprocess_image(image_path)
    if processed_img is None:
        return {"error": "Failed to process image"}
    
    # Make prediction
    prediction = model.predict(processed_img)
    
    # Get class probabilities
    benign_prob = float(prediction[0][0])
    malware_prob = float(prediction[0][1])
    
    # Determine result
    is_malware = malware_prob > benign_prob
    
    # Return result
    return {
        "result": "Malware" if is_malware else "Benign",
        "confidence": float(malware_prob if is_malware else benign_prob),
        "malware_probability": float(malware_prob),
        "benign_probability": float(benign_prob)
    }

@app.route('/')
def index():
    """Render the upload page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle file upload and return prediction"""
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    
    # Check if a file was selected
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Check file type
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    if not '.' in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({"error": "File type not supported. Please upload PNG, JPG, or JPEG"}), 400
    
    # Save the file temporarily
    temp_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(temp_path)
    
    try:
        # Process the file
        result = predict_malware(temp_path)
        
        # Clean up
        os.remove(temp_path)
        
        # Return the result
        return jsonify(result)
    except Exception as e:
        # Clean up in case of error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Make sure model loads on startup
    load_model()
    # Start the server
    app.run(debug=True, host='0.0.0.0', port=5000)
