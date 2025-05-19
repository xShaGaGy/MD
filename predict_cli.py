import os
import numpy as np
import tensorflow as tf
import cv2
import argparse
from tensorflow.keras.models import load_model

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess the image for the model"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    
    # Apply preprocessing specific to ResNet (as used in your model)
    img = tf.keras.applications.resnet_v2.preprocess_input(img)
    return np.expand_dims(img, axis=0)  # Add batch dimension

def predict_image(image_path, model_path):
    """Predict if the image is malware or benign"""
    # Load the model
    try:
        model = load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Preprocess image
    processed_img = preprocess_image(image_path)
    if processed_img is None:
        return
    
    # Make prediction
    try:
        prediction = model.predict(processed_img, verbose=1)
        
        # Get class probabilities
        benign_prob = float(prediction[0][0])
        malware_prob = float(prediction[0][1])
        
        # Determine result
        result = "Malware" if malware_prob > benign_prob else "Benign"
        confidence = malware_prob if result == "Malware" else benign_prob
        
        # Print results
        print("\n" + "="*50)
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Result: {result}")
        print(f"Confidence: {confidence*100:.2f}%")
        print(f"Benign probability: {benign_prob*100:.2f}%")
        print(f"Malware probability: {malware_prob*100:.2f}%")
        print("="*50)
        
    except Exception as e:
        print(f"Error during prediction: {e}")

def main():
    parser = argparse.ArgumentParser(description='Predict if an image contains malware or is benign')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--model', default='best_malware_detection_model.h5', help='Path to the model file')
    
    args = parser.parse_args()
    
    predict_image(args.image_path, args.model)

if __name__ == "__main__":
    main()
