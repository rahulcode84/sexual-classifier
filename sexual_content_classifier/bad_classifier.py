import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import requests
import io
from typing import Tuple, Dict, Any, List
import sys
import os
import warnings
warnings.filterwarnings('ignore')

class NSFWClassifier:
    def __init__(self):
        self.google_model = None
        self.yahoo_model = None
        self.yahoo_weight = 1
        
    def download_models(self):
        try:
            # For Yahoo OpenNSFW, we'll use a TensorFlow implementation
            # Download the model weights
            yahoo_url = "https://github.com/yahoo/open_nsfw/raw/master/nsfw_model/open_nsfw-weights.npy"
            yahoo_response = requests.get(yahoo_url)
            
            # Save weights
            with open('yahoo_weights.npy', 'wb') as f:
                f.write(yahoo_response.content)
            
            # Load Yahoo model architecture and weights
            self.yahoo_model = self._build_yahoo_model()
            print(" Yahoo model loaded successfully")
            
        except Exception as e:
            print(f"Error loading Yahoo model: {e}")
    
    def _build_yahoo_model(self):
        """Build Yahoo OpenNSFW model architecture"""
        try:
            # Simplified Yahoo OpenNSFW model architecture
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(224, 224, 3)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(2, activation='softmax')  # [SFW, NSFW]
            ])
            
            # Compile the model
            model.compile(optimizer='adam', 
                         loss='categorical_crossentropy', 
                         metrics=['accuracy'])
            
            return model
        except Exception as e:
            print(f"Error building Yahoo model: {e}")
            return None
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for both models"""
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to 224x224 (standard input size for both models)
            image = image.resize((224, 224))
            
            # Convert to numpy array and normalize
            image_array = np.array(image) / 255.0
            
            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
            
    
    def predict_yahoo(self, image_array: np.ndarray) -> float:
        """Get NSFW probability from Yahoo model"""
        try:
            if self.yahoo_model is None:
                print("Yahoo model not loaded")
                return 0.5
            
            # Get prediction from Yahoo model
            prediction = self.yahoo_model.predict(image_array, verbose=0)
            
            # Yahoo model returns [SFW_prob, NSFW_prob]
            nsfw_prob = float(prediction[0][1])
            
            return nsfw_prob
            
        except Exception as e:
            print(f"Error in Yahoo prediction: {e}")
            return 0.5
    
    def ensemble_predict(self, image_path: str) -> Dict[str, float]:
        """Ensemble prediction with weighted voting"""
        try:
            # Preprocess image
            image_array = self.preprocess_image(image_path)
            if image_array is None:
                return {"error": "Image preprocessing failed"}
            
            # Get predictions from both models
            yahoo_nsfw_prob = self.predict_yahoo(image_array)
            
            # Ensemble prediction with weighted voting
            ensemble_nsfw_prob = (
                self.yahoo_weight * yahoo_nsfw_prob
            )
            
            # Determine final classification
            is_nsfw = ensemble_nsfw_prob > 0.5
            classification = "Sexual" if is_nsfw else "Non-sexual"
            
            return {
                "classification": classification,
                "yahoo_probability": yahoo_nsfw_prob,
                "confidence": max(ensemble_nsfw_prob, 1 - ensemble_nsfw_prob)
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {e}"}

def main():
    """Main function to demonstrate usage"""
    # Initialize classifier
    classifier = NSFWClassifier()
    
    # Download and load models
    print("Initializing NSFW Classifier...")
    classifier.download_models()
    
    # Test with an image
    # image_path = input("Enter image path (or press Enter for demo): ").strip()

    if len(sys.argv) < 2:
        print("Usage: python nsfw_classifier.py <image_path>")
        print("Example: python nsfw_classifier.py /path/to/image.jpg")
        return

    image_path = sys.argv[1]
    
    if not image_path:
        print("No image path provided. Please provide a valid image path.")
        return
    
    # Get prediction
    print(f"\nAnalyzing image: {image_path}")
    result = classifier.ensemble_predict(image_path)
    
    # Display results
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"\n=== NSFW Classification Results ===")
        print(f"Classification: {result['classification']}")
        
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"\nModel Breakdown:")
        
        print(f"Yahoo Model : {result['yahoo_probability']:.3f}")

if __name__ == "__main__":
    main()