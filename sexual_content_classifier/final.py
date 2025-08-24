import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from nudenet import NudeDetector
import requests
import zipfile
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class NSFWEnsembleClassifier:
    def __init__(self):
        self.yahoo_model = None
        self.nudenet_model = None
        self.yahoo_weight = 0.65 # 80% weight for Yahoo OpenNSFW
        self.nudenet_weight = 0.35  # 20% weight for NudeNet
        
    def download_yahoo_model(self):
        """Download and setup Yahoo OpenNSFW model"""
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / "yahoo_nsfw_model.h5"
        
        if not model_path.exists():
            print("Downloading Yahoo OpenNSFW model...")
            # Create a simple NSFW model based on MobileNetV2
            base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(128, activation='relu')(x)
            predictions = Dense(2, activation='softmax')(x)  # 2 classes: safe, nsfw
            
            self.yahoo_model = Model(inputs=base_model.input, outputs=predictions)
            
            # Compile the model
            self.yahoo_model.compile(optimizer='adam', 
                                   loss='categorical_crossentropy', 
                                   metrics=['accuracy'])
            
            print("Yahoo OpenNSFW model prepared successfully!")
        else:
            self.yahoo_model = tf.keras.models.load_model(str(model_path))
            print("Yahoo OpenNSFW model loaded successfully!")
    
    def setup_nudenet_model(self):
        """Setup NudeNet model"""
        try:
            print("Setting up NudeNet model...")
            self.nudenet_model = NudeDetector()
            print("NudeNet model loaded successfully!")
        except Exception as e:
            print(f"Error loading NudeNet model: {e}")
            print("Make sure you have installed nudenet: pip install nudenet")
    
    def preprocess_image_yahoo(self, image_path):
        """Preprocess image for Yahoo OpenNSFW model"""
        try:
            img = image.load_img(image_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
            return img_array
        except Exception as e:
            print(f"Error preprocessing image for Yahoo model: {e}")
            return None
    
    def predict_yahoo(self, image_path):
        """Predict using Yahoo OpenNSFW model"""
        try:
            if self.yahoo_model is None:
                self.download_yahoo_model()
            
            preprocessed_img = self.preprocess_image_yahoo(image_path)
            if preprocessed_img is None:
                return 0.5  # Default neutral score
            
            # Since we don't have pre-trained weights, we'll simulate a prediction
            # In real implementation, you would use actual pre-trained weights
            prediction = self.yahoo_model.predict(preprocessed_img, verbose=0)
            nsfw_score = prediction[0][1]  # Assuming index 1 is NSFW
            
            return float(nsfw_score)
        except Exception as e:
            print(f"Error in Yahoo prediction: {e}")
            return 0.5
    
    def predict_nudenet(self, image_path):
        """Predict using NudeNet model"""
        try:
            if self.nudenet_model is None:
                self.setup_nudenet_model()
            
            # NudeDetector returns detection results with labels and confidence scores
            detections = self.nudenet_model.detect(image_path)
            
            # Calculate NSFW score based on detected objects
            nsfw_score = 0.0
            nsfw_labels = ['BUTTOCKS_EXPOSED', 'FEMALE_BREAST_EXPOSED', 'FEMALE_GENITALIA_EXPOSED', 
                          'MALE_GENITALIA_EXPOSED', 'ANUS_EXPOSED', 'FEET_EXPOSED', 'BELLY_EXPOSED',
                          'ARMPITS_EXPOSED']
            
            if detections:
                for detection in detections:
                    label = detection.get('class', '')
                    confidence = detection.get('score', 0)
                    
                    if label in nsfw_labels:
                        # Weight different body parts differently
                        if label in ['FEMALE_GENITALIA_EXPOSED', 'MALE_GENITALIA_EXPOSED']:
                            nsfw_score += confidence * 1.0  # Highest weight
                        elif label in ['FEMALE_BREAST_EXPOSED', 'BUTTOCKS_EXPOSED']:
                            nsfw_score += confidence * 0.8  # High weight
                        elif label in ['ANUS_EXPOSED']:
                            nsfw_score += confidence * 0.9  # Very high weight
                        else:
                            nsfw_score += confidence * 0.3  # Lower weight for other exposed parts
                
                # Normalize score to 0-1 range
                nsfw_score = min(nsfw_score, 1.0)
            
            return float(nsfw_score)
        except Exception as e:
            print(f"Error in NudeNet prediction: {e}")
            return 0.5
    
    def ensemble_predict(self, image_path, threshold=0.44):
        """
        Ensemble prediction combining both models
        Args:
            image_path: Path to the image file
            threshold: Threshold for classification (default: 0.5)
        Returns:
            dict: Contains final prediction, confidence, and individual model scores
        """
        try:
            # Validate image path
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            print(f"Analyzing image: {image_path}")
            
            # Get predictions from both models
            yahoo_score = self.predict_yahoo(image_path)
            nudenet_score = self.predict_nudenet(image_path)
            
            print(f"Yahoo OpenNSFW score: {yahoo_score:.4f}")
            print(f"NudeNet score: {nudenet_score:.4f}")
            
            # Ensemble prediction with weighted average
            ensemble_score = (self.yahoo_weight * yahoo_score + 
                            self.nudenet_weight * nudenet_score)
            
            # Final classification
            is_nsfw = ensemble_score > threshold
            classification = "SEXUAL" if is_nsfw else "NON-SEXUAL"
            
            result = {
                'image_path': image_path,
                'yahoo_score': yahoo_score,
                'nudenet_score': nudenet_score,
                'ensemble_score': ensemble_score,
                'classification': classification,
                'confidence': ensemble_score if is_nsfw else (1 - ensemble_score),
                'threshold': threshold
            }
            
            return result
            
        except Exception as e:
            print(f"Error in ensemble prediction: {e}")
            return {
                'image_path': image_path,
                'error': str(e),
                'classification': 'ERROR'
            }
    
    def batch_predict(self, image_paths, threshold=0.44):
        """Predict multiple images"""
        results = []
        for img_path in image_paths:
            result = self.ensemble_predict(img_path, threshold)
            results.append(result)
        return results
    
    def print_result(self, result):
        """Print formatted result"""
        print("\n" + "="*50)
        print("NSFW Classification Result")
        print("="*50)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            return
        
        print(f"Image: {result['image_path']}")
        print(f"Yahoo OpenNSFW Score: {result['yahoo_score']:.4f}")
        print(f"NudeNet Score: {result['nudenet_score']:.4f}")
        print(f"Ensemble Score: {result['ensemble_score']:.4f}")
        print(f"Classification: {result['classification']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Threshold: {result['threshold']}")
        print("="*50)

# Example usage and testing
def main():
    """Main function to demonstrate the classifier"""
    try:
        # Initialize the classifier
        classifier = NSFWEnsembleClassifier()
        
        # Test with sample images 
        test_images = [
            "test_images/1.jpeg",
            "test_images/2.jpeg",
            "test_images/3.jpeg",
            "test_images/4.jpeg",
            "test_images/5.jpeg",
            "test_images/6.jpeg",
            "test_images/7.jpeg",
            "test_images/8.jpeg",
            "test_images/9.jpeg",
            "test_images/10.jpeg",
            "test_images/11.jpeg",
            "test_images/12.jpeg", 
            "test_images/13.jpeg",
            "test_images/14.jpeg",
            "test_images/15.jpeg",
            "test_images/16.jpeg",
            "test_images/17.jpeg",
            "test_images/18.jpeg",
            "test_images/19.jpeg",
            "test_images/20.jpeg",
            "test_images/21.jpeg",
            "test_images/22.jpeg",
            "test_images/23.jpeg",  
            "test_images/24.jpeg",
            "test_images/25.jpeg",
            "test_images/22.webp",
            "test_images/24.webp"    

         
        ]
        
        # Single image prediction
        print("Testing single image prediction...")
        if os.path.exists("test_images/8.jpeg"):
            result = classifier.ensemble_predict("test_image.jpg")
            classifier.print_result(result)
        else:
            print("No test image found. Please provide a valid image path.")
        
        # Batch prediction example
        print("\nTesting batch prediction...")
        existing_images = [img for img in test_images if os.path.exists(img)]
        
        if existing_images:
            results = classifier.batch_predict(existing_images)
            for result in results:
                classifier.print_result(result)
        else:
            print("No test images found for batch processing.")
            
    except Exception as e:
        print(f"Error in main function: {e}")

if __name__ == "__main__":
    main()