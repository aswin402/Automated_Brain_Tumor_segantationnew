#!/usr/bin/env python3
"""
Inference script for making predictions on new MRI images.
"""

import os
import numpy as np
import cv2
import logging
from config import PATHS
from preprocessing import ImagePreprocessor
from feature_extraction import RadiomicsFeatureExtractor
from classifiers import ModelFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BrainTumorInference:
    """Inference engine for brain tumor classification."""
    
    def __init__(self, model_type='xgboost', class_names=None):
        self.model_type = model_type
        self.class_names = class_names or ['no_tumor', 'glioma_tumor', 'meningioma_tumor', 'pituitary_tumor']
        self.preprocessor = ImagePreprocessor(target_size=(256, 256))
        self.feature_extractor = RadiomicsFeatureExtractor()
        self.model = None
        
    def load_model(self, model_path=None):
        """Load trained classifier."""
        if model_path is None:
            model_path = os.path.join(PATHS['models_dir'], f'{self.model_type}_model.pkl')
        
        logger.info(f"Loading {self.model_type} model from {model_path}")
        
        if self.model_type == 'xgboost':
            self.model = ModelFactory.create_xgboost()
        elif self.model_type == 'adaboost':
            self.model = ModelFactory.create_adaboost()
        elif self.model_type == 'decision_tree':
            self.model = ModelFactory.create_decision_tree()
        elif self.model_type == 'svm':
            self.model = ModelFactory.create_svm()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model.load(model_path)
        logger.info(f"Model loaded successfully")
        
        return self.model
    
    def predict_single_image(self, image_path):
        """Predict tumor class for a single MRI image."""
        # Load and preprocess image
        logger.info(f"Loading image from {image_path}")
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        
        # Preprocess
        processed_image = self.preprocessor.preprocess_single_image(image)
        
        # Extract features
        features = self.feature_extractor.extract_all_features(processed_image)
        features_array = np.array(list(features.values())).reshape(1, -1)
        
        # Predict
        prediction = self.model.predict(features_array)[0]
        probability = self.model.predict_proba(features_array)[0]
        
        result = {
            'image_path': image_path,
            'predicted_class': self.class_names[prediction],
            'prediction_id': prediction,
            'confidence': float(probability[prediction]),
            'all_probabilities': {self.class_names[i]: float(prob) for i, prob in enumerate(probability)},
        }
        
        return result
    
    def predict_batch(self, image_paths):
        """Predict tumor class for multiple images."""
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict_single_image(image_path)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                continue
        
        return results
    
    def predict_from_directory(self, directory_path):
        """Predict for all images in a directory."""
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_paths = []
        
        for filename in os.listdir(directory_path):
            if filename.lower().endswith(image_extensions):
                image_paths.append(os.path.join(directory_path, filename))
        
        logger.info(f"Found {len(image_paths)} images in {directory_path}")
        return self.predict_batch(image_paths)


def main_single_image():
    """Example: Predict for a single image."""
    # Initialize inference engine
    inference = BrainTumorInference(model_type='xgboost')
    
    # Load model
    inference.load_model()
    
    # Predict on a test image
    test_image = input("\nEnter image path: ").strip()
    
    result = inference.predict_single_image(test_image)
    
    if result:
        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        print(f"Image: {result['image_path']}")
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("\nProbabilities:")
        for class_name, prob in result['all_probabilities'].items():
            print(f"  {class_name}: {prob:.4f}")
        print("="*60)


def main_batch():
    """Example: Predict for multiple images in a directory."""
    # Initialize inference engine
    inference = BrainTumorInference(model_type='xgboost')
    
    # Load model
    inference.load_model()
    
    # Predict on all images in a directory
    test_dir = input("\nEnter directory path: ").strip()
    
    results = inference.predict_from_directory(test_dir)
    
    print("\n" + "="*60)
    print("BATCH PREDICTION RESULTS")
    print("="*60)
    
    for result in results:
        print(f"\nImage: {os.path.basename(result['image_path'])}")
        print(f"  Predicted: {result['predicted_class']}")
        print(f"  Confidence: {result['confidence']:.2%}")


def main_compare_models():
    """Example: Compare predictions from all models."""
    test_image_path = input("\nEnter image path: ").strip()
    
    model_types = ['xgboost', 'adaboost', 'decision_tree', 'svm']
    
    print("\n" + "="*80)
    print("COMPARING ALL CLASSIFIERS")
    print("="*80)
    
    for model_type in model_types:
        try:
            inference = BrainTumorInference(model_type=model_type)
            inference.load_model()
            result = inference.predict_single_image(test_image_path)
            
            if result:
                print(f"\n{model_type.upper()}")
                print(f"  Prediction: {result['predicted_class']}")
                print(f"  Confidence: {result['confidence']:.2%}")
        except Exception as e:
            print(f"\n{model_type.upper()}: Error - {e}")


if __name__ == "__main__":
    print("Brain Tumor Classification Inference Engine")
    print("==========================================\n")
    
    print("Choose an option:")
    print("1. Single image prediction")
    print("2. Batch prediction from directory")
    print("3. Compare all models")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice == '1':
        main_single_image()
    elif choice == '2':
        main_batch()
    elif choice == '3':
        main_compare_models()
    else:
        print("Invalid choice")
