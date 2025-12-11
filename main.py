#!/usr/bin/env python3
"""
Brain Tumor Segmentation and Classification Pipeline
Using U-Net (Segmentation) + XGBoost/AdaBoost/DT/SVM/ANN (Classification)
"""

import os
import numpy as np
import logging
from datetime import datetime
import argparse

from config import PATHS, RANDOM_SEED
from data_loader import get_data_splits
from preprocessing import PreprocessingPipeline
try:
    from unet_model import build_unet_model, compile_unet_model, TENSORFLOW_AVAILABLE
except ImportError:
    TENSORFLOW_AVAILABLE = False
from feature_extraction import extract_features_batch, save_features_to_csv
from train_classifiers import ClassifierTrainer
from evaluation import Visualizer, generate_results_report

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PATHS['logs_dir'], 'pipeline.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

np.random.seed(RANDOM_SEED)


class BrainTumorPipeline:
    """Complete brain tumor segmentation and classification pipeline."""
    
    def __init__(self, train_segmentation=False, train_classification=True):
        self.train_segmentation = train_segmentation
        self.train_classification = train_classification
        self.unet_model = None
        self.classifiers = {}
        self.results = {}
        
    def log_section(self, title):
        """Log section header."""
        logger.info("\n" + "="*80)
        logger.info(title)
        logger.info("="*80)
    
    def step_1_load_data(self):
        """Step 1: Load training and testing data."""
        self.log_section("STEP 1: LOADING DATA")
        
        self.data = get_data_splits(test_size=0.2, val_size=0.2)
        
        logger.info(f"Training samples: {len(self.data['X_train'])}")
        logger.info(f"Validation samples: {len(self.data['X_val'])}")
        logger.info(f"Testing samples: {len(self.data['X_test'])}")
        logger.info(f"Classes: {self.data['classes']}")
        
        return self.data
    
    def step_2_preprocess_images(self):
        """Step 2: Preprocess images."""
        self.log_section("STEP 2: PREPROCESSING IMAGES")
        
        self.pipeline = PreprocessingPipeline()
        
        self.X_train, self.X_val, self.X_test, self.y_train = self.pipeline.prepare_data(
            self.data['X_train'],
            self.data['X_val'],
            self.data['X_test'],
            self.data['y_train'],
            augment=True
        )
        
        self.y_val = self.data['y_val']
        self.y_test = self.data['y_test']
        
        logger.info(f"Training set shape: {self.X_train.shape}")
        logger.info(f"Validation set shape: {self.X_val.shape}")
        logger.info(f"Testing set shape: {self.X_test.shape}")
        
        return self.X_train, self.X_val, self.X_test
    
    def step_3_build_unet(self):
        """Step 3: Build U-Net architecture."""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available - skipping U-Net segmentation...")
            return None
        
        if not self.train_segmentation:
            logger.info("Skipping U-Net training...")
            return None
        
        self.log_section("STEP 3: BUILDING U-NET MODEL")
        
        from config import UNET_CONFIG
        
        self.unet_model = build_unet_model(
            input_shape=UNET_CONFIG['input_shape'],
            num_classes=UNET_CONFIG['num_classes'],
            filters_start=UNET_CONFIG['filters_start'],
            dropout_rate=UNET_CONFIG['dropout_rate'],
            use_batch_norm=UNET_CONFIG.get('use_batch_norm', True)
        )
        
        self.unet_model = compile_unet_model(
            self.unet_model,
            learning_rate=UNET_CONFIG['learning_rate']
        )
        
        self.unet_model.summary()
        logger.info("U-Net model built and compiled successfully")
        
        return self.unet_model
    
    def step_4_extract_features(self):
        """Step 4: Extract radiomics features."""
        self.log_section("STEP 4: EXTRACTING RADIOMICS FEATURES")
        
        logger.info("Extracting features from training set...")
        self.features_train, _ = extract_features_batch(self.X_train)
        
        logger.info("Extracting features from validation set...")
        self.features_val, _ = extract_features_batch(self.X_val)
        
        logger.info("Extracting features from test set...")
        self.features_test, _ = extract_features_batch(self.X_test)
        
        features_train_path = os.path.join(PATHS['features_dir'], 'features_train.csv')
        features_val_path = os.path.join(PATHS['features_dir'], 'features_val.csv')
        features_test_path = os.path.join(PATHS['features_dir'], 'features_test.csv')
        
        save_features_to_csv(self.features_train, features_train_path)
        save_features_to_csv(self.features_val, features_val_path)
        save_features_to_csv(self.features_test, features_test_path)
        
        logger.info(f"Extracted {self.features_train.shape[1]} features")
        logger.info(f"Training set: {self.features_train.shape[0]} samples")
        logger.info(f"Validation set: {self.features_val.shape[0]} samples")
        logger.info(f"Testing set: {self.features_test.shape[0]} samples")
        
        return self.features_train, self.features_val, self.features_test
    
    def step_5_train_classifiers(self):
        """Step 5: Train all classification models."""
        if not self.train_classification:
            logger.info("Skipping classifier training...")
            return None
        
        self.log_section("STEP 5: TRAINING CLASSIFIERS")
        
        trainer = ClassifierTrainer()
        trainer.class_names = self.data['classes']
        trainer.label_encoder = self.data['label_encoder']
        
        data = {
            'X_train': self.features_train.values,
            'X_val': self.features_val.values,
            'X_test': self.features_test.values,
            'y_train': self.y_train,
            'y_val': self.y_val,
            'y_test': self.y_test,
        }
        
        predictions, probabilities, y_test = trainer.train_all_models(data, use_class_weights=True)
        self.results = trainer.results
        self.trainer = trainer
        
        return predictions, probabilities, y_test
    
    def step_6_evaluate_results(self, predictions, probabilities, y_test):
        """Step 6: Evaluate and visualize results."""
        self.log_section("STEP 6: EVALUATION AND VISUALIZATION")
        
        for model_name in predictions.keys():
            logger.info(f"Generating visualizations for {model_name}...")
            
            cm_path = os.path.join(PATHS['results_dir'], 
                                  f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
            Visualizer.plot_confusion_matrix(y_test, predictions[model_name], 
                                            self.data['classes'], model_name, cm_path)
            
            roc_path = os.path.join(PATHS['results_dir'], 
                                   f'roc_curve_{model_name.lower().replace(" ", "_")}.png')
            Visualizer.plot_roc_curve(y_test, probabilities[model_name], 
                                     len(self.data['classes']), model_name, roc_path)
        
        comparison_path = os.path.join(PATHS['results_dir'], 'metrics_comparison.png')
        Visualizer.plot_metrics_comparison(self.results, comparison_path)
        
        report_path = os.path.join(PATHS['results_dir'], 'results_report.txt')
        generate_results_report(self.results, self.data['classes'], report_path)
    
    def save_summary(self):
        """Save pipeline summary."""
        import pandas as pd
        
        results_df = pd.DataFrame(self.results).T
        results_path = os.path.join(PATHS['results_dir'], 'model_results.csv')
        results_df.to_csv(results_path)
        
        logger.info("\n" + "="*80)
        logger.info("MODEL PERFORMANCE SUMMARY")
        logger.info("="*80)
        print(results_df.to_string())
        logger.info("="*80)
    
    def run_full_pipeline(self):
        """Run complete pipeline."""
        self.log_section("BRAIN TUMOR SEGMENTATION AND CLASSIFICATION PIPELINE")
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Train Segmentation: {self.train_segmentation}")
        logger.info(f"Train Classification: {self.train_classification}")
        
        try:
            self.step_1_load_data()
            self.step_2_preprocess_images()
            self.step_3_build_unet()
            self.step_4_extract_features()
            
            if self.train_classification:
                results = self.step_5_train_classifiers()
                if results is not None:
                    predictions, probabilities, y_test = results
                    self.step_6_evaluate_results(predictions, probabilities, y_test)
                    self.save_summary()
            
            self.log_section("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"Results saved to: {PATHS['results_dir']}")
            logger.info(f"Models saved to: {PATHS['models_dir']}")
            logger.info(f"Features saved to: {PATHS['features_dir']}")
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}", exc_info=True)
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Brain Tumor Segmentation and Classification Pipeline'
    )
    parser.add_argument('--train-segmentation', action='store_true', 
                       help='Train U-Net segmentation model')
    parser.add_argument('--train-classification', action='store_true', default=True,
                       help='Train classification models (default: True)')
    parser.add_argument('--no-classification', action='store_true',
                       help='Skip classification training')
    
    args = parser.parse_args()
    
    train_seg = args.train_segmentation
    train_clf = not args.no_classification
    
    pipeline = BrainTumorPipeline(
        train_segmentation=train_seg,
        train_classification=train_clf
    )
    
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
