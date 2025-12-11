#!/usr/bin/env python3
"""
Enhanced training script with parallel feature extraction and optimized classifiers
"""

import os
import numpy as np
import pandas as pd
import logging
from joblib import parallel_backend, Parallel, delayed
from datetime import datetime

from config import PATHS, RANDOM_SEED, TRAINING_CONFIG
from data_loader import get_data_splits
from preprocessing import PreprocessingPipeline
from feature_extraction import RadiomicsFeatureExtractor, save_features_to_csv
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


def extract_features_parallel(images, n_jobs=-1):
    """Extract features from images using parallel processing."""
    extractor = RadiomicsFeatureExtractor()
    
    def extract_single(img):
        return extractor.extract_all_features(img)
    
    with parallel_backend('threading', n_jobs=n_jobs):
        all_features = Parallel()(
            delayed(extract_single)(img) for img in images
        )
    
    return pd.DataFrame(all_features)


def main():
    logger.info("\n" + "="*80)
    logger.info("ENHANCED BRAIN TUMOR CLASSIFICATION PIPELINE")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    try:
        logger.info("\n" + "="*80)
        logger.info("STEP 1: LOADING DATA")
        logger.info("="*80)
        
        data = get_data_splits(test_size=0.2, val_size=0.2)
        
        logger.info(f"Training samples: {len(data['X_train'])}")
        logger.info(f"Validation samples: {len(data['X_val'])}")
        logger.info(f"Testing samples: {len(data['X_test'])}")
        logger.info(f"Classes: {data['classes']}")
        
        logger.info("\n" + "="*80)
        logger.info("STEP 2: PREPROCESSING IMAGES")
        logger.info("="*80)
        
        pipeline = PreprocessingPipeline()
        
        X_train, X_val, X_test, y_train = pipeline.prepare_data(
            data['X_train'],
            data['X_val'],
            data['X_test'],
            data['y_train'],
            augment=True
        )
        
        y_val = data['y_val']
        y_test = data['y_test']
        
        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Validation set shape: {X_val.shape}")
        logger.info(f"Testing set shape: {X_test.shape}")
        
        logger.info("\n" + "="*80)
        logger.info("STEP 3: EXTRACTING RADIOMICS FEATURES (PARALLEL)")
        logger.info("="*80)
        
        logger.info("Extracting features from training set (using parallel processing)...")
        features_train = extract_features_parallel(X_train, n_jobs=-1)
        logger.info(f"Training features shape: {features_train.shape}")
        
        logger.info("Extracting features from validation set...")
        features_val = extract_features_parallel(X_val, n_jobs=-1)
        logger.info(f"Validation features shape: {features_val.shape}")
        
        logger.info("Extracting features from test set...")
        features_test = extract_features_parallel(X_test, n_jobs=-1)
        logger.info(f"Test features shape: {features_test.shape}")
        
        features_train_path = os.path.join(PATHS['features_dir'], 'features_train.csv')
        features_val_path = os.path.join(PATHS['features_dir'], 'features_val.csv')
        features_test_path = os.path.join(PATHS['features_dir'], 'features_test.csv')
        
        save_features_to_csv(features_train, features_train_path)
        save_features_to_csv(features_val, features_val_path)
        save_features_to_csv(features_test, features_test_path)
        
        logger.info(f"Extracted {features_train.shape[1]} features from {features_train.shape[0]} samples")
        
        logger.info("\n" + "="*80)
        logger.info("STEP 4: TRAINING CLASSIFIERS")
        logger.info("="*80)
        
        trainer = ClassifierTrainer()
        trainer.class_names = data['classes']
        trainer.label_encoder = data['label_encoder']
        
        clf_data = {
            'X_train': features_train.values,
            'X_val': features_val.values,
            'X_test': features_test.values,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
        }
        
        use_class_weights = TRAINING_CONFIG.get('use_class_weights', True)
        predictions, probabilities, y_test = trainer.train_all_models(
            clf_data,
            use_class_weights=use_class_weights
        )
        
        logger.info("\n" + "="*80)
        logger.info("STEP 5: EVALUATION AND VISUALIZATION")
        logger.info("="*80)
        
        for model_name in predictions.keys():
            logger.info(f"Generating visualizations for {model_name}...")
            
            cm_path = os.path.join(PATHS['results_dir'], 
                                  f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
            Visualizer.plot_confusion_matrix(y_test, predictions[model_name], 
                                            data['classes'], model_name, cm_path)
            
            roc_path = os.path.join(PATHS['results_dir'], 
                                   f'roc_curve_{model_name.lower().replace(" ", "_")}.png')
            Visualizer.plot_roc_curve(y_test, probabilities[model_name], 
                                     len(data['classes']), model_name, roc_path)
        
        comparison_path = os.path.join(PATHS['results_dir'], 'metrics_comparison.png')
        Visualizer.plot_metrics_comparison(trainer.results, comparison_path)
        
        report_path = os.path.join(PATHS['results_dir'], 'results_report.txt')
        generate_results_report(trainer.results, data['classes'], report_path)
        
        logger.info("\n" + "="*80)
        logger.info("MODEL PERFORMANCE SUMMARY")
        logger.info("="*80)
        
        results_df = pd.DataFrame(trainer.results).T
        results_path = os.path.join(PATHS['results_dir'], 'model_results.csv')
        results_df.to_csv(results_path)
        
        print(results_df.to_string())
        
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Results saved to: {PATHS['results_dir']}")
        logger.info(f"Models saved to: {PATHS['models_dir']}")
        logger.info(f"Features saved to: {PATHS['features_dir']}")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
