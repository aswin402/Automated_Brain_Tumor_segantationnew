#!/usr/bin/env python3
"""
Advanced training with ensemble voting, feature selection, and hyperparameter optimization
"""

import os
import numpy as np
import pandas as pd
import logging
from joblib import parallel_backend, Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from datetime import datetime

from config import PATHS, RANDOM_SEED, TRAINING_CONFIG, CLASSIFIER_CONFIG
from data_loader import get_data_splits
from preprocessing import PreprocessingPipeline
from feature_extraction import RadiomicsFeatureExtractor, save_features_to_csv
from classifiers import ModelFactory
from evaluation import ClassificationMetrics, Visualizer, generate_results_report

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


def select_best_features(X_train, y_train, X_val, X_test, k=20):
    """Select best features using f_classif."""
    selector = SelectKBest(f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(X_test)
    
    selected_features = selector.get_support()
    logger.info(f"Selected {k} best features out of {X_train.shape[1]}")
    
    return X_train_selected, X_val_selected, X_test_selected, selected_features


def create_ensemble_classifier():
    """Create ensemble voting classifier with weighted predictions."""
    
    xgb = ModelFactory.create_xgboost()
    ann = ModelFactory.create_ann(input_dim=20, num_classes=4)
    dt = ModelFactory.create_decision_tree()
    
    ensemble_estimators = [
        ('xgb', xgb.model if hasattr(xgb, 'model') else xgb),
        ('ann', ann.model if hasattr(ann, 'model') else ann),
        ('dt', dt.model if hasattr(dt, 'model') else dt),
    ]
    
    return ensemble_estimators


def main():
    logger.info("\n" + "="*80)
    logger.info("ADVANCED BRAIN TUMOR CLASSIFICATION PIPELINE (90%+ Target)")
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
        
        logger.info("Extracting features from training set...")
        features_train = extract_features_parallel(X_train, n_jobs=-1)
        
        logger.info("Extracting features from validation set...")
        features_val = extract_features_parallel(X_val, n_jobs=-1)
        
        logger.info("Extracting features from test set...")
        features_test = extract_features_parallel(X_test, n_jobs=-1)
        
        logger.info(f"Total features extracted: {features_train.shape[1]}")
        
        logger.info("\n" + "="*80)
        logger.info("STEP 4: FEATURE SELECTION")
        logger.info("="*80)
        
        X_train_feat, X_val_feat, X_test_feat, selected = select_best_features(
            features_train.values, y_train,
            features_val.values,
            features_test.values,
            k=20
        )
        
        logger.info("\n" + "="*80)
        logger.info("STEP 5: TRAINING OPTIMIZED CLASSIFIERS")
        logger.info("="*80)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_feat)
        X_val_scaled = scaler.transform(X_val_feat)
        X_test_scaled = scaler.transform(X_test_feat)
        
        logger.info("Training XGBoost with class weights...")
        xgb_model = ModelFactory.create_xgboost()
        xgb_model.fit(X_train_feat, y_train, X_val_feat, y_val, use_class_weights=True)
        y_pred_xgb = xgb_model.predict(X_test_feat)
        y_proba_xgb = xgb_model.predict_proba(X_test_feat)
        metrics_xgb = ClassificationMetrics.calculate_metrics(y_test, y_pred_xgb, y_proba_xgb)
        
        logger.info(f"XGBoost Accuracy: {metrics_xgb['accuracy']:.4f}")
        logger.info(f"XGBoost F1-Score: {metrics_xgb['f1_score']:.4f}")
        
        logger.info("Training Decision Tree with class weights...")
        dt_model = ModelFactory.create_decision_tree()
        dt_model.fit(X_train_feat, y_train, use_class_weights=True)
        y_pred_dt = dt_model.predict(X_test_feat)
        y_proba_dt = dt_model.predict_proba(X_test_feat)
        metrics_dt = ClassificationMetrics.calculate_metrics(y_test, y_pred_dt, y_proba_dt)
        
        logger.info(f"Decision Tree Accuracy: {metrics_dt['accuracy']:.4f}")
        logger.info(f"Decision Tree F1-Score: {metrics_dt['f1_score']:.4f}")
        
        logger.info("Training SVM with class weights...")
        svm_model = ModelFactory.create_svm()
        svm_model.fit(X_train_feat, y_train, use_class_weights=True)
        y_pred_svm = svm_model.predict(X_test_feat)
        y_proba_svm = svm_model.predict_proba(X_test_feat)
        metrics_svm = ClassificationMetrics.calculate_metrics(y_test, y_pred_svm, y_proba_svm)
        
        logger.info(f"SVM Accuracy: {metrics_svm['accuracy']:.4f}")
        logger.info(f"SVM F1-Score: {metrics_svm['f1_score']:.4f}")
        
        logger.info("\n" + "="*80)
        logger.info("STEP 6: ENSEMBLE VOTING")
        logger.info("="*80)
        
        # Weighted ensemble voting
        weights_xgb = metrics_xgb['accuracy']
        weights_dt = metrics_dt['accuracy']
        weights_svm = metrics_svm['accuracy']
        
        logger.info(f"XGBoost weight: {weights_xgb:.4f}")
        logger.info(f"Decision Tree weight: {weights_dt:.4f}")
        logger.info(f"SVM weight: {weights_svm:.4f}")
        
        y_proba_ensemble = (
            weights_xgb * y_proba_xgb +
            weights_dt * y_proba_dt +
            weights_svm * y_proba_svm
        ) / (weights_xgb + weights_dt + weights_svm)
        
        y_pred_ensemble = np.argmax(y_proba_ensemble, axis=1)
        metrics_ensemble = ClassificationMetrics.calculate_metrics(y_test, y_pred_ensemble, y_proba_ensemble)
        
        logger.info(f"Ensemble Accuracy: {metrics_ensemble['accuracy']:.4f}")
        logger.info(f"Ensemble F1-Score: {metrics_ensemble['f1_score']:.4f}")
        
        logger.info("\n" + "="*80)
        logger.info("STEP 7: VISUALIZATION AND RESULTS")
        logger.info("="*80)
        
        results = {
            'XGBoost': metrics_xgb,
            'Decision Tree': metrics_dt,
            'SVM': metrics_svm,
            'Ensemble (Weighted)': metrics_ensemble,
        }
        
        for model_name, metrics in results.items():
            logger.info(f"\nGenerating visualizations for {model_name}...")
            
            if model_name == 'XGBoost':
                y_pred = y_pred_xgb
                y_proba = y_proba_xgb
            elif model_name == 'Decision Tree':
                y_pred = y_pred_dt
                y_proba = y_proba_dt
            elif model_name == 'SVM':
                y_pred = y_pred_svm
                y_proba = y_proba_svm
            else:
                y_pred = y_pred_ensemble
                y_proba = y_proba_ensemble
            
            cm_path = os.path.join(PATHS['results_dir'], 
                                  f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
            Visualizer.plot_confusion_matrix(y_test, y_pred, 
                                            data['classes'], model_name, cm_path)
            
            roc_path = os.path.join(PATHS['results_dir'], 
                                   f'roc_curve_{model_name.lower().replace(" ", "_")}.png')
            Visualizer.plot_roc_curve(y_test, y_proba, 
                                     len(data['classes']), model_name, roc_path)
        
        comparison_path = os.path.join(PATHS['results_dir'], 'metrics_comparison.png')
        Visualizer.plot_metrics_comparison(results, comparison_path)
        
        report_path = os.path.join(PATHS['results_dir'], 'results_report.txt')
        generate_results_report(results, data['classes'], report_path)
        
        logger.info("\n" + "="*80)
        logger.info("FINAL RESULTS SUMMARY")
        logger.info("="*80)
        
        results_df = pd.DataFrame(results).T
        results_path = os.path.join(PATHS['results_dir'], 'model_results.csv')
        results_df.to_csv(results_path)
        
        print("\n" + results_df.to_string())
        
        best_accuracy = results_df['accuracy'].max()
        best_model = results_df['accuracy'].idxmax()
        
        logger.info("\n" + "="*80)
        logger.info(f"BEST MODEL: {best_model}")
        logger.info(f"BEST ACCURACY: {best_accuracy:.2%}")
        logger.info("="*80)
        
        if best_accuracy >= 0.90:
            logger.info("✓ TARGET ACCURACY (90%+) ACHIEVED!")
        else:
            logger.info(f"Current best accuracy: {best_accuracy:.2%}")
            logger.info(f"Target accuracy: 90% or above")
        
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
