#!/usr/bin/env python3
"""
Efficiently trains an Ensemble Classifier using existing feature CSVs.
Target: >90% accuracy and high confidence.
"""

import os
import numpy as np
import pandas as pd
import logging
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from config import PATHS, DATA_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Efficient Ensemble Training...")
    
    # 1. Load existing features
    f_train = os.path.join(PATHS['features_dir'], 'features_train.csv')
    f_val = os.path.join(PATHS['features_dir'], 'features_val.csv')
    f_test = os.path.join(PATHS['features_dir'], 'features_test.csv')
    
    if not all(os.path.exists(p) for p in [f_train, f_val, f_test]):
        logger.error("Feature CSVs not found! Please run cpu_pipeline.py first to generate them.")
        return

    logger.info("Loading feature CSVs...")
    df_train = pd.read_csv(f_train)
    df_val = pd.read_csv(f_val)
    df_test = pd.read_csv(f_test)
    
    # 2. Get labels
    from data_loader import get_data_splits
    data = get_data_splits()
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    
    # If augmentation was used, we need to handle y_train length mismatch
    # In cpu_pipeline.py, augmentation is False, so it should match.
    if len(df_train) != len(y_train):
        logger.warning(f"Feature count ({len(df_train)}) != Label count ({len(y_train)}).")
        # Fallback: re-extracting labels or adjusting
        # For simplicity, we assume they match if cpu_pipeline was used.
    
    X_train = df_train.values
    X_val = df_val.values
    X_test = df_test.values
    
    # 3. Import and train models
    from classifiers import ModelFactory
    from evaluation import ClassificationMetrics
    
    logger.info("Training XGBoost...")
    xgb = ModelFactory.create_xgboost()
    xgb.fit(X_train, y_train, X_val, y_val)
    
    logger.info("Training SVM...")
    svm = ModelFactory.create_svm()
    svm.fit(X_train, y_train)
    
    logger.info("Training ANN...")
    ann = ModelFactory.create_ann(input_dim=X_train.shape[1], num_classes=len(DATA_CONFIG['classes']))
    ann.fit(X_train, y_train, X_val, y_val)
    
    # 4. Ensemble Prediction (Stacking / Meta-model)
    logger.info("Training Meta-Classifier (Stacking)...")
    from sklearn.linear_model import LogisticRegression
    
    # Get probas for training meta-model
    p_train_xgb = xgb.predict_proba(X_train)
    p_train_svm = svm.predict_proba(X_train)
    p_train_ann = ann.predict_proba(X_train)
    
    X_meta_train = np.hstack([p_train_xgb, p_train_ann, p_train_svm])
    
    meta_model = LogisticRegression(random_state=42)
    meta_model.fit(X_meta_train, y_train)
    
    # Evaluate on test
    p_test_xgb = xgb.predict_proba(X_test)
    p_test_svm = svm.predict_proba(X_test)
    p_test_ann = ann.predict_proba(X_test)
    
    X_meta_test = np.hstack([p_test_xgb, p_test_ann, p_test_svm])
    y_pred = meta_model.predict(X_meta_test)
    p_ensemble = meta_model.predict_proba(X_meta_test)
    
    metrics = ClassificationMetrics.calculate_metrics(y_test, y_pred, p_ensemble)
    logger.info(f"Stacking Ensemble Accuracy: {metrics['accuracy']:.2%}")
    
    # 5. Save Model for Inference
    ensemble_data = {
        'models': {
            'xgboost': xgb,
            'ann': ann,
            'svm': svm
        },
        'meta_model': meta_model,
        'classes': DATA_CONFIG['classes']
    }
    
    ensemble_path = os.path.join(PATHS['models_dir'], 'ensemble_model.pkl')
    joblib.dump(ensemble_data, ensemble_path)
    logger.info(f"Ensemble model saved to {ensemble_path}")

if __name__ == "__main__":
    main()
