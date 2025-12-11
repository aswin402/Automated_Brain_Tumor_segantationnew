import numpy as np
import pandas as pd
import os
import logging
from datetime import datetime
from tqdm import tqdm

from config import PATHS, RANDOM_SEED
from classifiers import ModelFactory
from evaluation import ClassificationMetrics, Visualizer
from data_loader import get_data_splits
from preprocessing import PreprocessingPipeline
from feature_extraction import extract_features_batch, save_features_to_csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

np.random.seed(RANDOM_SEED)


class ClassifierTrainer:
    """Train and evaluate all classifiers."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.class_names = None
        self.label_encoder = None
        
    def prepare_data(self, augment=True):
        """Load, preprocess, and extract features from data."""
        logger.info("="*80)
        logger.info("STEP 1: LOADING DATA")
        logger.info("="*80)
        
        data = get_data_splits(test_size=0.2, val_size=0.2)
        self.class_names = data['classes']
        self.label_encoder = data['label_encoder']
        
        logger.info("="*80)
        logger.info("STEP 2: PREPROCESSING IMAGES")
        logger.info("="*80)
        
        pipeline = PreprocessingPipeline()
        X_train, X_val, X_test, y_train = pipeline.prepare_data(
            data['X_train'],
            data['X_val'],
            data['X_test'],
            data['y_train'],
            augment=augment
        )
        
        y_val = data['y_val']
        y_test = data['y_test']
        
        logger.info("="*80)
        logger.info("STEP 3: EXTRACTING RADIOMICS FEATURES")
        logger.info("="*80)
        
        logger.info("Extracting features from training set...")
        features_train, extractor = extract_features_batch(X_train)
        
        logger.info("Extracting features from validation set...")
        features_val, _ = extract_features_batch(X_val)
        
        logger.info("Extracting features from test set...")
        features_test, _ = extract_features_batch(X_test)
        
        features_train_path = os.path.join(PATHS['features_dir'], 'features_train.csv')
        features_val_path = os.path.join(PATHS['features_dir'], 'features_val.csv')
        features_test_path = os.path.join(PATHS['features_dir'], 'features_test.csv')
        
        save_features_to_csv(features_train, features_train_path)
        save_features_to_csv(features_val, features_val_path)
        save_features_to_csv(features_test, features_test_path)
        
        logger.info(f"Feature dimensions: {features_train.shape}")
        
        return {
            'X_train': features_train.values,
            'X_val': features_val.values,
            'X_test': features_test.values,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'features_train': features_train,
            'features_val': features_val,
            'features_test': features_test,
        }
    
    def train_xgboost(self, X_train, y_train, X_val, y_val, X_test, y_test, use_class_weights=True):
        """Train XGBoost classifier."""
        logger.info("\n" + "="*80)
        logger.info("TRAINING: XGBoost Classifier (with Class Weights)")
        logger.info("="*80)
        
        model = ModelFactory.create_xgboost()
        model.fit(X_train, y_train, X_val, y_val, use_class_weights=use_class_weights)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        metrics = ClassificationMetrics.calculate_metrics(y_test, y_pred, y_proba)
        self.results['XGBoost'] = metrics
        self.models['XGBoost'] = model
        
        logger.info(f"XGBoost Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"XGBoost Test F1-Score: {metrics['f1_score']:.4f}")
        
        model_path = os.path.join(PATHS['models_dir'], 'xgboost_model.pkl')
        model.save(model_path)
        
        return y_pred, y_proba
    
    def train_adaboost(self, X_train, y_train, X_test, y_test, use_class_weights=True):
        """Train AdaBoost classifier."""
        logger.info("\n" + "="*80)
        logger.info("TRAINING: AdaBoost Classifier (with Class Weights)")
        logger.info("="*80)
        
        model = ModelFactory.create_adaboost()
        model.fit(X_train, y_train, use_class_weights=use_class_weights)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        metrics = ClassificationMetrics.calculate_metrics(y_test, y_pred, y_proba)
        self.results['AdaBoost'] = metrics
        self.models['AdaBoost'] = model
        
        logger.info(f"AdaBoost Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"AdaBoost Test F1-Score: {metrics['f1_score']:.4f}")
        
        model_path = os.path.join(PATHS['models_dir'], 'adaboost_model.pkl')
        model.save(model_path)
        
        return y_pred, y_proba
    
    def train_decision_tree(self, X_train, y_train, X_test, y_test, use_class_weights=True):
        """Train Decision Tree classifier."""
        logger.info("\n" + "="*80)
        logger.info("TRAINING: Decision Tree Classifier (with Class Weights)")
        logger.info("="*80)
        
        model = ModelFactory.create_decision_tree()
        model.fit(X_train, y_train, use_class_weights=use_class_weights)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        metrics = ClassificationMetrics.calculate_metrics(y_test, y_pred, y_proba)
        self.results['Decision Tree'] = metrics
        self.models['Decision Tree'] = model
        
        logger.info(f"Decision Tree Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Decision Tree Test F1-Score: {metrics['f1_score']:.4f}")
        
        model_path = os.path.join(PATHS['models_dir'], 'decision_tree_model.pkl')
        model.save(model_path)
        
        return y_pred, y_proba
    
    def train_svm(self, X_train, y_train, X_test, y_test, use_class_weights=True):
        """Train SVM classifier."""
        logger.info("\n" + "="*80)
        logger.info("TRAINING: Support Vector Machine (SVM) (with Class Weights)")
        logger.info("="*80)
        
        model = ModelFactory.create_svm()
        model.fit(X_train, y_train, use_class_weights=use_class_weights)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        metrics = ClassificationMetrics.calculate_metrics(y_test, y_pred, y_proba)
        self.results['SVM'] = metrics
        self.models['SVM'] = model
        
        logger.info(f"SVM Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"SVM Test F1-Score: {metrics['f1_score']:.4f}")
        
        model_path = os.path.join(PATHS['models_dir'], 'svm_model.pkl')
        model.save(model_path)
        
        return y_pred, y_proba
    
    def train_ann(self, X_train, y_train, X_val, y_val, X_test, y_test, num_classes):
        """Train Artificial Neural Network classifier."""
        logger.info("\n" + "="*80)
        logger.info("TRAINING: Artificial Neural Network (ANN)")
        logger.info("="*80)
        
        input_dim = X_train.shape[1]
        model = ModelFactory.create_ann(input_dim, num_classes)
        
        model.fit(X_train, y_train, X_val, y_val, verbose=1)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        metrics = ClassificationMetrics.calculate_metrics(y_test, y_pred, y_proba)
        self.results['ANN'] = metrics
        self.models['ANN'] = model
        
        logger.info(f"ANN Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"ANN Test F1-Score: {metrics['f1_score']:.4f}")
        
        model_path = os.path.join(PATHS['models_dir'], 'ann_model.pkl')
        model.save(model_path)
        
        return y_pred, y_proba
    
    def train_all_models(self, data, use_class_weights=True):
        """Train all classifiers."""
        logger.info("\n" + "="*80)
        logger.info("TRAINING ALL CLASSIFIERS")
        if use_class_weights:
            logger.info("Using class weights for imbalanced dataset handling")
        logger.info("="*80)
        
        X_train = data['X_train']
        X_val = data['X_val']
        X_test = data['X_test']
        y_train = data['y_train']
        y_val = data['y_val']
        y_test = data['y_test']
        num_classes = len(self.class_names)
        
        predictions = {}
        probabilities = {}
        
        y_pred_xgb, y_proba_xgb = self.train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, use_class_weights)
        predictions['XGBoost'] = y_pred_xgb
        probabilities['XGBoost'] = y_proba_xgb
        
        y_pred_ada, y_proba_ada = self.train_adaboost(X_train, y_train, X_test, y_test, use_class_weights)
        predictions['AdaBoost'] = y_pred_ada
        probabilities['AdaBoost'] = y_proba_ada
        
        y_pred_dt, y_proba_dt = self.train_decision_tree(X_train, y_train, X_test, y_test, use_class_weights)
        predictions['Decision Tree'] = y_pred_dt
        probabilities['Decision Tree'] = y_proba_dt
        
        y_pred_svm, y_proba_svm = self.train_svm(X_train, y_train, X_test, y_test, use_class_weights)
        predictions['SVM'] = y_pred_svm
        probabilities['SVM'] = y_proba_svm
        
        try:
            y_pred_ann, y_proba_ann = self.train_ann(X_train, y_train, X_val, y_val, X_test, y_test, num_classes)
            predictions['ANN'] = y_pred_ann
            probabilities['ANN'] = y_proba_ann
        except ImportError as e:
            logger.warning(f"Skipping ANN training: {e}")
        
        return predictions, probabilities, y_test
    
    def evaluate_and_visualize(self, predictions, probabilities, y_test):
        """Generate evaluation metrics and visualizations."""
        logger.info("\n" + "="*80)
        logger.info("EVALUATION AND VISUALIZATION")
        logger.info("="*80)
        
        for model_name in predictions.keys():
            logger.info(f"\nGenerating visualizations for {model_name}...")
            
            cm_path = os.path.join(PATHS['results_dir'], f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
            Visualizer.plot_confusion_matrix(y_test, predictions[model_name], self.class_names, model_name, cm_path)
            
            roc_path = os.path.join(PATHS['results_dir'], f'roc_curve_{model_name.lower().replace(" ", "_")}.png')
            Visualizer.plot_roc_curve(y_test, probabilities[model_name], len(self.class_names), model_name, roc_path)
        
        comparison_path = os.path.join(PATHS['results_dir'], 'metrics_comparison.png')
        Visualizer.plot_metrics_comparison(self.results, comparison_path)
        
        report_path = os.path.join(PATHS['results_dir'], 'results_report.txt')
        from evaluation import generate_results_report
        generate_results_report(self.results, self.class_names, report_path)
    
    def save_results_to_csv(self):
        """Save all results to CSV."""
        results_df = pd.DataFrame(self.results).T
        results_path = os.path.join(PATHS['results_dir'], 'model_results.csv')
        results_df.to_csv(results_path)
        logger.info(f"Results saved to {results_path}")
        
        print("\n" + "="*80)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*80)
        print(results_df.to_string())
        print("="*80)


def main():
    """Main training pipeline."""
    logger.info("\n" + "="*80)
    logger.info("BRAIN TUMOR SEGMENTATION AND CLASSIFICATION PIPELINE")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    trainer = ClassifierTrainer()
    
    data = trainer.prepare_data(augment=True)
    
    predictions, probabilities, y_test = trainer.train_all_models(data)
    
    trainer.evaluate_and_visualize(predictions, probabilities, y_test)
    
    trainer.save_results_to_csv()
    
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*80)


if __name__ == "__main__":
    main()
