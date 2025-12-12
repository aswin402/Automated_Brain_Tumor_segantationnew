import os
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from config import PATHS, RANDOM_SEED, TRANSFER_LEARNING_CONFIG, DATA_CONFIG
from data_loader import get_data_splits
from preprocessing import ImagePreprocessor
from classifiers import ModelFactory
from evaluation import ClassificationMetrics, Visualizer, generate_results_report

try:
    import tensorflow as tf
    tf.random.set_seed(RANDOM_SEED)
except ImportError:
    raise ImportError("TensorFlow is required for transfer learning training. Install with: pip install tensorflow")

np.random.seed(RANDOM_SEED)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PATHS['logs_dir'], 'transfer_learning.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SimpleHistory:
    def __init__(self, history_dict):
        self.history = history_dict


def preprocess_images(preprocessor, images):
    processed = preprocessor.preprocess_batch(images)
    return processed.astype('float32')


def merge_histories(history_dict):
    merged = {}
    for stage_history in history_dict.values():
        for key, values in stage_history.items():
            merged.setdefault(key, [])
            merged[key].extend(values)
    return merged


def main():
    logger.info("\n" + "="*80)
    logger.info("TRANSFER LEARNING PIPELINE FOR BRAIN TUMOR CLASSIFICATION")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)

    data = get_data_splits(test_size=DATA_CONFIG['test_split'], val_size=DATA_CONFIG['validation_split'])
    class_names = data['classes']

    preprocessor = ImagePreprocessor(target_size=TRANSFER_LEARNING_CONFIG['image_size'], normalize=True)
    logger.info("Preprocessing datasets to match transfer learning input requirements")
    X_train = preprocess_images(preprocessor, data['X_train'])
    X_val = preprocess_images(preprocessor, data['X_val'])
    X_test = preprocess_images(preprocessor, data['X_test'])

    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']

    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight_map = {cls: weight for cls, weight in zip(classes, class_weights)}

    logger.info("Initializing EfficientNet transfer learning classifier")
    classifier = ModelFactory.create_transfer_learning(num_classes=DATA_CONFIG['num_classes'])

    model_path = os.path.join(PATHS['models_dir'], 'transfer_learning_classifier.keras')
    logger.info("Training staged transfer learning model with aggressive augmentations")
    classifier.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        checkpoint_path=model_path,
        verbose=1,
        class_weight=class_weight_map
    )

    logger.info("Evaluating on validation set for sanity check")
    val_metrics = classifier.evaluate(X_val, y_val)
    logger.info(f"Validation metrics: {val_metrics}")

    logger.info("Performing TTA-backed inference on test split")
    y_proba = classifier.predict_proba(X_test, use_tta=True)
    y_pred = np.argmax(y_proba, axis=1)

    metrics = ClassificationMetrics.calculate_metrics(y_test, y_pred, y_proba)
    results = {'TransferLearning-EfficientNet': metrics}
    logger.info(f"Test metrics: {metrics}")

    predictions_path = os.path.join(PATHS['results_dir'], 'transfer_learning_predictions.csv')
    pd.DataFrame({
        'true_label': y_test,
        'predicted_label': y_pred,
    }).to_csv(predictions_path, index=False)
    logger.info(f"Predictions saved to {predictions_path}")

    cm_path = os.path.join(PATHS['results_dir'], 'confusion_matrix_transfer_learning.png')
    roc_path = os.path.join(PATHS['results_dir'], 'roc_curve_transfer_learning.png')
    metrics_path = os.path.join(PATHS['results_dir'], 'metrics_comparison_transfer_learning.png')
    curves_path = os.path.join(PATHS['results_dir'], 'transfer_learning_training_curves.png')
    report_path = os.path.join(PATHS['results_dir'], 'transfer_learning_report.txt')

    Visualizer.plot_confusion_matrix(y_test, y_pred, class_names, 'TransferLearning', cm_path)
    Visualizer.plot_roc_curve(y_test, y_proba, len(class_names), 'TransferLearning', roc_path)
    Visualizer.plot_metrics_comparison(results, metrics_path)

    merged_history = merge_histories(classifier.history)
    Visualizer.plot_loss_curves(SimpleHistory(merged_history), 'TransferLearning', curves_path)

    generate_results_report(results, class_names, report_path)
    logger.info("Transfer learning training pipeline completed")


if __name__ == '__main__':
    main()
