import numpy as np
import pandas as pd
import logging
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from config import CLASSIFIER_CONFIG, RANDOM_SEED

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not installed, will use GradientBoostingClassifier instead")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    TENSORFLOW_AVAILABLE = True
    tf.random.set_seed(RANDOM_SEED)
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not installed, ANN classifier will not be available")
    tf = None

np.random.seed(RANDOM_SEED)


class XGBoostClassifier:
    """XGBoost or GradientBoosting Classifier for brain tumor classification."""
    
    def __init__(self, config=None):
        self.config = config or CLASSIFIER_CONFIG['xgboost']
        
        if XGBOOST_AVAILABLE:
            self.model = xgb.XGBClassifier(**self.config)
            self.is_xgboost = True
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(
                n_estimators=self.config.get('n_estimators', 200),
                learning_rate=self.config.get('learning_rate', 0.1),
                max_depth=self.config.get('max_depth', 6),
                random_state=self.config.get('random_state', 42)
            )
            self.is_xgboost = False
            logger.warning("Using sklearn GradientBoostingClassifier instead of XGBoost")
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train XGBoost or GradientBoosting model."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if self.is_xgboost and X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            eval_set = [(X_val_scaled, y_val)]
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=eval_set,
                verbose=10
            )
        else:
            self.model.fit(X_train_scaled, y_train)
        
        self.is_fitted = True
        logger.info("Gradient Boosting model trained successfully")
        
    def predict(self, X):
        """Predict on new data."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Predict probabilities."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def save(self, filepath):
        """Save model to disk."""
        joblib.dump({'model': self.model, 'scaler': self.scaler}, filepath)
        logger.info(f"XGBoost model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from disk."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_fitted = True
        logger.info(f"XGBoost model loaded from {filepath}")


class AdaBoostClassifierModel:
    """AdaBoost Classifier."""
    
    def __init__(self, config=None, base_estimator=None):
        self.config = config or CLASSIFIER_CONFIG['adaboost']
        if base_estimator is None:
            base_estimator = DecisionTreeClassifier(max_depth=1)
        
        self.model = AdaBoostClassifier(estimator=base_estimator, **self.config)
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X_train, y_train):
        """Train AdaBoost model."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        logger.info("AdaBoost model trained successfully")
        
    def predict(self, X):
        """Predict on new data."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Predict probabilities."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def save(self, filepath):
        """Save model to disk."""
        joblib.dump({'model': self.model, 'scaler': self.scaler}, filepath)
        logger.info(f"AdaBoost model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from disk."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_fitted = True
        logger.info(f"AdaBoost model loaded from {filepath}")


class DecisionTreeClassifierModel:
    """Decision Tree Classifier."""
    
    def __init__(self, config=None):
        self.config = config or CLASSIFIER_CONFIG['decision_tree']
        self.model = DecisionTreeClassifier(**self.config)
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X_train, y_train):
        """Train Decision Tree model."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        logger.info("Decision Tree model trained successfully")
        
    def predict(self, X):
        """Predict on new data."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Predict probabilities."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def save(self, filepath):
        """Save model to disk."""
        joblib.dump({'model': self.model, 'scaler': self.scaler}, filepath)
        logger.info(f"Decision Tree model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from disk."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_fitted = True
        logger.info(f"Decision Tree model loaded from {filepath}")


class SVMClassifierModel:
    """Support Vector Machine Classifier."""
    
    def __init__(self, config=None):
        self.config = config or CLASSIFIER_CONFIG['svm']
        self.model = SVC(**self.config)
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X_train, y_train):
        """Train SVM model."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        logger.info("SVM model trained successfully")
        
    def predict(self, X):
        """Predict on new data."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Predict probabilities."""
        X_scaled = self.scaler.transform(X)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)
        else:
            distances = self.model.decision_function(X_scaled)
            proba = 1 / (1 + np.exp(-distances))
            return proba
    
    def save(self, filepath):
        """Save model to disk."""
        joblib.dump({'model': self.model, 'scaler': self.scaler}, filepath)
        logger.info(f"SVM model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from disk."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_fitted = True
        logger.info(f"SVM model loaded from {filepath}")


class ANNClassifier:
    """Artificial Neural Network Classifier (if TensorFlow available)."""
    
    def __init__(self, input_dim, num_classes, config=None):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for ANN classifier. Install with: pip install tensorflow")
        
        self.config = config or CLASSIFIER_CONFIG['ann']
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.scaler = StandardScaler()
        self.model = self._build_model()
        self.is_fitted = False
        
    def _build_model(self):
        """Build ANN model architecture."""
        model = keras.Sequential()
        
        model.add(layers.Input(shape=(self.input_dim,)))
        
        for hidden_dim in self.config['hidden_layers']:
            model.add(layers.Dense(hidden_dim, activation=self.config['activation']))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(self.config['dropout_rate']))
        
        if self.num_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:
            model.add(layers.Dense(self.num_classes, activation='softmax'))
            loss = 'sparse_categorical_crossentropy'
        
        optimizer = keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        
        logger.info("ANN model built successfully")
        return model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=1):
        """Train ANN model."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        callbacks = []
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config['patience'],
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
        
        self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val) if X_val is not None else None,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_fitted = True
        logger.info("ANN model trained successfully")
    
    def predict(self, X):
        """Predict on new data."""
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled, verbose=0)
        
        if self.num_classes == 2:
            return (predictions > 0.5).astype(int).flatten()
        else:
            return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X):
        """Predict probabilities."""
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled, verbose=0)
        
        if self.num_classes == 2:
            return np.hstack([1 - predictions, predictions])
        else:
            return predictions
    
    def save(self, filepath):
        """Save model to disk."""
        self.model.save(filepath.replace('.pkl', '_ann.h5'))
        joblib.dump(self.scaler, filepath)
        logger.info(f"ANN model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from disk."""
        self.model = keras.models.load_model(filepath.replace('.pkl', '_ann.h5'))
        self.scaler = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"ANN model loaded from {filepath}")


class ModelFactory:
    """Factory for creating classifier instances."""
    
    @staticmethod
    def create_xgboost(config=None):
        return XGBoostClassifier(config)
    
    @staticmethod
    def create_adaboost(config=None):
        return AdaBoostClassifierModel(config)
    
    @staticmethod
    def create_decision_tree(config=None):
        return DecisionTreeClassifierModel(config)
    
    @staticmethod
    def create_svm(config=None):
        return SVMClassifierModel(config)
    
    @staticmethod
    def create_ann(input_dim, num_classes, config=None):
        return ANNClassifier(input_dim, num_classes, config)


if __name__ == "__main__":
    from feature_extraction import extract_features_batch
    from data_loader import get_data_splits
    from preprocessing import PreprocessingPipeline
    
    data = get_data_splits()
    pipeline = PreprocessingPipeline()
    
    X_train, X_val, X_test, y_train = pipeline.prepare_data(
        data['X_train'][:100],
        data['X_val'][:20],
        data['X_test'][:20],
        data['y_train'][:100],
        augment=False
    )
    
    y_val = data['y_val'][:20]
    y_test = data['y_test'][:20]
    
    features_train, _ = extract_features_batch(X_train)
    features_val, _ = extract_features_batch(X_val)
    features_test, _ = extract_features_batch(X_test)
    
    xgb_model = ModelFactory.create_xgboost()
    xgb_model.fit(features_train.values, y_train, features_val.values, y_val)
    
    print("XGBoost test accuracy:", np.mean(xgb_model.predict(features_test.values) == y_test))
