import numpy as np
import pandas as pd
import logging
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from config import CLASSIFIER_CONFIG, RANDOM_SEED, TRANSFER_LEARNING_CONFIG

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
        
    def fit(self, X_train, y_train, X_val=None, y_val=None, use_class_weights=True):
        """Train XGBoost or GradientBoosting model."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if use_class_weights:
            classes = np.unique(y_train)
            weights = compute_class_weight('balanced', classes=classes, y=y_train)
            sample_weights = np.array([weights[int(y)] for y in y_train])
        else:
            sample_weights = None
        
        if self.is_xgboost and X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            eval_set = [(X_val_scaled, y_val)]
            self.model.fit(
                X_train_scaled, y_train,
                sample_weight=sample_weights,
                eval_set=eval_set,
                verbose=10
            )
        else:
            self.model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        
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
        
    def fit(self, X_train, y_train, use_class_weights=True):
        """Train AdaBoost model."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if use_class_weights:
            classes = np.unique(y_train)
            weights = compute_class_weight('balanced', classes=classes, y=y_train)
            sample_weights = np.array([weights[int(y)] for y in y_train])
        else:
            sample_weights = None
        
        self.model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
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
        self.config['class_weight'] = 'balanced'
        self.model = DecisionTreeClassifier(**self.config)
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X_train, y_train, use_class_weights=True):
        """Train Decision Tree model."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if use_class_weights:
            self.model.set_params(class_weight='balanced')
        
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
        self.config['class_weight'] = 'balanced'
        self.model = SVC(**self.config)
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X_train, y_train, use_class_weights=True):
        """Train SVM model."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if use_class_weights:
            self.model.set_params(class_weight='balanced')
        
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


class TransferLearningClassifier:
    """High-capacity image classifier using transfer learning."""
    
    def __init__(self, num_classes, config=None):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for TransferLearningClassifier")
        
        self.num_classes = num_classes
        self.config = config or TRANSFER_LEARNING_CONFIG
        self.input_shape = (
            self.config['image_size'][0],
            self.config['image_size'][1],
            self.config['input_channels']
        )
        self.autotune = tf.data.AUTOTUNE
        self.augmentation_layers = self._build_augmenter()
        self.base_layer_name = None
        self.model = self._build_model()
        self.history = {}
        self.is_fitted = False
    
    def _build_augmenter(self):
        aug = self.config['augmentations']
        return keras.Sequential([
            layers.RandomFlip('horizontal_and_vertical'),
            layers.RandomRotation(aug['rotation'] / 360.0),
            layers.RandomTranslation(aug['height_shift'], aug['width_shift']),
            layers.RandomZoom((-aug['zoom'], aug['zoom'])),
            layers.RandomContrast(aug['contrast_delta']),
        ])
    
    def _create_base_model(self):
        base_name = self.config['base_model']
        if not hasattr(tf.keras.applications, base_name):
            raise ValueError(f"Unsupported base model: {base_name}")
        base_cls = getattr(tf.keras.applications, base_name)
        base_model = base_cls(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape,
            pooling='avg'
        )
        base_model.trainable = False
        return base_model
    
    def _build_model(self):
        self.base_model = self._create_base_model()
        self.base_layer_name = self.base_model.name
        inputs = layers.Input(shape=self.input_shape)
        x = layers.Lambda(tf.keras.applications.efficientnet.preprocess_input)(inputs)
        x = self.base_model(x, training=False)
        x = layers.Dropout(self.config['dropout_rate'])(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        model = keras.Model(inputs, outputs, name='transfer_learning_classifier')
        optimizer = keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=self.config['label_smoothing']),
            metrics=[
                keras.metrics.CategoricalAccuracy(name='accuracy'),
                keras.metrics.TopKCategoricalAccuracy(k=2, name='top2'),
                keras.metrics.AUC(name='auc'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.Precision(name='precision'),
            ]
        )
        return model
    
    def _format_image(self, image):
        image = tf.cast(image, tf.float32)
        image = tf.expand_dims(image, -1)
        image = tf.image.grayscale_to_rgb(image)
        image = tf.image.resize(image, self.config['image_size'])
        scale = tf.cond(tf.reduce_max(image) <= 1.5, lambda: tf.constant(255.0), lambda: tf.constant(1.0))
        image = image * scale
        return image
    
    def _apply_color_jitter(self, image):
        aug = self.config['augmentations']
        normalized = image / 255.0
        normalized = tf.image.random_brightness(normalized, aug['brightness_delta'])
        normalized = tf.image.random_contrast(
            normalized,
            1 - aug['contrast_delta'],
            1 + aug['contrast_delta']
        )
        normalized = tf.image.random_saturation(
            normalized,
            1 - aug['saturation_delta'],
            1 + aug['saturation_delta']
        )
        normalized = tf.image.random_hue(normalized, aug['hue_delta'])
        normalized = tf.clip_by_value(normalized, 0.0, 1.0)
        return normalized * 255.0
    
    def _augment(self, image, label):
        image = self.augmentation_layers(image, training=True)
        image = self._apply_color_jitter(image)
        return image, label
    
    def _mixup_batch(self, images, labels):
        if not self.config['use_mixup']:
            return images, labels
        batch_size = tf.shape(images)[0]
        indices = tf.random.shuffle(tf.range(batch_size))
        shuffled_images = tf.gather(images, indices)
        shuffled_labels = tf.gather(labels, indices)
        lam = tf.random.uniform([], 0.0, 1.0)
        lam = tf.maximum(lam, 1.0 - lam)
        images = lam * images + (1.0 - lam) * shuffled_images
        labels = lam * labels + (1.0 - lam) * shuffled_labels
        return images, labels
    
    def _dataset(self, images, labels=None, training=False):
        images = images.astype('float32')
        if labels is None:
            ds = tf.data.Dataset.from_tensor_slices(images)
            ds = ds.map(lambda x: self._format_image(x), num_parallel_calls=self.autotune)
            if training:
                ds = ds.map(lambda x: self.augmentation_layers(x, training=True), num_parallel_calls=self.autotune)
                ds = ds.map(lambda x: self._apply_color_jitter(x), num_parallel_calls=self.autotune)
            ds = ds.batch(self.config['batch_size']).prefetch(self.autotune)
            return ds
        labels = labels.astype('int32')
        ds = tf.data.Dataset.from_tensor_slices((images, labels))
        ds = ds.map(
            lambda x, y: (self._format_image(x), tf.one_hot(y, self.num_classes)),
            num_parallel_calls=self.autotune
        )
        if training:
            ds = ds.shuffle(4096)
            ds = ds.map(self._augment, num_parallel_calls=self.autotune)
        ds = ds.batch(self.config['batch_size'])
        if training:
            ds = ds.map(self._mixup_batch, num_parallel_calls=self.autotune)
        ds = ds.prefetch(self.autotune)
        return ds
    
    def _recompile(self, lr):
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.model.compile(
            optimizer=optimizer,
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=self.config['label_smoothing']),
            metrics=self.model.metrics,
        )
    
    def fine_tune(self):
        self.base_model.trainable = True
        for layer in self.base_model.layers[:self.config['fine_tune_at']]:
            layer.trainable = False
        self._recompile(self.config['fine_tune_learning_rate'])
    
    def fit(self, X_train, y_train, X_val, y_val, checkpoint_path=None, verbose=1, class_weight=None):
        train_ds = self._dataset(X_train, y_train, training=True)
        val_ds = self._dataset(X_val, y_val, training=False)
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=4, min_lr=1e-7),
        ]
        if checkpoint_path:
            callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    checkpoint_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    save_weights_only=False
                )
            )
        history_stage_1 = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config['epochs_stage_1'],
            verbose=verbose,
            callbacks=callbacks,
            class_weight=class_weight
        )
        self.history['stage_1'] = history_stage_1.history
        self.fine_tune()
        history_stage_2 = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config['epochs_stage_2'],
            verbose=verbose,
            callbacks=callbacks,
            class_weight=class_weight
        )
        self.history['stage_2'] = history_stage_2.history
        self.is_fitted = True
        logger.info("Transfer learning model trained successfully")
    
    def predict(self, X):
        ds = self._dataset(X)
        predictions = self.model.predict(ds, verbose=0)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X, use_tta=False):
        if not use_tta:
            ds = self._dataset(X)
            return self.model.predict(ds, verbose=0)
        probabilities = []
        for _ in range(self.config['tta_runs']):
            ds = self._dataset(X, training=True)
            probabilities.append(self.model.predict(ds, verbose=0))
        return np.mean(probabilities, axis=0)
    
    def evaluate(self, X, y):
        ds = self._dataset(X, y, training=False)
        return self.model.evaluate(ds, verbose=0)
    
    def save(self, filepath):
        self.model.save(filepath)
        logger.info(f"Transfer learning model saved to {filepath}")
    
    def load(self, filepath):
        self.model = keras.models.load_model(filepath)
        target_layer = self.base_layer_name or self.config['base_model'].lower()
        try:
            self.base_model = self.model.get_layer(target_layer)
        except ValueError:
            self.base_model = next((layer for layer in self.model.layers if isinstance(layer, keras.Model)), None)
        self.is_fitted = True
        logger.info(f"Transfer learning model loaded from {filepath}")


class EnsembleVotingClassifier:
    """Ensemble Voting Classifier combining multiple models."""
    
    def __init__(self, models=None):
        self.models = models or {}
        self.voting_model = None
        self.is_fitted = False
        
    def fit(self, X_train, y_train, X_val=None, y_val=None, use_class_weights=True):
        """Train all models in the ensemble."""
        estimators = []
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            if hasattr(model, 'fit'):
                if name in ['xgboost']:
                    model.fit(X_train, y_train, X_val, y_val, use_class_weights)
                else:
                    model.fit(X_train, y_train, use_class_weights)
                
                estimators.append((name, model.model))
        
        if estimators:
            self.voting_model = VotingClassifier(estimators=estimators, voting='soft')
            self.voting_model.fit(X_train, y_train)
            self.is_fitted = True
            logger.info("Ensemble voting classifier trained successfully")
    
    def predict(self, X):
        """Predict using ensemble voting."""
        if self.voting_model is None:
            raise ValueError("Model not fitted yet")
        
        scaled_X = self.models[list(self.models.keys())[0]].scaler.transform(X)
        return self.voting_model.predict(scaled_X)
    
    def predict_proba(self, X):
        """Predict probabilities using ensemble voting."""
        if self.voting_model is None:
            raise ValueError("Model not fitted yet")
        
        scaled_X = self.models[list(self.models.keys())[0]].scaler.transform(X)
        return self.voting_model.predict_proba(scaled_X)


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
    
    @staticmethod
    def create_transfer_learning(num_classes, config=None):
        return TransferLearningClassifier(num_classes, config)


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
