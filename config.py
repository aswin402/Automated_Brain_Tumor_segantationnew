import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_CONFIG = {
    'training_dir': os.path.join(PROJECT_ROOT, 'Training'),
    'testing_dir': os.path.join(PROJECT_ROOT, 'Testing'),
    'image_size': (256, 256),
    'classes': ['no_tumor', 'glioma_tumor', 'meningioma_tumor', 'pituitary_tumor'],
    'num_classes': 4,
    'test_split': 0.2,
    'validation_split': 0.2,
}

PREPROCESSING_CONFIG = {
    'resize_size': (256, 256),
    'normalize': True,
    'augmentation': True,
    'rotation_range': 20,
    'shift_range': 0.1,
    'zoom_range': 0.2,
    'flip_horizontal': True,
    'flip_vertical': True,
}

UNET_CONFIG = {
    'input_shape': (256, 256, 1),
    'num_classes': 1,
    'filters_start': 32,
    'dropout_rate': 0.5,
    'learning_rate': 0.001,
    'batch_size': 16,
    'epochs': 50,
    'validation_split': 0.2,
}

TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.001,
    'validation_split': 0.2,
    'patience': 10,
}

CLASSIFIER_CONFIG = {
    'xgboost': {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
    },
    'adaboost': {
        'n_estimators': 100,
        'learning_rate': 1.0,
        'random_state': 42,
    },
    'decision_tree': {
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
    },
    'svm': {
        'kernel': 'rbf',
        'C': 100,
        'gamma': 0.001,
        'probability': True,
        'random_state': 42,
    },
    'ann': {
        'hidden_layers': [256, 128, 64],
        'activation': 'relu',
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'epochs': 100,
        'batch_size': 32,
        'patience': 10,
    },
}

RADIOMICS_CONFIG = {
    'bin_width': 25,
    'force_2d': True,
    'interpolator': 'sitkLinear',
}

PATHS = {
    'models_dir': os.path.join(PROJECT_ROOT, 'models'),
    'results_dir': os.path.join(PROJECT_ROOT, 'results'),
    'features_dir': os.path.join(PROJECT_ROOT, 'features'),
    'logs_dir': os.path.join(PROJECT_ROOT, 'logs'),
}

for directory in PATHS.values():
    os.makedirs(directory, exist_ok=True)

RANDOM_SEED = 42
