import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_CONFIG = {
    'training_dir': os.path.join(PROJECT_ROOT, 'Training2'),
    'testing_dir': os.path.join(PROJECT_ROOT, 'Testing1'),
    'image_size': (256, 256),
    'classes': ['notumor', 'glioma', 'meningioma', 'pituitary'],
    'num_classes': 4,
    'test_split': 0.2,
    'validation_split': 0.2,
}

PREPROCESSING_CONFIG = {
    'resize_size': (256, 256),
    'normalize': True,
    'augmentation': True,
    'rotation_range': 30,
    'shift_range': 0.15,
    'zoom_range': 0.25,
    'flip_horizontal': True,
    'flip_vertical': True,
    'clahe_enabled': True,
    'clahe_clip_limit': 2.0,
    'adaptive_histogram': True,
}

UNET_CONFIG = {
    'input_shape': (256, 256, 1),
    'num_classes': 1,
    'filters_start': 64,
    'dropout_rate': 0.3,
    'learning_rate': 0.0005,
    'batch_size': 16,
    'epochs': 100,
    'validation_split': 0.2,
    'use_batch_norm': True,
}

TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 150,
    'learning_rate': 0.0005,
    'validation_split': 0.2,
    'patience': 20,
    'use_class_weights': True,
    'use_cross_validation': True,
    'cv_folds': 5,
}

CLASSIFIER_CONFIG = {
    'xgboost': {
        'max_depth': 8,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'min_child_weight': 1,
        'gamma': 0.1,
        'random_state': 42,
    },
    'adaboost': {
        'n_estimators': 200,
        'learning_rate': 0.5,
        'random_state': 42,
    },
    'decision_tree': {
        'max_depth': 20,
        'min_samples_split': 3,
        'min_samples_leaf': 1,
        'random_state': 42,
    },
    'svm': {
        'kernel': 'rbf',
        'C': 500,
        'gamma': 0.0001,
        'probability': True,
        'random_state': 42,
    },
    'ann': {
        'hidden_layers': [512, 256, 128, 64],
        'activation': 'relu',
        'dropout_rate': 0.4,
        'learning_rate': 0.0005,
        'epochs': 200,
        'batch_size': 32,
        'patience': 30,
    },
}

TRANSFER_LEARNING_CONFIG = {
    'image_size': (300, 300),
    'input_channels': 3,
    'batch_size': 32,
    'epochs_stage_1': 20,
    'epochs_stage_2': 40,
    'base_model': 'EfficientNetB3',
    'dropout_rate': 0.4,
    'learning_rate': 0.0001,
    'fine_tune_learning_rate': 1e-05,
    'fine_tune_at': 380,
    'label_smoothing': 0.1,
    'use_mixup': True,
    'mixup_alpha': 0.2,
    'use_cutmix': False,
    'cutmix_alpha': 1.0,
    'tta_runs': 8,
    'augmentations': {
        'rotation': 25,
        'width_shift': 0.1,
        'height_shift': 0.1,
        'zoom': 0.15,
        'brightness_delta': 0.15,
        'contrast_delta': 0.15,
        'hue_delta': 0.02,
        'saturation_delta': 0.2,
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
