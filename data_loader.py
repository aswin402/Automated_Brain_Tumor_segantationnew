import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging
from config import DATA_CONFIG, RANDOM_SEED

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, training_dir, testing_dir, image_size=(256, 256)):
        self.training_dir = training_dir
        self.testing_dir = testing_dir
        self.image_size = image_size
        self.classes = DATA_CONFIG['classes']
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.classes)
        
    def load_images_from_directory(self, directory_path):
        """Load all images from directory structure with class labels."""
        images = []
        labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(directory_path, class_name)
            if not os.path.exists(class_dir):
                logger.warning(f"Directory not found: {class_dir}")
                continue
                
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            logger.info(f"Loading {len(image_files)} images from {class_name}")
            
            for image_file in image_files:
                image_path = os.path.join(class_dir, image_file)
                try:
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        logger.warning(f"Failed to read: {image_path}")
                        continue
                        
                    image = cv2.resize(image, self.image_size)
                    images.append(image)
                    labels.append(class_name)
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {e}")
                    continue
        
        return np.array(images), np.array(labels)
    
    def load_data(self):
        """Load training and testing data."""
        logger.info("Loading training data...")
        X_train_raw, y_train = self.load_images_from_directory(self.training_dir)
        
        logger.info("Loading testing data...")
        X_test_raw, y_test = self.load_images_from_directory(self.testing_dir)
        
        # Encode labels
        y_train_encoded = self.label_encoder.transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        logger.info(f"Training samples: {len(X_train_raw)}")
        logger.info(f"Testing samples: {len(X_test_raw)}")
        logger.info(f"Classes: {self.classes}")
        
        return X_train_raw, y_train_encoded, X_test_raw, y_test_encoded, y_train, y_test
    
    def split_training_data(self, X_train, y_train, test_size=0.2, val_size=0.2):
        """Split training data into train, validation, and test sets."""
        X_train_split, X_temp, y_train_split, y_temp = train_test_split(
            X_train, y_train, test_size=(test_size + val_size), 
            random_state=RANDOM_SEED, stratify=y_train
        )
        
        val_ratio = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=1-val_ratio,
            random_state=RANDOM_SEED, stratify=y_temp
        )
        
        logger.info(f"Train: {len(X_train_split)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train_split, X_val, X_test, y_train_split, y_val, y_test


def get_data_splits(test_size=0.2, val_size=0.2):
    """Convenience function to load and split data."""
    loader = DataLoader(DATA_CONFIG['training_dir'], DATA_CONFIG['testing_dir'])
    X_train, y_train, X_test_orig, y_test_orig, y_train_labels, y_test_labels = loader.load_data()
    
    X_train_split, X_val, X_test, y_train_split, y_val, y_test = loader.split_training_data(
        X_train, y_train, test_size=test_size, val_size=val_size
    )
    
    return {
        'X_train': X_train_split,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train_split,
        'y_val': y_val,
        'y_test': y_test,
        'X_test_original': X_test_orig,
        'y_test_original': y_test_orig,
        'y_train_labels': y_train_labels,
        'y_test_labels': y_test_labels,
        'label_encoder': loader.label_encoder,
        'classes': loader.classes,
    }


if __name__ == "__main__":
    data = get_data_splits()
    print(f"Training set shape: {data['X_train'].shape}")
    print(f"Validation set shape: {data['X_val'].shape}")
    print(f"Testing set shape: {data['X_test'].shape}")
    print(f"Classes: {data['classes']}")
