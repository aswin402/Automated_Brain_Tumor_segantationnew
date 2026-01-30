import numpy as np
import cv2
from scipy import ndimage
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging
from config import PREPROCESSING_CONFIG, RANDOM_SEED

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

np.random.seed(RANDOM_SEED)


class ImagePreprocessor:
    def __init__(self, target_size=(256, 256), normalize=True):
        self.target_size = target_size
        self.normalize = normalize
        self.scaler = StandardScaler()
        
    def resize_image(self, image):
        """Resize image to target size."""
        if image.shape != self.target_size:
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        return image
    
    def normalize_intensity(self, image):
        """Normalize image intensity to [0, 1] range."""
        min_val = np.min(image)
        max_val = np.max(image)
        
        if max_val - min_val > 0:
            normalized = (image - min_val) / (max_val - min_val)
        else:
            normalized = image
        
        return normalized
    
    def skull_strip_simple(self, image):
        """Simple skull stripping using threshold and morphological operations."""
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        image_stripped = image * (binary / 255.0)
        return image_stripped.astype(np.uint8)
    
    def remove_noise(self, image):
        """Apply bilateral filter to remove noise while preserving edges."""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        return denoised / 255.0
    
    def apply_clahe(self, image):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        return enhanced / 255.0
    
    def adaptive_histogram_equalization(self, image):
        """Apply adaptive histogram equalization for better contrast."""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        result = cv2.addWeighted(image, 0.7, sure_bg, 0.3, 0)
        return result / 255.0
    
    def preprocess_single_image(self, image):
        """Complete preprocessing pipeline for a single image."""
        image = self.resize_image(image)
        image = self.skull_strip_simple(image)
        image = self.remove_noise(image)
        
        if PREPROCESSING_CONFIG.get('clahe_enabled', False):
            image = self.apply_clahe(image)
        
        if PREPROCESSING_CONFIG.get('adaptive_histogram', False):
            image = self.adaptive_histogram_equalization(image)
        
        image = self.normalize_intensity(image)
        return image.astype(np.float32)
    
    def preprocess_batch(self, images):
        """Preprocess a batch of images."""
        processed = []
        for img in images:
            processed_img = self.preprocess_single_image(img)
            processed.append(processed_img)
        return np.array(processed)


class DataAugmenter:
    def __init__(self, rotation_range=20, shift_range=0.1, zoom_range=0.2,
                 flip_horizontal=True, flip_vertical=True):
        self.rotation_range = rotation_range
        self.shift_range = shift_range
        self.zoom_range = zoom_range
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        
    def rotate_image(self, image, angle):
        """Rotate image by specified angle."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        return rotated
    
    def shift_image(self, image, shift_x, shift_y):
        """Shift image by specified pixels."""
        h, w = image.shape[:2]
        matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        shifted = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        return shifted
    
    def zoom_image(self, image, zoom_factor):
        """Zoom image by specified factor."""
        h, w = image.shape[:2]
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
        
        zoomed = cv2.resize(image, (new_w, new_h))
        
        if zoom_factor < 1:
            top = (h - new_h) // 2
            left = (w - new_w) // 2
            result = np.zeros_like(image)
            result[top:top+new_h, left:left+new_w] = zoomed
        else:
            top = (new_h - h) // 2
            left = (new_w - w) // 2
            result = zoomed[top:top+h, left:left+w]
        
        return result
    
    def flip_image(self, image, direction='horizontal'):
        """Flip image horizontally or vertically."""
        if direction == 'horizontal':
            return cv2.flip(image, 1)
        else:
            return cv2.flip(image, 0)
    
    def augment_image(self, image):
        """Apply random augmentation to image."""
        aug_image = image.copy()
        
        if np.random.rand() > 0.5 and self.rotation_range > 0:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            aug_image = self.rotate_image(aug_image, angle)
        
        if np.random.rand() > 0.5 and self.shift_range > 0:
            h, w = aug_image.shape[:2]
            shift_x = int(np.random.uniform(-self.shift_range * w, self.shift_range * w))
            shift_y = int(np.random.uniform(-self.shift_range * h, self.shift_range * h))
            aug_image = self.shift_image(aug_image, shift_x, shift_y)
        
        if np.random.rand() > 0.5 and self.zoom_range > 0:
            zoom = np.random.uniform(1 - self.zoom_range, 1 + self.zoom_range)
            aug_image = self.zoom_image(aug_image, zoom)
        
        if np.random.rand() > 0.5 and self.flip_horizontal:
            aug_image = self.flip_image(aug_image, 'horizontal')
        
        if np.random.rand() > 0.5 and self.flip_vertical:
            aug_image = self.flip_image(aug_image, 'vertical')
        
        return aug_image
    
    def augment_batch(self, images, labels=None, augmentation_factor=1):
        """Augment a batch of images."""
        augmented_images = list(images)
        augmented_labels = list(labels) if labels is not None else None
        
        for i in range(len(images)):
            for _ in range(augmentation_factor):
                aug_img = self.augment_image(images[i])
                augmented_images.append(aug_img)
                if augmented_labels is not None:
                    augmented_labels.append(labels[i])
        
        if augmented_labels is not None:
            return np.array(augmented_images), np.array(augmented_labels)
        return np.array(augmented_images)


class PreprocessingPipeline:
    def __init__(self):
        self.preprocessor = ImagePreprocessor(
            target_size=PREPROCESSING_CONFIG['resize_size'],
            normalize=PREPROCESSING_CONFIG['normalize']
        )
        self.augmenter = DataAugmenter(
            rotation_range=PREPROCESSING_CONFIG['rotation_range'],
            shift_range=PREPROCESSING_CONFIG['shift_range'],
            zoom_range=PREPROCESSING_CONFIG['zoom_range'],
            flip_horizontal=PREPROCESSING_CONFIG['flip_horizontal'],
            flip_vertical=PREPROCESSING_CONFIG['flip_vertical'],
        )
        self.scaler = StandardScaler()
        
    def prepare_data(self, X_train, X_val, X_test, y_train=None, augment=True):
        """Complete preprocessing and augmentation pipeline."""
        logger.info("Preprocessing training data...")
        X_train = self.preprocessor.preprocess_batch(X_train)
        
        logger.info("Preprocessing validation data...")
        X_val = self.preprocessor.preprocess_batch(X_val)
        
        logger.info("Preprocessing test data...")
        X_test = self.preprocessor.preprocess_batch(X_test)
        
        if augment and y_train is not None:
            logger.info("Augmenting training data...")
            X_train, y_train = self.augmenter.augment_batch(X_train, y_train, augmentation_factor=1)
        
        # NOTE: Pixel-level StandardScaler removed to ensure consistency with inference
        # which uses [0, 1] normalization only.
        
        return X_train, X_val, X_test, y_train


def prepare_image_for_segmentation(image, size=(256, 256)):
    """Prepare single image for U-Net segmentation."""
    preprocessor = ImagePreprocessor(target_size=size)
    return preprocessor.preprocess_single_image(image)


if __name__ == "__main__":
    from data_loader import get_data_splits
    
    data = get_data_splits()
    pipeline = PreprocessingPipeline()
    
    X_train, X_val, X_test, y_train = pipeline.prepare_data(
        data['X_train'],
        data['X_val'],
        data['X_test'],
        data['y_train'],
        augment=True
    )
    
    print(f"Processed training shape: {X_train.shape}")
    print(f"Processed validation shape: {X_val.shape}")
    print(f"Processed test shape: {X_test.shape}")
    print(f"Min value: {X_train.min()}, Max value: {X_train.max()}")
