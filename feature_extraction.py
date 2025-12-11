import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import label
import logging
from skimage import feature as skimage_feature
from config import RADIOMICS_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RadiomicsFeatureExtractor:
    """Extract radiomics features from medical images."""
    
    def __init__(self, bin_width=25):
        self.bin_width = bin_width
        
    def extract_shape_features(self, image, mask=None):
        """Extract shape-based features."""
        if mask is None:
            mask = (image > image.mean()).astype(int)
        
        features = {}
        
        labeled, num_features = label(mask)
        features['num_regions'] = num_features
        
        if num_features > 0:
            region_sizes = np.bincount(labeled.flat)[1:]
            features['region_size_mean'] = np.mean(region_sizes)
            features['region_size_std'] = np.std(region_sizes)
            features['region_size_max'] = np.max(region_sizes)
            features['region_size_min'] = np.min(region_sizes)
        else:
            features['region_size_mean'] = 0
            features['region_size_std'] = 0
            features['region_size_max'] = 0
            features['region_size_min'] = 0
        
        tumor_pixels = np.sum(mask)
        image_area = mask.shape[0] * mask.shape[1]
        features['tumor_area_ratio'] = tumor_pixels / (image_area + 1e-6)
        features['tumor_area'] = tumor_pixels
        
        contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        
        if len(contours) > 0:
            cnt = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(cnt, True)
            features['perimeter'] = perimeter
            features['circularity'] = (4 * np.pi * tumor_pixels) / ((perimeter ** 2) + 1e-6)
            
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            features['solidity'] = tumor_pixels / (hull_area + 1e-6)
        else:
            features['perimeter'] = 0
            features['circularity'] = 0
            features['solidity'] = 0
        
        return features
    
    def extract_intensity_features(self, image, mask=None):
        """Extract intensity-based features."""
        if mask is None:
            mask = (image > image.mean()).astype(int)
        
        masked_image = image[mask > 0]
        
        features = {}
        if len(masked_image) > 0:
            features['intensity_mean'] = np.mean(masked_image)
            features['intensity_std'] = np.std(masked_image)
            features['intensity_max'] = np.max(masked_image)
            features['intensity_min'] = np.min(masked_image)
            features['intensity_median'] = np.median(masked_image)
            features['intensity_range'] = features['intensity_max'] - features['intensity_min']
            features['intensity_iqr'] = np.percentile(masked_image, 75) - np.percentile(masked_image, 25)
            features['intensity_energy'] = np.sum(masked_image ** 2)
            features['intensity_entropy'] = self._calculate_entropy(masked_image)
            
            features['intensity_skewness'] = self._calculate_skewness(masked_image)
            features['intensity_kurtosis'] = self._calculate_kurtosis(masked_image)
            features['intensity_variance'] = np.var(masked_image)
            features['intensity_sum'] = np.sum(masked_image)
            features['intensity_q1'] = np.percentile(masked_image, 25)
            features['intensity_q3'] = np.percentile(masked_image, 75)
            features['intensity_rmse'] = np.sqrt(np.mean(masked_image ** 2))
        else:
            features['intensity_mean'] = 0
            features['intensity_std'] = 0
            features['intensity_max'] = 0
            features['intensity_min'] = 0
            features['intensity_median'] = 0
            features['intensity_range'] = 0
            features['intensity_iqr'] = 0
            features['intensity_energy'] = 0
            features['intensity_entropy'] = 0
            features['intensity_skewness'] = 0
            features['intensity_kurtosis'] = 0
            features['intensity_variance'] = 0
            features['intensity_sum'] = 0
            features['intensity_q1'] = 0
            features['intensity_q3'] = 0
            features['intensity_rmse'] = 0
        
        return features
    
    def extract_texture_features(self, image, mask=None):
        """Extract texture-based features using GLCM and LBP."""
        if mask is None:
            mask = (image > image.mean()).astype(int)
        
        masked_image = image * mask
        features = {}
        
        try:
            glcm = self._calculate_glcm(masked_image, mask)
            glcm_features = self._glcm_features(glcm)
            features.update(glcm_features)
        except:
            logger.warning("GLCM calculation failed, using zeros")
            features.update({
                'contrast': 0, 'dissimilarity': 0, 'homogeneity': 0,
                'asm': 0, 'energy': 0, 'correlation': 0
            })
        
        try:
            lbp_features = self._extract_lbp_features(masked_image, mask)
            features.update(lbp_features)
        except:
            logger.warning("LBP calculation failed, using zeros")
            features.update({
                'lbp_mean': 0, 'lbp_std': 0, 'lbp_hist_entropy': 0
            })
        
        return features
    
    def _calculate_glcm(self, image, mask, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
        """Calculate Gray-Level Co-occurrence Matrix."""
        image_normalized = ((image - image.min()) / (image.max() - image.min() + 1e-6) * 255).astype(np.uint8)
        
        glcm = np.zeros((256, 256))
        masked_image = image_normalized * mask
        
        for distance in distances:
            for angle in angles:
                dx = int(distance * np.cos(angle))
                dy = int(distance * np.sin(angle))
                
                for i in range(masked_image.shape[0] - abs(dx)):
                    for j in range(masked_image.shape[1] - abs(dy)):
                        if mask[i, j] > 0 and mask[i + dx, j + dy] > 0:
                            val1 = masked_image[i, j]
                            val2 = masked_image[i + dx, j + dy]
                            glcm[val1, val2] += 1
        
        glcm = glcm / (glcm.sum() + 1e-6)
        return glcm
    
    def _glcm_features(self, glcm):
        """Calculate GLCM-based texture features."""
        features = {}
        
        i, j = np.meshgrid(np.arange(glcm.shape[0]), np.arange(glcm.shape[1]), indexing='ij')
        
        mean_i = np.sum(i * glcm)
        mean_j = np.sum(j * glcm)
        
        std_i = np.sqrt(np.sum((i - mean_i) ** 2 * glcm))
        std_j = np.sqrt(np.sum((j - mean_j) ** 2 * glcm))
        
        features['contrast'] = np.sum((i - j) ** 2 * glcm)
        features['dissimilarity'] = np.sum(np.abs(i - j) * glcm)
        features['homogeneity'] = np.sum(glcm / (1 + np.abs(i - j)))
        features['asm'] = np.sum(glcm ** 2)
        features['energy'] = np.sqrt(features['asm'])
        
        if std_i > 0 and std_j > 0:
            features['correlation'] = np.sum((i - mean_i) * (j - mean_j) * glcm) / (std_i * std_j)
        else:
            features['correlation'] = 0
        
        return features
    
    def _extract_lbp_features(self, image, mask, P=8, R=1):
        """Extract Local Binary Pattern features."""
        if image.dtype != np.uint8:
            image = ((image - image.min()) / (image.max() - image.min() + 1e-6) * 255).astype(np.uint8)
        
        lbp = skimage_feature.local_binary_pattern(image, P, R, method='uniform')
        lbp_masked = lbp[mask > 0]
        
        features = {}
        features['lbp_mean'] = np.mean(lbp_masked) if len(lbp_masked) > 0 else 0
        features['lbp_std'] = np.std(lbp_masked) if len(lbp_masked) > 0 else 0
        
        hist, _ = np.histogram(lbp_masked, bins=P + 2, range=(0, P + 2))
        hist = hist / (len(lbp_masked) + 1e-6)
        features['lbp_hist_entropy'] = -np.sum(hist * np.log(hist + 1e-6))
        
        return features
    
    def _calculate_entropy(self, data, bins=256):
        """Calculate entropy of data."""
        hist, _ = np.histogram(data, bins=bins, range=(data.min(), data.max() + 1e-6))
        hist = hist / len(data)
        entropy = -np.sum(hist * np.log(hist + 1e-6))
        return entropy
    
    def _calculate_skewness(self, data):
        """Calculate skewness of data."""
        if len(data) < 2:
            return 0
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-6:
            return 0
        skewness = np.mean(((data - mean) / std) ** 3)
        return skewness
    
    def _calculate_kurtosis(self, data):
        """Calculate kurtosis of data."""
        if len(data) < 2:
            return 0
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-6:
            return 0
        kurtosis = np.mean(((data - mean) / std) ** 4) - 3
        return kurtosis
    
    def extract_all_features(self, image, mask=None):
        """Extract all radiomics features from image."""
        features = {}
        
        shape_features = self.extract_shape_features(image, mask)
        features.update({f'shape_{k}': v for k, v in shape_features.items()})
        
        intensity_features = self.extract_intensity_features(image, mask)
        features.update({f'intensity_{k}': v for k, v in intensity_features.items()})
        
        texture_features = self.extract_texture_features(image, mask)
        features.update({f'texture_{k}': v for k, v in texture_features.items()})
        
        return features


def extract_features_batch(images, masks=None, feature_names=None):
    """Extract features from a batch of images."""
    extractor = RadiomicsFeatureExtractor(bin_width=RADIOMICS_CONFIG['bin_width'])
    
    all_features = []
    
    for idx, image in enumerate(images):
        mask = None if masks is None else masks[idx]
        features = extractor.extract_all_features(image, mask)
        all_features.append(features)
    
    df = pd.DataFrame(all_features)
    
    logger.info(f"Extracted {len(df.columns)} features from {len(df)} images")
    
    return df, extractor


def save_features_to_csv(features_df, filepath):
    """Save extracted features to CSV."""
    features_df.to_csv(filepath, index=False)
    logger.info(f"Features saved to {filepath}")


def load_features_from_csv(filepath):
    """Load features from CSV."""
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
    return df


if __name__ == "__main__":
    from data_loader import get_data_splits
    from preprocessing import PreprocessingPipeline
    
    data = get_data_splits()
    pipeline = PreprocessingPipeline()
    
    X_train, X_val, X_test, y_train = pipeline.prepare_data(
        data['X_train'],
        data['X_val'],
        data['X_test'],
        data['y_train'],
        augment=False
    )
    
    logger.info("Extracting features from test set...")
    features_df, extractor = extract_features_batch(X_test[:10])
    
    print(f"Features shape: {features_df.shape}")
    print(f"Features columns: {list(features_df.columns)}")
    print(features_df.head())
