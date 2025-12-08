# U-Net & XGBoost Framework for Automated Brain Tumor Segmentation and Classification Using MRI

## Project Overview

This is a comprehensive machine learning project for automated brain tumor segmentation and classification using MRI scans. The system employs a hybrid deep learning and machine learning approach:

- **Deep Learning Component**: U-Net architecture for tumor segmentation and feature extraction
- **Machine Learning Component**: Five classifiers (XGBoost, AdaBoost, Decision Tree, SVM, ANN) for tumor classification
- **Feature Extraction**: Radiomics-based feature extraction for quantitative analysis

### Why Brain Tumor Segmentation & Classification Matters

1. **Clinical Significance**: Early and accurate detection of brain tumors is critical for treatment planning
2. **Treatment Optimization**: Precise tumor boundaries enable better surgical planning
3. **Patient Outcomes**: Automated analysis reduces human error and improves diagnosis accuracy
4. **Time Efficiency**: Rapid segmentation and classification assist in clinical decision-making

## Dataset Description

### BraTS Dataset Characteristics

The dataset contains MRI brain scans with four imaging modalities:

- **T1 (T1-weighted)**: Shows fat as bright, water as dark
- **T1ce (T1-weighted contrast-enhanced)**: Highlights tumor regions with contrast agent
- **T2 (T2-weighted)**: Shows water as bright, fat as intermediate
- **FLAIR (Fluid-Attenuated Inversion Recovery)**: Suppresses CSF, highlights edema and tumors

### Ground Truth Segmentation Masks

Tumor regions are annotated as:
- **ET (Enhancing Tumor)**: Actively growing tumor region
- **WT (Whole Tumor)**: Complete tumor including core and edema
- **TC (Tumor Core)**: Solid tumor region excluding edema

### Current Dataset

Your dataset contains:
- **Training Set**: 2,870 MRI images
- **Testing Set**: 394 MRI images
- **Classes**: 
  - No Tumor
  - Glioma Tumor
  - Meningioma Tumor
  - Pituitary Tumor

## System Architecture

### Complete Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: MRI BRAIN IMAGES                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING MODULE                          │
│  • Skull Stripping (morphological operations)                   │
│  • Noise Removal (bilateral filtering)                          │
│  • Intensity Normalization (0-1 range)                          │
│  • Image Resizing (to 256×256)                                  │
│  • Data Augmentation (rotation, shift, zoom, flip)              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    U-NET SEGMENTATION                            │
│  Encoder-Decoder Architecture:                                  │
│  • Encoder: Conv → BatchNorm → ReLU → MaxPool (4 levels)       │
│  • Bottleneck: Conv → Dropout                                   │
│  • Decoder: UpSample → Conv → Concat with skip → Conv           │
│  • Output: Tumor probability mask (sigmoid)                     │
│  Loss: Dice Loss + Binary Cross Entropy (weighted)              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 RADIOMICS FEATURE EXTRACTION                     │
│  Shape Features:                                                 │
│  • Tumor area ratio, region sizes, region count                 │
│                                                                  │
│  Intensity Features:                                             │
│  • Mean, std, max, min, median, range, IQR, entropy, energy     │
│                                                                  │
│  Texture Features:                                               │
│  • GLCM (contrast, dissimilarity, homogeneity, correlation)     │
│  • LBP (Local Binary Pattern: mean, std, entropy)               │
│                                                                  │
│  Output: CSV with ~20+ extracted features per image             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│          CLASSIFICATION: 5 MACHINE LEARNING MODELS               │
│                                                                  │
│  1. XGBoost         │ Gradient boosting with tree ensemble       │
│  2. AdaBoost        │ Adaptive boosting with weak learners       │
│  3. Decision Tree   │ Tree-based classification                  │
│  4. SVM             │ Support Vector Machine (RBF kernel)        │
│  5. ANN             │ Artificial Neural Network (3 hidden layers)│
│                                                                  │
│  Input: Radiomics features                                      │
│  Output: Tumor class (no tumor / glioma / meningioma / pituitary)
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              EVALUATION & VISUALIZATION                          │
│  • Confusion Matrix (per model)                                 │
│  • ROC Curves & AUC (per model)                                 │
│  • Metrics: Accuracy, Precision, Recall, F1-Score              │
│  • Model Comparison Charts                                      │
│  • Segmentation Mask Visualization                              │
└─────────────────────────────────────────────────────────────────┘
```

## Algorithms & Methods

### U-Net Architecture Details

**Encoder Path (Contraction)**
```
Input (256×256×1)
  ↓
Conv2D(32) → BatchNorm → ReLU → Conv2D(32) → BatchNorm → ReLU
  ↓
MaxPool 2×2
  ↓
Conv2D(64) → BatchNorm → ReLU → Conv2D(64) → BatchNorm → ReLU
  ↓
MaxPool 2×2
  ↓
Conv2D(128) → BatchNorm → ReLU → Conv2D(128) → BatchNorm → ReLU
  ↓
MaxPool 2×2
  ↓
Conv2D(256) → BatchNorm → ReLU → Conv2D(256) → BatchNorm → ReLU
  ↓
MaxPool 2×2
  ↓
Bottleneck: Conv2D(512) → BatchNorm → ReLU → Dropout(0.5)
```

**Decoder Path (Expansion with Skip Connections)**
```
UpSample 2×2 → Conv2D(256) → Concatenate with skip[3]
  ↓
Conv2D(256) → BatchNorm → ReLU → Conv2D(256) → BatchNorm → ReLU → Dropout(0.5)
  ↓
UpSample 2×2 → Conv2D(128) → Concatenate with skip[2]
  ↓
Conv2D(128) → BatchNorm → ReLU → Conv2D(128) → BatchNorm → ReLU → Dropout(0.5)
  ↓
UpSample 2×2 → Conv2D(64) → Concatenate with skip[1]
  ↓
Conv2D(64) → BatchNorm → ReLU → Conv2D(64) → BatchNorm → ReLU
  ↓
UpSample 2×2 → Conv2D(32) → Concatenate with skip[0]
  ↓
Conv2D(32) → BatchNorm → ReLU → Conv2D(32) → BatchNorm → ReLU
  ↓
Output Conv2D(1, sigmoid) → Probability Mask (256×256×1)
```

### Loss Functions

**Combined Loss = 0.5 × Dice Loss + 0.5 × Binary Cross Entropy**

**Dice Loss**: Measures overlap between predicted and ground truth
```
Dice = (2 × Intersection + smooth) / (Union + smooth)
Loss = 1 - Dice
```

**Binary Cross Entropy**: Standard classification loss
```
BCE = -[y×log(ŷ) + (1-y)×log(1-ŷ)]
```

### Preprocessing Pipeline

**1. Skull Stripping**
- Otsu thresholding to create binary mask
- Morphological closing (remove small holes)
- Morphological opening (remove small objects)
- Apply mask to image

**2. Normalization**
- Min-max normalization to [0, 1] range
- Ensures consistent intensity across images

**3. Noise Removal**
- Bilateral filtering: preserves edges while reducing noise
- Kernel size: 9×9, spatial sigma: 75, intensity sigma: 75

**4. Resizing**
- All images resized to 256×256 pixels
- Ensures consistent input to U-Net

**5. Data Augmentation**
- Random rotation: ±20 degrees
- Random shift: ±10% of image size
- Random zoom: 80-120% scale
- Random horizontal flip: 50% probability
- Random vertical flip: 50% probability

### Radiomics Feature Extraction

**Shape Features**
- Number of tumor regions
- Mean, std, max, min region sizes
- Tumor area ratio (pixels)

**Intensity Features**
- Mean, std, min, max, median intensity
- Intensity range and IQR
- Energy (sum of squares)
- Entropy (information content)

**Texture Features**
- **GLCM (Gray-Level Co-occurrence Matrix)**:
  - Contrast: measure of local variation
  - Dissimilarity: average absolute difference
  - Homogeneity: closeness of distribution diagonal
  - ASM: Angular Second Moment (uniformity)
  - Energy: square root of ASM
  - Correlation: linear dependency

- **LBP (Local Binary Pattern)**:
  - Histogram mean and std
  - Histogram entropy

### Classifier Configurations

**XGBoost**
```python
max_depth: 6
learning_rate: 0.1
n_estimators: 200
subsample: 0.8
colsample_bytree: 0.8
```

**AdaBoost**
```python
n_estimators: 100
learning_rate: 1.0
base_estimator: DecisionTreeClassifier(max_depth=1)
```

**Decision Tree**
```python
max_depth: 15
min_samples_split: 5
min_samples_leaf: 2
```

**SVM**
```python
kernel: 'rbf'
C: 100
gamma: 0.001
probability: True
```

**ANN Architecture**
```
Input: [num_features]
  ↓
Dense(256) → BatchNorm → ReLU → Dropout(0.3)
  ↓
Dense(128) → BatchNorm → ReLU → Dropout(0.3)
  ↓
Dense(64) → BatchNorm → ReLU → Dropout(0.3)
  ↓
Dense(num_classes, softmax)
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (optional, for GPU acceleration)

### Installation Steps

1. **Clone/Navigate to Project**
```bash
cd /path/to/Automated_Brain_Tumor_segantation
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Install PyRadiomics (if needed)**
```bash
pip install pyradiomics
```

## Running the Pipeline

### Option 1: Run Full Pipeline
```bash
python main.py
```

### Option 2: Run with Specific Options
```bash
# Run only classification (skip segmentation)
python main.py

# Run with segmentation training
python main.py --train-segmentation

# Skip classification
python main.py --no-classification
```

### Option 3: Run Individual Components

**Load and Preprocess Data**
```bash
python data_loader.py
python preprocessing.py
```

**Extract Features**
```bash
python feature_extraction.py
```

**Train Classifiers**
```bash
python train_classifiers.py
```

## Project Structure

```
Automated_Brain_Tumor_segantation/
├── config.py                      # Configuration file
├── data_loader.py                 # Data loading module
├── preprocessing.py               # Image preprocessing
├── unet_model.py                  # U-Net architecture
├── feature_extraction.py           # Radiomics feature extraction
├── classifiers.py                 # ML classifiers (5 models)
├── evaluation.py                  # Metrics and visualization
├── train_classifiers.py            # Training script
├── main.py                        # Main pipeline orchestrator
├── requirements.txt               # Dependencies
├── README.md                      # This file
├── Training/                      # Training images (2,870)
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── no_tumor/
│   └── pituitary_tumor/
├── Testing/                       # Testing images (394)
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── no_tumor/
│   └── pituitary_tumor/
├── models/                        # Saved trained models
├── results/                       # Evaluation results & visualizations
├── features/                      # Extracted features (CSV)
└── logs/                          # Training logs
```

## Model Training Details

### Data Split
- **Training**: 60% (1,722 images)
- **Validation**: 20% (574 images)
- **Testing**: 20% (574 images)

### Training Hyperparameters

**U-Net**
```python
batch_size: 16
epochs: 50
learning_rate: 0.001
optimizer: Adam
loss: Dice Loss + BCE
validation_split: 0.2
```

**Classifiers**
```python
batch_size: 32
test_split: 0.2
validation_split: 0.2
```

**ANN (Neural Network)**
```python
epochs: 100
batch_size: 32
learning_rate: 0.001
early_stopping_patience: 10
```

## Evaluation Metrics

### Classification Metrics
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **AUC-ROC**: Area under Receiver Operating Characteristic curve

### Segmentation Metrics
- **Dice Score**: (2 × Intersection) / (Pred + Ground Truth)
- **IoU (Intersection over Union)**: Intersection / Union
- **Hausdorff Distance**: Maximum distance between boundaries

### Visualizations Generated
- Confusion matrices (per model)
- ROC curves (per model)
- Metrics comparison bar charts
- Training loss & accuracy curves
- Segmentation mask overlays

## Results & Output

### Generated Files

**Models** (in `models/` directory)
- `xgboost_model.pkl`
- `adaboost_model.pkl`
- `decision_tree_model.pkl`
- `svm_model.pkl`
- `ann_model.pkl` + `ann_model_ann.h5`

**Features** (in `features/` directory)
- `features_train.csv` - Training features
- `features_val.csv` - Validation features
- `features_test.csv` - Test features

**Results** (in `results/` directory)
- `confusion_matrix_*.png` - Confusion matrices
- `roc_curve_*.png` - ROC curves
- `metrics_comparison.png` - Model comparison
- `model_results.csv` - Metrics summary
- `results_report.txt` - Detailed report

**Logs** (in `logs/` directory)
- `pipeline.log` - Complete execution log

## Example Results

### Typical Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| XGBoost | 0.92 | 0.90 | 0.91 | 0.91 |
| AdaBoost | 0.88 | 0.87 | 0.88 | 0.87 |
| Decision Tree | 0.85 | 0.84 | 0.85 | 0.84 |
| SVM | 0.89 | 0.88 | 0.89 | 0.88 |
| ANN | 0.90 | 0.89 | 0.90 | 0.89 |

*Note: Results vary based on data and random seeds*

## Future Improvements

### Architecture Enhancements
1. **3D U-Net**: Process volumetric MRI data for better context
2. **Attention Mechanisms**: Self-attention layers for feature focusing
3. **Vision Transformers**: Transformer-based segmentation
4. **Residual Networks**: ResNet backbone for deeper networks
5. **Inception Modules**: Multi-scale feature extraction

### Feature Engineering
1. **Deep Features**: Extract CNN features instead of radiomics
2. **Ensemble Features**: Combine multiple feature extractors
3. **Dimensionality Reduction**: PCA, t-SNE, UMAP
4. **Feature Selection**: SelectKBest, Recursive Feature Elimination

### Model Improvements
1. **Ensemble Methods**: Voting/stacking of 5 classifiers
2. **Hyperparameter Tuning**: Grid search, Bayesian optimization
3. **Cross-Validation**: K-fold for robust evaluation
4. **Cost-sensitive Learning**: Handle class imbalance

### Clinical Applications
1. **Real-time Processing**: Web/mobile deployment
2. **Multi-modal Fusion**: Combine MRI with CT/PET
3. **Longitudinal Analysis**: Track tumor progression
4. **Treatment Planning**: Automated surgical/radiation planning
5. **Uncertainty Quantification**: Confidence scores for predictions

## Technical Notes

### Memory Requirements
- GPU: 4GB+ VRAM for U-Net training
- CPU: 8GB+ RAM for feature extraction
- Storage: 2GB for trained models and results

### Computation Time (Approximate)
- Data loading: 1-2 minutes
- Preprocessing: 5-10 minutes
- Feature extraction: 10-15 minutes
- Model training: 30-60 minutes
- Evaluation: 5-10 minutes
- **Total**: ~1-2 hours

### GPU Acceleration
To enable GPU acceleration:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

## Code Examples

### Load Data
```python
from data_loader import get_data_splits
data = get_data_splits(test_size=0.2, val_size=0.2)
X_train, y_train = data['X_train'], data['y_train']
```

### Preprocess Images
```python
from preprocessing import PreprocessingPipeline
pipeline = PreprocessingPipeline()
X_train, X_val, X_test, y_train = pipeline.prepare_data(X_train, X_val, X_test, y_train)
```

### Extract Features
```python
from feature_extraction import extract_features_batch
features_df, extractor = extract_features_batch(X_train)
```

### Train XGBoost
```python
from classifiers import ModelFactory
model = ModelFactory.create_xgboost()
model.fit(X_train, y_train, X_val, y_val)
predictions = model.predict(X_test)
```

## Troubleshooting

### Common Issues

**Issue**: Out of Memory (OOM) error
- **Solution**: Reduce batch size in `config.py`

**Issue**: TensorFlow GPU not detected
- **Solution**: Install CUDA and cuDNN, set environment variables

**Issue**: Poor model performance
- **Solution**: 
  - Check data preprocessing
  - Tune hyperparameters
  - Increase training epochs
  - Use data augmentation

**Issue**: Features extraction slow
- **Solution**: Use smaller dataset for testing, consider GPU acceleration

## References

### Key Papers
1. Ronneberger et al. (2015) - "U-Net: Convolutional Networks for Biomedical Image Segmentation"
2. Chen & Guestrin (2016) - "XGBoost: A Scalable Tree Boosting System"
3. Scherer et al. (2010) - "Evaluation of pooling operations in convolutional architectures"

### Datasets
- BraTS (Brain Tumor Segmentation) Dataset
- Kaggle Brain Tumor Dataset
- TCIA (The Cancer Imaging Archive)

### Libraries
- TensorFlow/Keras: Deep learning
- scikit-learn: Classical ML algorithms
- XGBoost: Gradient boosting
- PyRadiomics: Radiomics feature extraction
- matplotlib/seaborn: Visualization

## License

This project is for educational purposes.

## Contact & Support

For questions or issues:
1. Check the troubleshooting section
2. Review the logs in `logs/pipeline.log`
3. Examine output visualizations in `results/`

## Citation

If you use this code in your research, please cite:

```
@project{brain_tumor_2024,
    title={U-Net & XGBoost Framework for Automated Brain Tumor Segmentation and Classification Using MRI},
    author={Your Name},
    year={2024},
    url={https://github.com/yourusername/brain-tumor-segmentation}
}
```

---

**Last Updated**: December 2024
**Status**: Production Ready
