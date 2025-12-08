import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score, classification_report
)
import pandas as pd
import logging
from config import PATHS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class ClassificationMetrics:
    """Calculate classification metrics."""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_proba=None):
        """Calculate all classification metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        if y_proba is not None:
            try:
                if y_proba.shape[1] == 2:
                    metrics['auc_roc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    metrics['auc_roc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', zero_division=0)
            except:
                metrics['auc_roc'] = 0.0
        
        return metrics
    
    @staticmethod
    def get_detailed_metrics(y_true, y_pred, class_names=None):
        """Get detailed classification report."""
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        if class_names:
            detailed_report = {}
            for idx, class_name in enumerate(class_names):
                if str(idx) in report:
                    detailed_report[class_name] = report[str(idx)]
        else:
            detailed_report = report
        
        return detailed_report
    
    @staticmethod
    def confusion_matrix_metrics(y_true, y_pred):
        """Calculate confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        return cm


class Visualizer:
    """Visualization utilities."""
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names, model_name, save_path=None):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
        plt.close()
    
    @staticmethod
    def plot_roc_curve(y_true, y_proba, num_classes, model_name, save_path=None):
        """Plot ROC curve for multi-class classification."""
        if num_classes == 2:
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
        else:
            plt.figure(figsize=(10, 8))
            
            for i in range(num_classes):
                y_true_bin = (y_true == i).astype(int)
                fpr, tpr, _ = roc_curve(y_true_bin, y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
            
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title(f'ROC Curves - {model_name}', fontsize=14, fontweight='bold')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.show()
        plt.close()
    
    @staticmethod
    def plot_metrics_comparison(results_dict, save_path=None):
        """Plot metrics comparison across models."""
        metrics_df = pd.DataFrame(results_dict).T
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Metrics Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        
        for metric, pos in zip(metrics, positions):
            if metric in metrics_df.columns:
                ax = axes[pos]
                metrics_df[metric].plot(kind='bar', ax=ax, color='steelblue')
                ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
                ax.set_ylabel('Score', fontsize=11)
                ax.set_xlabel('Model', fontsize=11)
                ax.set_ylim([0, 1])
                ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Baseline')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics comparison saved to {save_path}")
        
        plt.show()
        plt.close()
    
    @staticmethod
    def plot_loss_curves(history, model_name, save_path=None):
        """Plot training and validation loss curves."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
        if 'val_loss' in history.history:
            axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_title('Loss', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=11)
        axes[0].set_ylabel('Loss', fontsize=11)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        if 'val_accuracy' in history.history:
            axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[1].set_title('Accuracy', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=11)
        axes[1].set_ylabel('Accuracy', fontsize=11)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        fig.suptitle(f'Training Curves - {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Loss curves saved to {save_path}")
        
        plt.show()
        plt.close()
    
    @staticmethod
    def plot_segmentation_masks(images, masks_true, masks_pred, save_path=None, num_samples=5):
        """Plot segmentation results."""
        num_samples = min(num_samples, len(images))
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for idx in range(num_samples):
            axes[idx, 0].imshow(images[idx], cmap='gray')
            axes[idx, 0].set_title('Original Image', fontweight='bold')
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(images[idx], cmap='gray')
            axes[idx, 1].imshow(masks_true[idx], cmap='Reds', alpha=0.5)
            axes[idx, 1].set_title('Ground Truth Mask', fontweight='bold')
            axes[idx, 1].axis('off')
            
            axes[idx, 2].imshow(images[idx], cmap='gray')
            axes[idx, 2].imshow(masks_pred[idx], cmap='Blues', alpha=0.5)
            axes[idx, 2].set_title('Predicted Mask', fontweight='bold')
            axes[idx, 2].axis('off')
        
        plt.suptitle('Segmentation Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Segmentation masks saved to {save_path}")
        
        plt.show()
        plt.close()


class SegmentationMetrics:
    """Calculate segmentation metrics."""
    
    @staticmethod
    def dice_score(y_true, y_pred, smooth=1e-6):
        """Calculate Dice Score."""
        y_pred = (y_pred > 0.5).astype(int)
        
        intersection = np.sum(y_true * y_pred)
        union = np.sum(y_true) + np.sum(y_pred)
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice
    
    @staticmethod
    def iou_score(y_true, y_pred, smooth=1e-6):
        """Calculate IoU (Intersection over Union) Score."""
        y_pred = (y_pred > 0.5).astype(int)
        
        intersection = np.sum(y_true * y_pred)
        union = np.sum(np.maximum(y_true, y_pred))
        
        iou = (intersection + smooth) / (union + smooth)
        return iou
    
    @staticmethod
    def hausdorff_distance(y_true, y_pred):
        """Calculate Hausdorff Distance."""
        from scipy.spatial.distance import directed_hausdorff
        
        y_pred = (y_pred > 0.5).astype(int)
        
        true_points = np.argwhere(y_true)
        pred_points = np.argwhere(y_pred)
        
        if len(true_points) == 0 or len(pred_points) == 0:
            return float('inf')
        
        dist1 = directed_hausdorff(true_points, pred_points)[0]
        dist2 = directed_hausdorff(pred_points, true_points)[0]
        
        return max(dist1, dist2)
    
    @staticmethod
    def calculate_segmentation_metrics(y_true, y_pred):
        """Calculate all segmentation metrics."""
        metrics = {
            'dice_score': SegmentationMetrics.dice_score(y_true, y_pred),
            'iou_score': SegmentationMetrics.iou_score(y_true, y_pred),
        }
        
        return metrics


def generate_results_report(results_dict, class_names, save_path=None):
    """Generate comprehensive results report."""
    report_lines = [
        "="*80,
        "BRAIN TUMOR SEGMENTATION AND CLASSIFICATION RESULTS",
        "="*80,
        "",
    ]
    
    for model_name, metrics in results_dict.items():
        report_lines.append(f"\n{model_name}")
        report_lines.append("-" * 40)
        
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                report_lines.append(f"  {metric_name}: {value:.4f}")
            else:
                report_lines.append(f"  {metric_name}: {value}")
    
    report_lines.append("\n" + "="*80)
    
    report_text = "\n".join(report_lines)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        logger.info(f"Results report saved to {save_path}")
    
    print(report_text)
    return report_text


if __name__ == "__main__":
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 1, 3, 0, 1, 3, 3])
    y_proba = np.array([
        [0.9, 0.05, 0.03, 0.02],
        [0.1, 0.8, 0.05, 0.05],
        [0.15, 0.7, 0.1, 0.05],
        [0.05, 0.05, 0.1, 0.8],
        [0.85, 0.1, 0.03, 0.02],
        [0.2, 0.7, 0.05, 0.05],
        [0.1, 0.15, 0.7, 0.05],
        [0.05, 0.05, 0.1, 0.8],
    ])
    
    metrics = ClassificationMetrics.calculate_metrics(y_true, y_pred, y_proba)
    print("Metrics:", metrics)
    
    cm = ClassificationMetrics.confusion_matrix_metrics(y_true, y_pred)
    print("Confusion Matrix:\n", cm)
