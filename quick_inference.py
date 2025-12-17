#!/usr/bin/env python3
"""
Quick tumor detection on single images or folders
Usage:
  python3 quick_inference.py image.jpg          # Single image
  python3 quick_inference.py /path/to/folder/   # Batch from folder
"""

import sys
import os
from inference import BrainTumorInference

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 quick_inference.py <image_path_or_folder>")
        print("\nExamples:")
        print("  python3 quick_inference.py scan.jpg")
        print("  python3 quick_inference.py ~/my_images/")
        sys.exit(1)
    
    path = sys.argv[1].strip()
    
    # Load model
    print("🔄 Loading XGBoost model...")
    inference = BrainTumorInference(model_type='xgboost')
    inference.load_model()
    print("✓ Model loaded\n")
    
    # Check if file or directory
    if os.path.isfile(path):
        # Single image
        print(f"🔍 Analyzing: {os.path.basename(path)}\n")
        result = inference.predict_single_image(path)
        
        if result:
            print("="*60)
            print("TUMOR DETECTION RESULT")
            print("="*60)
            print(f"Image: {os.path.basename(result['image_path'])}")
            print(f"\n🎯 Predicted Class: {result['predicted_class'].upper()}")
            print(f"✓ Confidence: {result['confidence']:.1%}\n")
            print("Class Probabilities:")
            for class_name, prob in result['all_probabilities'].items():
                bar = '▓' * int(prob * 40)
                print(f"  {class_name:20s}: {prob:6.2%} {bar}")
            print("="*60)
    
    elif os.path.isdir(path):
        # Batch prediction
        print(f"📁 Scanning folder: {path}\n")
        results = inference.predict_from_directory(path)
        
        if results:
            print("="*70)
            print("BATCH TUMOR DETECTION RESULTS")
            print("="*70)
            
            for i, result in enumerate(results, 1):
                tumor_class = result['predicted_class']
                confidence = result['confidence']
                filename = os.path.basename(result['image_path'])
                
                # Color indicator
                if confidence > 0.8:
                    indicator = "🔴"  # High confidence
                elif confidence > 0.6:
                    indicator = "🟡"  # Medium confidence
                else:
                    indicator = "🟢"  # Low confidence
                
                print(f"\n{i}. {filename}")
                print(f"   {indicator} Detected: {tumor_class}")
                print(f"   ✓ Confidence: {confidence:.1%}")
            
            print("\n" + "="*70)
            print(f"Total: {len(results)} images analyzed")
            print("="*70)
        else:
            print("❌ No images found in directory")
    
    else:
        print(f"❌ Path not found: {path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
