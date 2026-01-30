#!/usr/bin/env python3
import sys
import os
from inference import BrainTumorInference

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 run_ensemble_inference.py <image_path>")
        sys.exit(1)
    
    path = sys.argv[1].strip()
    
    print("🔄 Loading Weighted Ensemble (XGBoost + ANN + SVM)...")
    inference = BrainTumorInference(model_type='ensemble')
    
    ensemble_path = os.path.join('models', 'ensemble_model.pkl')
    if not os.path.exists(ensemble_path):
        print(f"❌ Ensemble model not found at {ensemble_path}")
        print("Please run: python3 train_ensemble_from_csv.py")
        sys.exit(1)
        
    inference.load_model(ensemble_path)
    
    if os.path.isfile(path):
        result = inference.predict_single_image(path)
        if result:
            print("\n" + "="*60)
            print("ENSEMBLE DETECTION RESULT")
            print("="*60)
            print(f"Image: {os.path.basename(result['image_path'])}")
            print(f"🎯 Predicted Class: {result['predicted_class'].upper()}")
            print(f"✓ Confidence: {result['confidence']:.1%}")
            print("\nClass Probabilities:")
            for class_name, prob in result['all_probabilities'].items():
                bar = '▓' * int(prob * 40)
                print(f"  {class_name:20s}: {prob:6.2%} {bar}")
            print("="*60)

if __name__ == "__main__":
    main()
