#!/usr/bin/env python3
import sys
import json
import os
from inference import BrainTumorInference
import logging

logging.basicConfig(level=logging.ERROR)

def main():
    if len(sys.argv) < 2:
        print(json.dumps({
            "error": "Image path required",
            "usage": "python inference_cli.py <image_path>"
        }))
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(json.dumps({
            "error": f"Image file not found: {image_path}"
        }))
        sys.exit(1)
    
    try:
        model_type = sys.argv[2] if len(sys.argv) > 2 else 'xgboost'
        
        inference = BrainTumorInference(model_type=model_type)
        inference.load_model()
        
        result = inference.predict_single_image(image_path)
        
        if result:
            output = {
                "predicted_class": result['predicted_class'],
                "confidence": result['confidence'],
                "probabilities": {
                    "no_tumor": result['all_probabilities'].get('no_tumor_tumor', result['all_probabilities'].get('no_tumor', 0)),
                    "glioma": result['all_probabilities'].get('glioma_tumor', result['all_probabilities'].get('glioma', 0)),
                    "meningioma": result['all_probabilities'].get('meningioma_tumor', result['all_probabilities'].get('meningioma', 0)),
                    "pituitary": result['all_probabilities'].get('pituitary_tumor', result['all_probabilities'].get('pituitary', 0))
                },
                "model": model_type
            }
            print(json.dumps(output))
        else:
            print(json.dumps({"error": "Prediction failed"}))
            sys.exit(1)
            
    except Exception as e:
        print(json.dumps({
            "error": str(e)
        }))
        sys.exit(1)

if __name__ == "__main__":
    main()
