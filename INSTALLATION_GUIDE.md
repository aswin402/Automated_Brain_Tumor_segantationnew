# Installation Guide

## System Requirements

- **OS**: Linux, macOS, or Windows
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **GPU**: Optional (NVIDIA with CUDA for acceleration)

---

## Step 1: Create Virtual Environment

### On Linux/macOS:
```bash
cd /home/aswin/programming/vscode/projects/Automated_Brain_Tumor_segantation
python3 -m venv venv
source venv/bin/activate
```

### On Windows:
```bash
cd C:\path\to\Automated_Brain_Tumor_segantation
python -m venv venv
venv\Scripts\activate.bat
```

---

## Step 2: Upgrade pip and setuptools

```bash
pip install --upgrade pip setuptools wheel
```

---

## Step 3: Install Dependencies

### Option A: Install Core Libraries (Faster)

If TensorFlow/Keras installation is too slow, use minimal requirements:

```bash
pip install -r requirements_minimal.txt
```

This installs:
- numpy, scipy, pandas
- opencv-python
- scikit-learn, matplotlib, seaborn
- xgboost, joblib, tqdm

Then manually install ML libraries:

```bash
pip install tensorflow keras
```

Or without deep learning (classification only):

```bash
# Already included in requirements_minimal.txt
# No need to install anything else
```

### Option B: Install Everything (Complete)

```bash
pip install -r requirements.txt
```

This may take 30+ minutes depending on your system.

---

## Step 4: Verify Installation

Test that all imports work:

```bash
python -c "import numpy, sklearn, xgboost; print('✓ Core libs OK')"
```

For deep learning (optional):

```bash
python -c "import tensorflow as tf; print('✓ TensorFlow OK')"
```

---

## Step 5: Run the Pipeline

### Classification Only (No Deep Learning)

```bash
python main.py
```

### With Segmentation (Requires TensorFlow)

```bash
python main.py --train-segmentation
```

---

## Troubleshooting

### Issue: "Environment is externally managed"

**Solution**: Use virtual environment (venv) as shown in Step 1

### Issue: TensorFlow installation is too slow

**Solution**: Skip it and use classification-only mode:

```bash
# Install minimal requirements
pip install -r requirements_minimal.txt

# Run classification without segmentation
python main.py
```

### Issue: "No module named 'tensorflow'"

**Solution**: Install TensorFlow separately:

```bash
pip install tensorflow>=2.10.0
pip install keras>=2.10.0
```

### Issue: GPU support not working

**Solution**: Install CUDA and cuDNN, then:

```bash
# For NVIDIA GPU
pip install tensorflow[and-cuda]
```

### Issue: PyRadiomics not installing

**Solution**: Skip it (optional for advanced feature extraction):

```bash
# The basic version without advanced radiomics works fine
# Just remove the radiomics section from feature_extraction.py if needed
```

---

## Installation Verification

After installation, verify with:

```bash
python3 << EOF
import sys
print(f"Python: {sys.version}")

try:
    import numpy as np
    print(f"✓ numpy {np.__version__}")
except:
    print("✗ numpy")

try:
    import pandas as pd
    print(f"✓ pandas {pd.__version__}")
except:
    print("✗ pandas")

try:
    import sklearn
    print(f"✓ scikit-learn {sklearn.__version__}")
except:
    print("✗ scikit-learn")

try:
    import xgboost as xgb
    print(f"✓ xgboost {xgb.__version__}")
except:
    print("✗ xgboost")

try:
    import cv2
    print(f"✓ opencv-python {cv2.__version__}")
except:
    print("✗ opencv-python")

try:
    import tensorflow as tf
    print(f"✓ tensorflow {tf.__version__}")
except:
    print("! tensorflow (optional)")

try:
    import keras
    print(f"✓ keras {keras.__version__}")
except:
    print("! keras (optional)")

print("\nInstallation verification complete!")
EOF
```

---

## Quick Commands

### Activate environment
```bash
source venv/bin/activate          # Linux/macOS
venv\Scripts\activate.bat         # Windows
```

### Deactivate environment
```bash
deactivate
```

### Install a package
```bash
pip install package_name
```

### Update a package
```bash
pip install --upgrade package_name
```

### List installed packages
```bash
pip list
```

### Generate requirements from current environment
```bash
pip freeze > requirements.txt
```

---

## Docker Setup (Optional)

If you prefer Docker:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "main.py"]
```

Build and run:

```bash
docker build -t brain-tumor .
docker run -v $(pwd)/Training:/app/Training -v $(pwd)/Testing:/app/Testing brain-tumor
```

---

## Next Steps

After successful installation:

1. **Run the pipeline**: `python main.py`
2. **Check results**: Look in `results/` directory
3. **View logs**: Check `logs/pipeline.log`
4. **Make predictions**: Use `inference.py`

---

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review `logs/pipeline.log` for detailed errors
3. Try installing with `--no-cache-dir`:
   ```bash
   pip install -r requirements.txt --no-cache-dir
   ```
4. Create a fresh venv if corrupted:
   ```bash
   rm -rf venv
   python3 -m venv venv
   source venv/bin/activate
   ```

---

**Last Updated**: December 2024
