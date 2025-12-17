# Brain Tumor Classification Dashboard - Full Stack Setup Guide

## 🎯 Overview

This dashboard provides a complete web interface for the Brain Tumor Classification ML pipeline. It includes:
- **Frontend**: React + Vite + Tailwind CSS + Recharts
- **Backend**: Node.js + Express + MongoDB
- **ML Integration**: Python inference pipeline

## 📋 Prerequisites

### System Requirements
- **Node.js**: v16+ 
- **Python**: 3.8+
- **MongoDB**: Local instance or MongoDB Atlas connection
- **Git**: For version control

### Python Dependencies
The ML models are already trained. Ensure dependencies are installed:

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start (5 Minutes)

### Option 1: Local MongoDB (Recommended for Development)

#### 1. Start MongoDB (if not running)
```bash
# On macOS (with Homebrew)
brew services start mongodb-community

# On Linux
sudo systemctl start mongod

# On Windows
# Open Services > MongoDB and start, or run:
# mongod --config "C:\Program Files\MongoDB\Server\5.0\etc\mongod.conf"
```

Verify MongoDB is running:
```bash
mongosh  # or: mongo
```

#### 2. Setup Backend

```bash
# Navigate to backend
cd backend

# Install dependencies
npm install

# Create .env file
cp .env.example .env
# Edit .env if needed (defaults work for local MongoDB)

# Start backend server
npm start
# Expected: "✓ Server running at http://localhost:5000"
```

#### 3. Setup Frontend

```bash
# In a new terminal, navigate to frontend
cd frontend

# Install dependencies
npm install

# Create .env file (optional, defaults to localhost:5000)
cp .env.example .env

# Start development server
npm run dev
# Expected: "✓ ready in 1234 ms" and visit http://localhost:5173
```

#### 4. Open Dashboard
Visit: **http://localhost:5173**

---

## 📦 Project Structure

```
Automated_Brain_Tumor_segmentation/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ui/              # shadcn/ui components
│   │   │   ├── Layout.jsx       # Navigation & sidebar
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx    # Overview page
│   │   │   ├── Upload.jsx       # MRI upload & prediction
│   │   │   ├── Performance.jsx  # Model metrics
│   │   │   ├── ConfusionMatrix.jsx # Confusion matrices
│   │   │   ├── Dataset.jsx      # Dataset insights
│   │   │   └── History.jsx      # Prediction history
│   │   ├── services/
│   │   │   └── api.js           # Axios API client
│   │   ├── App.jsx              # Router & app setup
│   │   ├── main.jsx             # Entry point
│   │   └── index.css            # Tailwind directives
│   ├── index.html
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   └── package.json
│
├── backend/
│   ├── models/
│   │   ├── Prediction.js        # Prediction schema
│   │   ├── Metric.js            # Metrics schema
│   │   └── Model.js             # Model info schema
│   ├── controllers/
│   │   ├── predictionController.js    # Predict & history
│   │   ├── metricController.js        # Metrics API
│   │   ├── modelController.js         # Models API
│   │   └── datasetController.js       # Dataset insights
│   ├── routes/
│   │   ├── predictions.js       # POST /predict
│   │   ├── metrics.js           # GET /metrics
│   │   ├── models.js            # GET /models
│   │   └── dataset.js           # GET /dataset-insights
│   ├── uploads/                 # Temporary image storage
│   ├── server.js                # Express server setup
│   ├── package.json
│   ├── .env.example
│   └── .gitignore
│
├── inference_cli.py             # Python CLI for ML inference
├── inference.py                 # Original inference engine
├── config.py                    # ML configuration
├── models/                      # Trained model files (.pkl)
└── results/                     # Confusion matrix & ROC images
```

---

## 🔌 API Endpoints

### Prediction
- **POST** `/api/predict`
  - Upload MRI image
  - Response: `{ predicted_class, confidence, probabilities, model }`

- **GET** `/api/history`
  - Get all past predictions
  - Response: Array of predictions

- **DELETE** `/api/history`
  - Clear all prediction history

### Metrics
- **GET** `/api/metrics`
  - Get all model metrics
  - Response: Array of metrics for each model

- **POST** `/api/metrics/initialize`
  - Initialize default metrics in DB

- **GET** `/api/metrics/:model`
  - Get specific model metrics

### Models
- **GET** `/api/models`
  - Get list of all trained models

- **POST** `/api/models/initialize`
  - Initialize models in DB

- **GET** `/api/models/:name`
  - Get specific model info

### Dataset
- **GET** `/api/dataset-insights`
  - Get class distribution & data splits

- **GET** `/api/dataset/confusion-matrix/:model`
  - Get confusion matrix for model

---

## 🎨 Frontend Pages

### 1. Dashboard (/)
- Project overview
- Best model accuracy badge
- Total MRI images count
- Tumor classes
- Models used (5 classifiers)

### 2. MRI Upload & Prediction (/upload)
- Drag-and-drop image upload
- Image preview
- "Analyze MRI" button
- Prediction result with:
  - Tumor class
  - Confidence percentage
  - Class probabilities
  - Progress bar visualization

### 3. Model Performance (/performance)
- Accuracy comparison (bar chart)
- Precision vs Recall (line chart)
- F1-Score comparison (bar chart)
- Metrics summary table

### 4. Confusion Matrix (/confusion-matrix)
- Model selector dropdown
- Confusion matrix image display
- ROC curve visualization
- Model performance stats

### 5. Dataset Insights (/dataset)
- Class distribution (pie chart)
- Train/Val/Test split (bar chart)
- Dataset statistics
- Class details table

### 6. Prediction History (/history)
- Table of all predictions
  - Image name
  - Predicted class
  - Confidence score
  - Date & time
- Statistics summary
- High/low confidence counts

---

## 🛠️ Development & Troubleshooting

### Backend Issues

#### MongoDB Connection Error
```
Error: connect ECONNREFUSED 127.0.0.1:27017
```

**Solution**: Start MongoDB or use MongoDB Atlas
```bash
# Local MongoDB
brew services start mongodb-community  # macOS
sudo systemctl start mongod            # Linux

# MongoDB Atlas - Update .env
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/brain-tumor
```

#### Python Not Found
```
Error: spawn python ENOENT
```

**Solution**: Add Python to PATH or use full path
```bash
# Windows
set PATH=%PATH%;C:\Python39
python --version

# macOS/Linux
which python3
```

#### PORT Already in Use
```
Error: EADDRINUSE :::5000
```

**Solution**: Use different port or kill existing process
```bash
# Linux/macOS
lsof -i :5000
kill -9 <PID>

# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Or change port in .env
PORT=5001
```

### Frontend Issues

#### Vite Port Conflict
```bash
# Use different port
npm run dev -- --port 5174
```

#### API Connection Error
- Check backend is running on `http://localhost:5000`
- Verify `.env` has correct API URL
- Check CORS is enabled in backend

#### Image Preview Not Showing
- Ensure image file < 50MB
- Try JPG or PNG format
- Check browser console for errors

---

## 📊 Features in Detail

### Image Upload Process
1. User selects/drags MRI image
2. Frontend previews image
3. User clicks "Analyze MRI"
4. Backend receives image via multipart/form-data
5. Backend calls Python inference: `python inference_cli.py <image_path>`
6. Python script outputs JSON: `{ predicted_class, confidence, probabilities }`
7. Backend saves result to MongoDB
8. Frontend displays prediction with confidence visualization

### Data Persistence
- **MongoDB collections**:
  - `predictions`: Stores all inference results
  - `metrics`: Model performance metrics
  - `models`: Model metadata

- **File Storage**: Uploaded images in `backend/uploads/` (auto-deleted after processing)

---

## 🚀 Deployment

### Prepare for Production

#### 1. Build Frontend
```bash
cd frontend
npm run build
# Creates optimized build in dist/
```

#### 2. Serve Frontend
```bash
# Option A: Static hosting (Vercel, Netlify)
npm run build
# Deploy dist/ folder

# Option B: From backend
cp -r frontend/dist backend/public
# Backend serves static files
```

#### 3. Environment Variables (Production)
```bash
# backend/.env
PORT=5000
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/brain-tumor
NODE_ENV=production

# frontend/.env.production
VITE_API_URL=https://api.yourdomain.com
```

#### 4. Deploy Backend
```bash
# Heroku example
heroku create my-brain-tumor-api
git push heroku main

# AWS example
npm install -g serverless
serverless deploy
```

---

## 📈 Expected Results

### Model Performance (on test set)
| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| XGBoost | 92% | 90% | 91% | 91% |
| AdaBoost | 88% | 87% | 88% | 87% |
| Decision Tree | 85% | 84% | 85% | 84% |
| SVM | 89% | 88% | 89% | 88% |
| ANN | 90% | 89% | 90% | 89% |

### Dataset Distribution
- **Total Images**: 2,870
- **Training**: 1,722 (60%)
- **Validation**: 574 (20%)
- **Testing**: 574 (20%)

### Tumor Classes
- No Tumor: 595 images
- Glioma: 826 images
- Meningioma: 822 images
- Pituitary: 627 images

---

## 🔐 Security Considerations

- ✅ File upload validation (JPG/PNG only, 50MB limit)
- ✅ CORS configured for localhost only
- ✅ Input sanitization in backend
- ❌ No authentication implemented (development only)

### For Production, Add:
```bash
npm install express-validator helmet dotenv
# Implement JWT authentication
# Add input validation middleware
# Enable HTTPS
# Add rate limiting
```

---

## 📞 Support & Debugging

### Check Logs
```bash
# Backend
# Logs in: backend/logs/

# Frontend
# Browser DevTools > Console (F12)
```

### Test API Manually
```bash
# Test backend health
curl http://localhost:5000/api/health

# Get metrics
curl http://localhost:5000/api/metrics

# Get models
curl http://localhost:5000/api/models

# Test prediction (with image)
curl -X POST -F "file=@image.jpg" http://localhost:5000/api/predict
```

### Reset Database
```bash
# Drop all data
db.dropDatabase()  # in mongosh

# Re-initialize
curl -X POST http://localhost:5000/api/metrics/initialize
curl -X POST http://localhost:5000/api/models/initialize
```

---

## 🎓 Learning Resources

- **React Docs**: https://react.dev
- **Express Guide**: https://expressjs.com
- **MongoDB**: https://docs.mongodb.com
- **Tailwind CSS**: https://tailwindcss.com
- **Recharts**: https://recharts.org

---

## 📝 Common Commands

```bash
# Frontend
cd frontend
npm install          # Install dependencies
npm run dev          # Start dev server (port 5173)
npm run build        # Build for production

# Backend
cd backend
npm install          # Install dependencies
npm start            # Start server (port 5000)
npm run dev          # Start with nodemon

# Python
python inference_cli.py image.jpg              # Test inference
python cpu_pipeline.py                         # Train models
python main.py                                 # Full pipeline
```

---

## ✅ Verification Checklist

- [ ] Python dependencies installed: `pip install -r requirements.txt`
- [ ] Node.js v16+: `node --version`
- [ ] MongoDB running: `mongosh` connects successfully
- [ ] Backend `.env` configured
- [ ] Frontend `.env` configured (optional)
- [ ] Backend starts: `npm start` in backend/
- [ ] Frontend starts: `npm run dev` in frontend/
- [ ] Dashboard opens: http://localhost:5173
- [ ] Can upload image and get prediction
- [ ] All pages load and display data

---

## 🎉 You're All Set!

Your full-stack Brain Tumor Classification Dashboard is ready to use. 

**Next Steps:**
1. Open http://localhost:5173
2. Upload an MRI image
3. View prediction results
4. Explore model performance metrics
5. Check prediction history

For issues or questions, check the troubleshooting section above or review the project structure.

Happy classifying! 🧠

---

**Last Updated**: December 2024
**Status**: Production Ready
**Version**: 1.0.0
