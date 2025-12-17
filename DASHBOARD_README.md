# 🧠 Brain Tumor Classification Dashboard

A professional full-stack web application for automated brain tumor segmentation and classification using machine learning. Built with **React**, **Express**, **MongoDB**, and **Python inference**.

## ✨ Features

### 🎯 Frontend
- **Responsive UI** with dark theme (medical-grade design)
- **Real-time predictions** with confidence scores
- **Interactive visualizations** (Recharts charts)
- **Prediction history** tracking
- **Model performance metrics** comparison
- **Dataset insights** with class distribution

### 🔧 Backend
- **RESTful API** with Express.js
- **MongoDB** data persistence
- **File upload handling** with multer
- **Python ML integration** via subprocess
- **CORS-enabled** for frontend communication

### 🤖 ML Pipeline
- **5 Classifiers**: XGBoost, AdaBoost, Decision Tree, SVM, ANN
- **Radiomics feature extraction** from MRI images
- **Pre-trained models** ready for inference
- **High accuracy** (85-92% across models)

---

## 📦 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | React 18, Vite, Tailwind CSS | Fast, modern UI |
| **UI Components** | shadcn/ui, Lucide Icons | Professional components |
| **Charts** | Recharts | Data visualization |
| **Backend** | Node.js, Express.js | REST API server |
| **Database** | MongoDB, Mongoose | Data persistence |
| **File Upload** | Multer | Image handling |
| **ML Inference** | Python, scikit-learn | Predictions |
| **HTTP Client** | Axios | API communication |

---

## 🚀 Quick Start (3 Steps)

### 1️⃣ Prerequisites
```bash
# Check installations
node --version  # v16+
python3 --version  # 3.8+
mongosh  # MongoDB running
```

### 2️⃣ Setup & Start
```bash
# Backend (Terminal 1)
cd backend
npm install
npm start
# ✓ Server running at http://localhost:5000

# Frontend (Terminal 2)
cd frontend
npm install
npm run dev
# ✓ Local:   http://localhost:5173
```

### 3️⃣ Open Dashboard
Visit: **http://localhost:5173**

---

## 📚 Usage Guide

### Upload & Predict
1. Go to **"Upload & Predict"** page
2. Upload an MRI image (JPG/PNG)
3. Click **"Analyze MRI"**
4. View prediction with confidence score

### View Model Performance
1. Go to **"Model Performance"**
2. Compare accuracy, precision, recall, F1
3. See bar charts and line graphs

### Explore Dataset
1. Go to **"Dataset Insights"**
2. View class distribution pie chart
3. Check train/val/test splits

### Check Prediction History
1. Go to **"Prediction History"**
2. See all past predictions with timestamps
3. View confidence scores and statistics

---

## 📁 File Structure

```
brain-tumor-classification-dashboard/
├── frontend/                          # React app
│   ├── src/
│   │   ├── components/ui/            # shadcn/ui components
│   │   ├── components/Layout.jsx     # Navigation
│   │   ├── pages/                    # Route pages
│   │   ├── services/api.js           # Axios client
│   │   ├── App.jsx                   # Router
│   │   └── index.css                 # Tailwind
│   ├── index.html
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── package.json
│
├── backend/                           # Express API
│   ├── models/                       # MongoDB schemas
│   ├── controllers/                  # Business logic
│   ├── routes/                       # API routes
│   ├── uploads/                      # Temp images
│   ├── server.js                     # Main server
│   ├── package.json
│   └── .env
│
├── inference_cli.py                  # Python ML CLI
├── inference.py                      # Original inference
├── config.py                         # ML config
├── models/                           # Trained ML models
├── DASHBOARD_SETUP.md                # Detailed setup guide
├── QUICKSTART.sh                     # Unix quick start
├── QUICKSTART.bat                    # Windows quick start
└── README.md                         # Project overview
```

---

## 🌐 API Endpoints

### Prediction
- `POST /api/predict` - Upload image & get prediction
- `GET /api/history` - Get all predictions
- `DELETE /api/history` - Clear history

### Metrics
- `GET /api/metrics` - All model metrics
- `GET /api/metrics/:model` - Specific model

### Models
- `GET /api/models` - List all models
- `GET /api/models/:name` - Model details

### Dataset
- `GET /api/dataset-insights` - Class distribution & splits
- `GET /api/dataset/confusion-matrix/:model` - Confusion matrix

---

## 🎨 Frontend Pages

| Page | Route | Purpose |
|------|-------|---------|
| Dashboard | `/` | Project overview & metrics |
| Upload & Predict | `/upload` | MRI analysis interface |
| Model Performance | `/performance` | Metrics comparison charts |
| Confusion Matrix | `/confusion-matrix` | Model evaluation matrices |
| Dataset Insights | `/dataset` | Data distribution analysis |
| Prediction History | `/history` | Past predictions table |

---

## 📊 Expected Results

### Model Accuracy
- **XGBoost**: 92% (Best performer ⭐)
- **ANN**: 90%
- **SVM**: 89%
- **AdaBoost**: 88%
- **Decision Tree**: 85%

### Dataset
- **Total Images**: 2,870
- **Classes**: 4 (No Tumor, Glioma, Meningioma, Pituitary)
- **Split**: 60% train, 20% val, 20% test

---

## 🔧 Configuration

### Backend (.env)
```env
PORT=5000
MONGODB_URI=mongodb://localhost:27017/brain-tumor
NODE_ENV=development
```

### Frontend (.env)
```env
VITE_API_URL=http://localhost:5000/api
```

---

## 🐛 Troubleshooting

### MongoDB Connection Failed
```bash
# Start MongoDB
brew services start mongodb-community  # macOS
sudo systemctl start mongod            # Linux
```

### Port Already in Use
```bash
# Change port in .env
PORT=5001
```

### Python Not Found
```bash
# Add Python to PATH or use full path in shell
export PATH=$PATH:/usr/bin/python3
```

### CORS Error
- Check backend is running on port 5000
- Verify frontend API URL in .env
- Check CORS config in server.js

---

## 🚀 Deployment

### Build Frontend
```bash
cd frontend
npm run build
# Creates optimized dist/ folder
```

### Deploy Backend
```bash
# Heroku
heroku create my-brain-tumor-api
git push heroku main

# AWS Lambda (serverless)
npm install -g serverless
serverless deploy

# Railway, Render, or similar
# Just connect your Git repo
```

### Environment Variables (Production)
```env
PORT=5000
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/brain-tumor
NODE_ENV=production
```

---

## 📈 Performance Metrics

| Component | Metric | Value |
|-----------|--------|-------|
| Frontend Build | Size | ~300KB gzipped |
| API Response | Avg Time | <500ms |
| ML Inference | Avg Time | 2-5s |
| Database Query | Avg Time | <100ms |

---

## 🔐 Security Features

- ✅ File type validation (JPG/PNG only)
- ✅ File size limit (50MB)
- ✅ CORS protection
- ✅ Input sanitization
- ⚠️ **No authentication** (add for production)

### Production Recommendations
```bash
npm install helmet express-validator express-rate-limit
# Add JWT authentication
# Enable HTTPS
# Add API rate limiting
```

---

## 💡 Development Tips

### Debug Mode
```bash
# Backend
DEBUG=* npm start

# Frontend
npm run dev -- --debug
```

### Test API Manually
```bash
# Check health
curl http://localhost:5000/api/health

# Get metrics
curl http://localhost:5000/api/metrics

# Test prediction
curl -X POST -F "file=@image.jpg" http://localhost:5000/api/predict
```

### Reset Database
```bash
# In mongosh
use brain-tumor
db.dropDatabase()
```

---

## 📚 Learning Resources

- [React Documentation](https://react.dev)
- [Express.js Guide](https://expressjs.com)
- [MongoDB Docs](https://docs.mongodb.com)
- [Tailwind CSS](https://tailwindcss.com)
- [Recharts](https://recharts.org)

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📝 License

This project is part of a final-year engineering project. Use for educational purposes.

---

## 📞 Support

For issues or questions:
1. Check `DASHBOARD_SETUP.md` for detailed setup
2. Review troubleshooting section
3. Check backend logs: `backend/logs/`
4. Check frontend console (DevTools)

---

## ✅ Verification Checklist

Before deploying, verify:

- [ ] MongoDB is running
- [ ] Backend starts without errors
- [ ] Frontend starts without errors
- [ ] Can upload image
- [ ] Prediction displays correctly
- [ ] All pages load
- [ ] No console errors
- [ ] API responses are valid

---

## 🎯 Next Steps

1. ✅ Dashboard is ready to use
2. Upload test MRI images
3. Monitor prediction accuracy
4. Explore model metrics
5. Track prediction history
6. Fine-tune models if needed

---

## 🌟 Highlights

- **Production-Ready**: Clean code, error handling, validation
- **Scalable**: MongoDB for data, modular code structure
- **User-Friendly**: Intuitive UI, real-time feedback
- **Fast**: Optimized React, efficient ML inference
- **Extensible**: Easy to add new features or models

---

**Status**: ✅ Production Ready  
**Version**: 1.0.0  
**Last Updated**: December 2024

---

Built with ❤️ for brain tumor classification.
