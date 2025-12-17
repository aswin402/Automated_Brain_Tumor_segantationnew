# 🚀 RUN DASHBOARD - Quick Commands

## Prerequisites Check
```bash
node --version        # Should be v16+
python3 --version    # Should be 3.8+
mongosh              # Should connect to MongoDB
```

---

## 🎯 Option 1: Automated Setup (Recommended)

### macOS / Linux
```bash
chmod +x QUICKSTART.sh
./QUICKSTART.sh
```

### Windows
```cmd
QUICKSTART.bat
```

Then follow the instructions in the output.

---

## 🎯 Option 2: Manual Setup

### Terminal 1 - MongoDB
```bash
# Make sure MongoDB is running (check if already running)
mongosh  # If this connects, MongoDB is ready
```

### Terminal 2 - Backend
```bash
cd backend
npm install
npm start
# ✓ Server running at http://localhost:5000
```

### Terminal 3 - Frontend
```bash
cd frontend
npm install
npm run dev
# ✓ Local: http://localhost:5173
```

### Step 4 - Open Dashboard
```
http://localhost:5173
```

---

## 🔌 Verify Everything Works

### Test Backend Health
```bash
curl http://localhost:5000/api/health
# Response: {"status":"Server is running"}
```

### Test API Endpoints
```bash
# Get all metrics
curl http://localhost:5000/api/metrics

# Get all models
curl http://localhost:5000/api/models

# Get dataset insights
curl http://localhost:5000/api/dataset-insights
```

### Test ML Inference
```bash
python3 inference_cli.py path/to/image.jpg
# Response: JSON with prediction
```

---

## 🎨 Access Dashboard Pages

- **Dashboard**: http://localhost:5173/
- **Upload & Predict**: http://localhost:5173/upload
- **Model Performance**: http://localhost:5173/performance
- **Confusion Matrix**: http://localhost:5173/confusion-matrix
- **Dataset Insights**: http://localhost:5173/dataset
- **Prediction History**: http://localhost:5173/history

---

## 🔧 Troubleshooting Quick Fixes

### MongoDB Not Running
```bash
# macOS
brew services start mongodb-community

# Linux
sudo systemctl start mongod

# Windows - Run as Administrator
net start MongoDB
```

### Port 5000 Already in Use
```bash
# Option 1: Use different port
cd backend
PORT=5001 npm start

# Option 2: Kill process on port 5000 (Linux/macOS)
lsof -i :5000
kill -9 <PID>

# Option 2: Kill process on port 5000 (Windows)
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

### Port 5173 Already in Use
```bash
cd frontend
npm run dev -- --port 5174
```

### Python Script Not Found
```bash
# Make sure you're in root directory
cd /path/to/Automated_Brain_Tumor_segantation
python3 inference_cli.py image.jpg
```

### CORS Error in Frontend
- Verify backend is running on localhost:5000
- Check browser console (F12) for actual error
- Ensure .env has correct API URL
- Restart frontend: Ctrl+C then npm run dev

---

## 📊 Test Workflow

1. **Upload Image**
   - Go to http://localhost:5173/upload
   - Click upload area or drag image
   - Click "Analyze MRI"

2. **View Prediction**
   - See predicted class
   - See confidence score
   - See all class probabilities

3. **Check History**
   - Go to http://localhost:5173/history
   - See your uploaded image
   - Verify confidence in table

4. **View Metrics**
   - Go to http://localhost:5173/performance
   - Compare all models
   - See charts and stats

---

## 🧹 Clean Up

### Clear Prediction History
```bash
curl -X DELETE http://localhost:5000/api/history
```

### Remove Uploaded Images
```bash
rm -rf backend/uploads/*
```

### Reset Database
```bash
mongosh
use brain-tumor
db.dropDatabase()
exit
```

---

## 📈 Performance Tips

### For Faster Inference
```bash
# Use Decision Tree (fastest)
python3 inference_cli.py image.jpg decision_tree

# Use XGBoost (best accuracy)
python3 inference_cli.py image.jpg xgboost
```

### For Development
```bash
# Frontend hot reload already enabled
# Backend watch mode
npm run dev --prefix backend
```

---

## 🛑 Stop Everything

### Stop Frontend
```
Terminal 3: Ctrl+C
```

### Stop Backend
```
Terminal 2: Ctrl+C
```

### Stop MongoDB
```bash
# macOS
brew services stop mongodb-community

# Linux
sudo systemctl stop mongod

# Windows
net stop MongoDB
```

---

## 📝 Logs

### Backend Logs
```bash
tail -f backend/logs/server.log
```

### Frontend Logs
```
Browser DevTools: F12 > Console
```

### Python Logs
```
Check stdout of backend terminal
```

---

## 🚀 Next Steps After First Run

1. ✅ Upload a test MRI image
2. ✅ Verify prediction works
3. ✅ Explore all dashboard pages
4. ✅ Check model performance
5. ✅ Review prediction history
6. ✅ Read DASHBOARD_SETUP.md for detailed info
7. ✅ Deploy to production (see setup guide)

---

## 💾 Database Commands

### View Database
```bash
mongosh
show databases
use brain-tumor
show collections
db.predictions.find()
db.metrics.find()
db.models.find()
```

### Count Predictions
```bash
mongosh
use brain-tumor
db.predictions.countDocuments()
```

### Export Data
```bash
mongoexport --db brain-tumor --collection predictions --out predictions.json
```

---

## 🎯 One-Line Start (After First Setup)

### Backend
```bash
cd backend && npm start
```

### Frontend (New Terminal)
```bash
cd frontend && npm run dev
```

### That's it!
```
http://localhost:5173
```

---

## 🆘 Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| "Cannot find module" | `npm install` in that directory |
| "Port already in use" | Change PORT in .env or kill process |
| "MongoDB connection error" | Start MongoDB service |
| "API not responding" | Check backend is running on port 5000 |
| "Image upload fails" | Check image < 50MB, format is JPG/PNG |
| "Blank page" | Hard refresh (Ctrl+Shift+R), check console |

---

## 📞 Debug Mode

### Enable Verbose Logging
```bash
# Backend
DEBUG=* npm start

# Frontend
npm run dev -- --debug
```

### Check Network Requests
```
Browser DevTools > Network tab (F12)
Monitor API calls to http://localhost:5000/api/*
```

---

## ✅ Success Checklist

- [ ] MongoDB running (mongosh works)
- [ ] Backend started (npm start in backend/)
- [ ] Frontend started (npm run dev in frontend/)
- [ ] Dashboard opens (http://localhost:5173)
- [ ] Can upload image
- [ ] Prediction displays
- [ ] No console errors
- [ ] All 6 pages load

---

**Ready to go!** 🎉

Visit http://localhost:5173 to start classifying brain tumors.
