# ✅ Setup Verification Checklist

## 1. Prerequisites Check

```bash
# Check Node.js
node --version
# Expected: v16.0.0 or higher

# Check Python
python3 --version
# Expected: Python 3.8+

# Check MongoDB
mongosh
# Expected: Connects to MongoDB
exit
```

## 2. Dependencies Installed

✅ Frontend dependencies: `npm ls` in frontend/
✅ Backend dependencies: `npm ls` in backend/

## 3. Environment Files

Create .env files:

### Backend (.env)
```bash
cd backend
cat > .env << 'BACKEND_ENV'
PORT=5000
MONGODB_URI=mongodb://localhost:27017/brain-tumor
NODE_ENV=development
BACKEND_ENV
```

### Frontend (.env) - Optional
```bash
cd frontend
cat > .env << 'FRONTEND_ENV'
VITE_API_URL=http://localhost:5000/api
FRONTEND_ENV
```

## 4. Start Services in Order

### Terminal 1: Start MongoDB
```bash
# macOS
brew services start mongodb-community

# Linux
sudo systemctl start mongod

# Windows
net start MongoDB

# Verify it's running
mongosh  # Should connect
```

### Terminal 2: Start Backend
```bash
cd backend
npm start

# Expected output:
# ✓ MongoDB connected
# ✓ Server running at http://localhost:5000
```

### Terminal 3: Start Frontend
```bash
cd frontend
npm run dev

# Expected output:
# ✓ ready in 1234 ms
# Local: http://localhost:5173
```

## 5. Test the Dashboard

1. Open browser: http://localhost:5173
2. Should see:
   - Navigation sidebar on left
   - Dashboard page with stats
   - All links working

## 6. Test Prediction API

```bash
# Test health
curl http://localhost:5000/api/health

# Get metrics
curl http://localhost:5000/api/metrics

# Get models
curl http://localhost:5000/api/models

# Get dataset insights
curl http://localhost:5000/api/dataset-insights
```

## 7. Test File Upload

1. Go to: http://localhost:5173/upload
2. Click upload area or drag image
3. Select a JPG/PNG image
4. Click "Analyze MRI"
5. Should see prediction result

## 8. Check All Pages Load

- [ ] Dashboard (/)
- [ ] Upload & Predict (/upload)
- [ ] Model Performance (/performance)
- [ ] Confusion Matrix (/confusion-matrix)
- [ ] Dataset Insights (/dataset)
- [ ] Prediction History (/history)

## 9. Verify MongoDB

```bash
mongosh
use brain-tumor
db.predictions.find()  # Should show predictions after testing
db.metrics.find()      # Should show model metrics
db.models.find()       # Should show trained models
exit
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| MongoDB not connecting | Start MongoDB service |
| Port 5000 in use | Change PORT in backend/.env |
| Port 5173 in use | npm run dev -- --port 5174 |
| Blank page | Hard refresh (Ctrl+Shift+R) |
| No predictions saved | Check MongoDB is running |
| API errors | Check backend console logs |

## Success Indicators

✅ Dashboard opens at http://localhost:5173
✅ All 6 pages load without errors
✅ Can upload image
✅ Get prediction with confidence
✅ Prediction saved to MongoDB
✅ History shows predictions
✅ All charts display data
✅ No browser console errors

---

If all checks pass, your dashboard is ready! 🎉
