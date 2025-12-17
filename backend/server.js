import express from 'express'
import cors from 'cors'
import dotenv from 'dotenv'
import mongoose from 'mongoose'
import bodyParser from 'body-parser'
import { fileURLToPath } from 'url'
import path from 'path'

import predictionRoutes from './routes/predictions.js'
import metricRoutes from './routes/metrics.js'
import modelRoutes from './routes/models.js'
import datasetRoutes from './routes/dataset.js'

dotenv.config()

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const app = express()
const PORT = process.env.PORT || 5000
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/brain-tumor'

app.use(cors({
  origin: [
    'http://localhost:5173',
    'http://localhost:3000',
    'http://127.0.0.1:5173'
  ],
  credentials: true
}))

app.use(bodyParser.json({ limit: '50mb' }))
app.use(bodyParser.urlencoded({ limit: '50mb', extended: true }))
app.use(express.json())
app.use(express.static('public'))

mongoose.connect(MONGODB_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true
}).then(() => {
  console.log('✓ MongoDB connected')
}).catch((error) => {
  console.error('MongoDB connection error:', error)
})

app.use('/api', predictionRoutes)
app.use('/api/metrics', metricRoutes)
app.use('/api/models', modelRoutes)
app.use('/api/dataset', datasetRoutes)

app.get('/api/health', (req, res) => {
  res.json({ status: 'Server is running' })
})

app.get('/api/metrics', (req, res) => {
  const metrics = [
    { name: 'XGBoost', accuracy: 92, precision: 90, recall: 91, f1: 91 },
    { name: 'AdaBoost', accuracy: 88, precision: 87, recall: 88, f1: 87 },
    { name: 'Decision Tree', accuracy: 85, precision: 84, recall: 85, f1: 84 },
    { name: 'SVM', accuracy: 89, precision: 88, recall: 89, f1: 88 },
    { name: 'ANN', accuracy: 90, precision: 89, recall: 90, f1: 89 }
  ]
  res.json(metrics)
})

app.get('/api/models', (req, res) => {
  const models = [
    { name: 'xgboost', displayName: 'XGBoost', type: 'Gradient Boosting' },
    { name: 'adaboost', displayName: 'AdaBoost', type: 'Adaptive Boosting' },
    { name: 'decision_tree', displayName: 'Decision Tree', type: 'Tree-based' },
    { name: 'svm', displayName: 'SVM', type: 'Support Vector Machine' },
    { name: 'ann', displayName: 'ANN', type: 'Artificial Neural Network' }
  ]
  res.json(models)
})

app.get('/api/history', async (req, res) => {
  try {
    const Prediction = mongoose.model('Prediction')
    const predictions = await Prediction.find().sort({ createdAt: -1 }).limit(100)
    res.json(predictions)
  } catch (error) {
    res.json([])
  }
})

app.get('/api/dataset-insights', (req, res) => {
  const insights = {
    classDistribution: [
      { name: 'No Tumor', value: 595 },
      { name: 'Glioma', value: 826 },
      { name: 'Meningioma', value: 822 },
      { name: 'Pituitary', value: 627 }
    ],
    dataSplit: [
      { name: 'Training', value: 1722 },
      { name: 'Validation', value: 574 },
      { name: 'Testing', value: 574 }
    ]
  }
  res.json(insights)
})

app.use((err, req, res, next) => {
  console.error('Error:', err)
  res.status(500).json({
    error: 'Internal server error',
    message: err.message
  })
})

app.listen(PORT, () => {
  console.log(`✓ Server running at http://localhost:${PORT}`)
})
