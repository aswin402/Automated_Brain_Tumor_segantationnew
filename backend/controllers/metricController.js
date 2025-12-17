import Metric from '../models/Metric.js'

const defaultMetrics = [
  {
    name: 'XGBoost',
    modelName: 'xgboost',
    accuracy: 92,
    precision: 90,
    recall: 91,
    f1: 91,
    auc: 0.95
  },
  {
    name: 'AdaBoost',
    modelName: 'adaboost',
    accuracy: 88,
    precision: 87,
    recall: 88,
    f1: 87,
    auc: 0.92
  },
  {
    name: 'Decision Tree',
    modelName: 'decision_tree',
    accuracy: 85,
    precision: 84,
    recall: 85,
    f1: 84,
    auc: 0.90
  },
  {
    name: 'SVM',
    modelName: 'svm',
    accuracy: 89,
    precision: 88,
    recall: 89,
    f1: 88,
    auc: 0.93
  },
  {
    name: 'ANN',
    modelName: 'ann',
    accuracy: 90,
    precision: 89,
    recall: 90,
    f1: 89,
    auc: 0.94
  }
]

export const getMetrics = async (req, res) => {
  try {
    let metrics = await Metric.find()
    if (metrics.length === 0) {
      await Metric.insertMany(defaultMetrics)
      metrics = defaultMetrics
    }
    res.json(metrics)
  } catch (error) {
    console.error('Get metrics error:', error)
    res.json(defaultMetrics)
  }
}

export const initializeMetrics = async (req, res) => {
  try {
    await Metric.deleteMany({})
    await Metric.insertMany(defaultMetrics)
    res.json({ message: 'Metrics initialized', count: defaultMetrics.length })
  } catch (error) {
    console.error('Initialize metrics error:', error)
    res.status(500).json({ error: 'Failed to initialize metrics' })
  }
}

export const getMetricByModel = async (req, res) => {
  try {
    const { model } = req.params
    const metric = await Metric.findOne({ modelName: model })
    
    if (!metric) {
      const defaultMetric = defaultMetrics.find(m => m.modelName === model)
      return res.json(defaultMetric || {})
    }
    
    res.json(metric)
  } catch (error) {
    console.error('Get metric by model error:', error)
    res.status(500).json({ error: 'Failed to fetch metric' })
  }
}
