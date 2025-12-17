import Model from '../models/Model.js'

const defaultModels = [
  {
    name: 'xgboost',
    displayName: 'XGBoost',
    type: 'Gradient Boosting',
    accuracy: 92,
    precision: 90,
    recall: 91,
    f1: 91,
    status: 'active'
  },
  {
    name: 'adaboost',
    displayName: 'AdaBoost',
    type: 'Adaptive Boosting',
    accuracy: 88,
    precision: 87,
    recall: 88,
    f1: 87,
    status: 'active'
  },
  {
    name: 'decision_tree',
    displayName: 'Decision Tree',
    type: 'Tree-based',
    accuracy: 85,
    precision: 84,
    recall: 85,
    f1: 84,
    status: 'active'
  },
  {
    name: 'svm',
    displayName: 'SVM',
    type: 'Support Vector Machine',
    accuracy: 89,
    precision: 88,
    recall: 89,
    f1: 88,
    status: 'active'
  },
  {
    name: 'ann',
    displayName: 'ANN',
    type: 'Artificial Neural Network',
    accuracy: 90,
    precision: 89,
    recall: 90,
    f1: 89,
    status: 'active'
  }
]

export const getModels = async (req, res) => {
  try {
    let models = await Model.find()
    if (models.length === 0) {
      await Model.insertMany(defaultModels)
      models = defaultModels
    }
    res.json(models)
  } catch (error) {
    console.error('Get models error:', error)
    res.json(defaultModels)
  }
}

export const initializeModels = async (req, res) => {
  try {
    await Model.deleteMany({})
    await Model.insertMany(defaultModels)
    res.json({ message: 'Models initialized', count: defaultModels.length })
  } catch (error) {
    console.error('Initialize models error:', error)
    res.status(500).json({ error: 'Failed to initialize models' })
  }
}

export const getModelByName = async (req, res) => {
  try {
    const { name } = req.params
    const model = await Model.findOne({ name })
    
    if (!model) {
      const defaultModel = defaultModels.find(m => m.name === name)
      return res.json(defaultModel || {})
    }
    
    res.json(model)
  } catch (error) {
    console.error('Get model by name error:', error)
    res.status(500).json({ error: 'Failed to fetch model' })
  }
}
