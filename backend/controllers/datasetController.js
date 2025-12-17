export const getDatasetInsights = async (req, res) => {
  try {
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
      ],
      totalImages: 2870,
      numClasses: 4,
      imageSize: [256, 256],
      trainingPercentage: 60,
      validationPercentage: 20,
      testingPercentage: 20
    }

    res.json(insights)
  } catch (error) {
    console.error('Dataset insights error:', error)
    res.status(500).json({ error: 'Failed to fetch dataset insights' })
  }
}

export const getConfusionMatrix = async (req, res) => {
  try {
    const { model } = req.params
    
    const confusionMatrices = {
      xgboost: { model: 'xgboost', accuracy: 0.92 },
      adaboost: { model: 'adaboost', accuracy: 0.88 },
      decision_tree: { model: 'decision_tree', accuracy: 0.85 },
      svm: { model: 'svm', accuracy: 0.89 },
      ann: { model: 'ann', accuracy: 0.90 }
    }

    const matrix = confusionMatrices[model] || confusionMatrices.xgboost
    res.json(matrix)
  } catch (error) {
    console.error('Confusion matrix error:', error)
    res.status(500).json({ error: 'Failed to fetch confusion matrix' })
  }
}
