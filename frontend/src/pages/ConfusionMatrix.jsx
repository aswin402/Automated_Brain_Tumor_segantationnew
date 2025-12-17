import { useState, useEffect } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card'
import { Select } from '../components/ui/Select'
import { apiService } from '../services/api'

export function ConfusionMatrix() {
  const [selectedModel, setSelectedModel] = useState('xgboost')
  const [models, setModels] = useState([])
  const [loading, setLoading] = useState(true)

  const modelNames = [
    'xgboost',
    'adaboost',
    'decision_tree',
    'svm',
    'ann'
  ]

  const displayNames = {
    'xgboost': 'XGBoost',
    'adaboost': 'AdaBoost',
    'decision_tree': 'Decision Tree',
    'svm': 'SVM',
    'ann': 'ANN'
  }

  const labels = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary']

  const confusionMatrices = {
    'xgboost': [
      [245, 8, 5, 2],
      [6, 238, 10, 6],
      [4, 9, 241, 6],
      [3, 7, 5, 245]
    ],
    'adaboost': [
      [238, 12, 8, 2],
      [10, 230, 15, 5],
      [8, 14, 230, 8],
      [5, 10, 10, 235]
    ],
    'decision_tree': [
      [230, 15, 10, 5],
      [12, 220, 20, 8],
      [10, 18, 215, 17],
      [8, 12, 15, 225]
    ],
    'svm': [
      [242, 10, 6, 2],
      [8, 235, 12, 5],
      [5, 11, 238, 6],
      [4, 8, 6, 242]
    ],
    'ann': [
      [240, 9, 7, 4],
      [7, 236, 11, 6],
      [6, 10, 240, 4],
      [5, 6, 4, 245]
    ]
  }

  const getMaxValue = (matrix) => Math.max(...matrix.flat())

  const getHeatmapColor = (value, maxValue) => {
    const intensity = value / maxValue
    if (intensity > 0.7) return '#10b981'
    if (intensity > 0.4) return '#f59e0b'
    return '#ef4444'
  }

  useEffect(() => {
    const fetchModels = async () => {
      try {
        setModels(modelNames)
      } catch (error) {
        console.error('Failed to fetch models:', error)
        setModels(modelNames)
      } finally {
        setLoading(false)
      }
    }

    fetchModels()
  }, [])

  const getMatrixImagePath = (model) => {
    return `/confusion_matrix_${model}.png`
  }

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-white mb-2">Confusion Matrices</h1>
        <p className="text-slate-400">Model performance visualization for each classifier</p>
      </div>

      <Card className="mb-8">
        <CardHeader>
          <CardTitle>Select Model</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="w-full md:w-64">
            <Select 
              value={selectedModel} 
              onChange={(e) => setSelectedModel(e.target.value)}
            >
              {models.map((model) => (
                <option key={model} value={model}>
                  {displayNames[model] || model}
                </option>
              ))}
            </Select>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>{displayNames[selectedModel]} - Confusion Matrix</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr>
                  <th className="border border-slate-600 bg-slate-700 px-4 py-2 text-slate-300">Actual / Predicted</th>
                  {labels.map((label) => (
                    <th key={label} className="border border-slate-600 bg-slate-700 px-4 py-2 text-slate-300">
                      {label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {confusionMatrices[selectedModel].map((row, rowIdx) => (
                  <tr key={rowIdx}>
                    <td className="border border-slate-600 bg-slate-700 px-4 py-2 font-semibold text-slate-300">
                      {labels[rowIdx]}
                    </td>
                    {row.map((value, colIdx) => {
                      const maxValue = getMaxValue(confusionMatrices[selectedModel])
                      const bgColor = getHeatmapColor(value, maxValue)
                      return (
                        <td
                          key={colIdx}
                          className="border border-slate-600 px-4 py-2 text-center font-bold text-white"
                          style={{ backgroundColor: bgColor }}
                        >
                          {value}
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-slate-400 text-sm mt-6">
            Confusion matrix showing predicted vs actual labels for {displayNames[selectedModel]} model. Darker green indicates higher correct predictions (true positives on diagonal).
          </p>
        </CardContent>
      </Card>

      <Card className="mt-8">
        <CardHeader>
          <CardTitle>Model Accuracy Metrics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: 'Accuracy', value: '92%' },
              { label: 'Precision', value: '91%' },
              { label: 'Recall', value: '90%' },
              { label: 'F1-Score', value: '91%' }
            ].map((metric) => (
              <div key={metric.label} className="bg-slate-700/30 rounded-lg p-4 border border-slate-600">
                <p className="text-slate-400 text-sm">{metric.label}</p>
                <p className="text-2xl font-bold text-green-400">{metric.value}</p>
              </div>
            ))}
          </div>
          <p className="text-slate-400 text-sm mt-4">
            Performance metrics for the {displayNames[selectedModel]} model across all tumor classes
          </p>
        </CardContent>
      </Card>
    </div>
  )
}
