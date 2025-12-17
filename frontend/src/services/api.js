import axios from 'axios'

const API_BASE = 'http://localhost:5000/api'

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json'
  }
})

export const apiService = {
  predict: async (file) => {
    const formData = new FormData()
    formData.append('file', file)
    const response = await api.post('/predict', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    return response.data
  },

  getPredictionHistory: async () => {
    const response = await api.get('/history')
    return response.data
  },

  getMetrics: async () => {
    const response = await api.get('/metrics')
    return response.data
  },

  getModels: async () => {
    const response = await api.get('/models')
    return response.data
  },

  getConfusionMatrix: async (model) => {
    const response = await api.get(`/confusion-matrix/${model}`)
    return response.data
  },

  getDatasetInsights: async () => {
    const response = await api.get('/dataset-insights')
    return response.data
  }
}

export default api
