import { useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card'
import { Button } from '../components/ui/Button'
import { apiService } from '../services/api'
import { Upload as UploadIcon, CheckCircle, AlertCircle } from 'lucide-react'

export function Upload() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0]
    if (selectedFile) {
      setFile(selectedFile)
      const reader = new FileReader()
      reader.onload = (event) => {
        setPreview(event.target.result)
      }
      reader.readAsDataURL(selectedFile)
      setError(null)
      setResult(null)
    }
  }

  const handlePredict = async () => {
    if (!file) {
      setError('Please select an MRI image')
      return
    }

    setLoading(true)
    setError(null)
    try {
      const prediction = await apiService.predict(file)
      setResult(prediction)
    } catch (err) {
      setError(err.response?.data?.error || 'Prediction failed. Please try again.')
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.85) return 'text-green-400'
    if (confidence >= 0.70) return 'text-yellow-400'
    return 'text-red-400'
  }

  const classColors = {
    'no_tumor': '#ef4444',
    'glioma': '#f97316',
    'meningioma': '#eab308',
    'pituitary': '#06b6d4'
  }

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-white mb-2">MRI Upload & Analysis</h1>
        <p className="text-slate-400">Upload a brain MRI image to get a tumor classification</p>
      </div>

      <div className="grid grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Upload MRI Image</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="border-2 border-dashed border-slate-600 rounded-lg p-8 text-center hover:border-blue-500 transition-colors cursor-pointer" 
              onClick={() => document.getElementById('file-input').click()}>
              <UploadIcon className="mx-auto mb-4 text-slate-400" size={40} />
              <p className="text-slate-300 font-medium">Click to upload or drag and drop</p>
              <p className="text-slate-500 text-sm mt-1">JPG, PNG (256x256 recommended)</p>
              <input
                id="file-input"
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="hidden"
              />
            </div>

            {preview && (
              <div className="space-y-3">
                <p className="text-sm text-slate-400">Selected File: {file?.name}</p>
                <img src={preview} alt="Preview" className="w-full rounded-lg border border-slate-600" />
              </div>
            )}

            <Button 
              onClick={handlePredict} 
              disabled={!file || loading}
              className="w-full"
              size="lg"
            >
              {loading ? 'Analyzing...' : 'Analyze MRI'}
            </Button>

            {error && (
              <div className="bg-red-500/20 border border-red-500 rounded-lg p-4 flex gap-3">
                <AlertCircle className="text-red-400 flex-shrink-0" size={20} />
                <p className="text-red-200 text-sm">{error}</p>
              </div>
            )}
          </CardContent>
        </Card>

        {result && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <CheckCircle className="text-green-400" size={24} />
                Prediction Result
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="bg-slate-700/50 rounded-lg p-4">
                <p className="text-slate-400 text-sm mb-2">Predicted Tumor Class</p>
                <p className="text-3xl font-bold text-white capitalize">
                  {result.predicted_class.replace('_', ' ')}
                </p>
              </div>

              <div className="bg-slate-700/50 rounded-lg p-4">
                <p className="text-slate-400 text-sm mb-2">Confidence Score</p>
                <div className="flex items-center gap-3">
                  <p className={`text-3xl font-bold ${getConfidenceColor(result.confidence)}`}>
                    {(result.confidence * 100).toFixed(1)}%
                  </p>
                  <div className="flex-1 bg-slate-600 rounded-full h-2">
                    <div 
                      className="bg-blue-500 h-2 rounded-full"
                      style={{ width: `${result.confidence * 100}%` }}
                    />
                  </div>
                </div>
              </div>

              <div className="space-y-3">
                <p className="text-slate-400 text-sm">Class Probabilities</p>
                {Object.entries(result.probabilities || {}).map(([className, prob]) => (
                  <div key={className} className="space-y-1">
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-300 capitalize">{className.replace('_', ' ')}</span>
                      <span className="text-slate-400">{(prob * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-slate-600 rounded-full h-1.5">
                      <div 
                        className="h-1.5 rounded-full transition-all"
                        style={{ 
                          width: `${prob * 100}%`,
                          backgroundColor: classColors[className] || '#3b82f6'
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
