import { useEffect, useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card'
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from '../components/ui/Table'
import { apiService } from '../services/api'
import { Loader2 } from 'lucide-react'

export function History() {
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const data = await apiService.getPredictionHistory()
        setHistory(Array.isArray(data) ? data : [])
      } catch (error) {
        console.error('Failed to fetch history:', error)
        setHistory([])
      } finally {
        setLoading(false)
      }
    }

    fetchHistory()
  }, [])

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.85) return 'text-green-400'
    if (confidence >= 0.70) return 'text-yellow-400'
    return 'text-red-400'
  }

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A'
    try {
      return new Date(dateString).toLocaleString()
    } catch {
      return dateString
    }
  }

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-white mb-2">Prediction History</h1>
        <p className="text-slate-400">All uploaded MRI images and their predictions</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Recent Predictions</CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="animate-spin text-blue-400" size={32} />
              <p className="ml-4 text-slate-400">Loading history...</p>
            </div>
          ) : history.length === 0 ? (
            <div className="text-center py-8">
              <p className="text-slate-400">No predictions yet. Start by uploading an MRI image!</p>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Image Name</TableHead>
                  <TableHead>Predicted Class</TableHead>
                  <TableHead>Confidence</TableHead>
                  <TableHead>Date & Time</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {history.map((item, index) => (
                  <TableRow key={item._id || index}>
                    <TableCell className="font-medium">
                      {item.filename || item.imageName || 'Unknown'}
                    </TableCell>
                    <TableCell className="capitalize">
                      {(item.predictedClass || item.predicted_class || 'N/A').replace('_', ' ')}
                    </TableCell>
                    <TableCell>
                      <span className={`font-semibold ${getConfidenceColor(item.confidence || 0)}`}>
                        {((item.confidence || 0) * 100).toFixed(1)}%
                      </span>
                    </TableCell>
                    <TableCell className="text-slate-400">
                      {formatDate(item.createdAt || item.timestamp)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      {history.length > 0 && (
        <Card className="mt-8">
          <CardHeader>
            <CardTitle>Statistics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-4 gap-4">
              <div className="bg-slate-700/30 rounded-lg p-4">
                <p className="text-slate-400 text-sm mb-2">Total Predictions</p>
                <p className="text-2xl font-bold text-white">{history.length}</p>
              </div>
              <div className="bg-slate-700/30 rounded-lg p-4">
                <p className="text-slate-400 text-sm mb-2">Average Confidence</p>
                <p className="text-2xl font-bold text-blue-400">
                  {(history.reduce((sum, h) => sum + (h.confidence || 0), 0) / history.length * 100).toFixed(1)}%
                </p>
              </div>
              <div className="bg-slate-700/30 rounded-lg p-4">
                <p className="text-slate-400 text-sm mb-2">High Confidence (>85%)</p>
                <p className="text-2xl font-bold text-green-400">
                  {history.filter(h => (h.confidence || 0) > 0.85).length}
                </p>
              </div>
              <div className="bg-slate-700/30 rounded-lg p-4">
                <p className="text-slate-400 text-sm mb-2">Low Confidence ({'<'}70%)</p>
                <p className="text-2xl font-bold text-red-400">
                  {history.filter(h => (h.confidence || 0) < 0.70).length}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
