import { useEffect, useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card'
import { apiService } from '../services/api'
import { Brain, AlertCircle, BarChart2, Zap } from 'lucide-react'

export function Dashboard() {
  const [stats, setStats] = useState({
    totalImages: 2870,
    numClasses: 4,
    modelsCount: 5,
    bestAccuracy: 92
  })
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [metrics, models] = await Promise.all([
          apiService.getMetrics(),
          apiService.getModels()
        ])
        
        if (metrics) {
          setStats(prev => ({
            ...prev,
            bestAccuracy: Math.max(...metrics.map(m => m.accuracy)) || 92
          }))
        }
      } catch (error) {
        console.error('Failed to fetch data:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [])

  const StatCard = ({ icon: Icon, title, value, unit = '' }) => (
    <Card>
      <CardContent className="flex items-center justify-between">
        <div>
          <p className="text-slate-400 text-sm mb-2">{title}</p>
          <p className="text-3xl font-bold text-white">
            {value}{unit}
          </p>
        </div>
        <div className="p-3 bg-blue-600/20 rounded-lg">
          <Icon className="text-blue-400" size={32} />
        </div>
      </CardContent>
    </Card>
  )

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-white mb-2">Dashboard</h1>
        <p className="text-slate-400">U-Net & XGBoost Framework for Brain Tumor Classification</p>
      </div>

      <div className="grid grid-cols-4 gap-6 mb-8">
        <StatCard icon={Brain} title="Best Model Accuracy" value={stats.bestAccuracy} unit="%" />
        <StatCard icon={AlertCircle} title="Total MRI Images" value={stats.totalImages} />
        <StatCard icon={Zap} title="Tumor Classes" value={stats.numClasses} />
        <StatCard icon={BarChart2} title="Models Trained" value={stats.modelsCount} />
      </div>

      <div className="grid grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Project Overview</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4 text-slate-300">
            <div>
              <h4 className="font-semibold text-white mb-2">Tumor Classifications:</h4>
              <ul className="space-y-1 text-sm">
                <li>• No Tumor</li>
                <li>• Glioma</li>
                <li>• Meningioma</li>
                <li>• Pituitary</li>
              </ul>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Models Used</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-slate-300">
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span>XGBoost</span>
                <span className="text-green-400">★★★★★</span>
              </div>
              <div className="flex justify-between">
                <span>AdaBoost</span>
                <span className="text-green-400">★★★★☆</span>
              </div>
              <div className="flex justify-between">
                <span>Decision Tree</span>
                <span className="text-green-400">★★★★☆</span>
              </div>
              <div className="flex justify-between">
                <span>SVM</span>
                <span className="text-green-400">★★★★☆</span>
              </div>
              <div className="flex justify-between">
                <span>ANN</span>
                <span className="text-green-400">★★★★☆</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
