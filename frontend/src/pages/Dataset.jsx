import { useEffect, useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card'
import { PieChart, Pie, BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { apiService } from '../services/api'

export function Dataset() {
  const [insights, setInsights] = useState(null)
  const [loading, setLoading] = useState(true)

  const defaultInsights = {
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

  useEffect(() => {
    const fetchInsights = async () => {
      try {
        const data = await apiService.getDatasetInsights()
        setInsights(data || defaultInsights)
      } catch (error) {
        console.error('Failed to fetch insights:', error)
        setInsights(defaultInsights)
      } finally {
        setLoading(false)
      }
    }

    fetchInsights()
  }, [])

  const COLORS = ['#ef4444', '#f97316', '#eab308', '#06b6d4']

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-white mb-2">Dataset Insights</h1>
        <p className="text-slate-400">Analysis of training data distribution and splits</p>
      </div>

      <div className="grid grid-cols-2 gap-6 mb-8">
        <Card>
          <CardHeader>
            <CardTitle>Class Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={insights?.classDistribution || defaultInsights.classDistribution}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, value }) => `${name}: ${value}`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {(insights?.classDistribution || defaultInsights.classDistribution).map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                  labelStyle={{ color: '#fff' }}
                />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Data Split</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={insights?.dataSplit || defaultInsights.dataSplit}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="name" stroke="#cbd5e1" />
                <YAxis stroke="#cbd5e1" />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                  labelStyle={{ color: '#fff' }}
                />
                <Bar dataKey="value" fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Dataset Statistics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-4 gap-4">
            <div className="bg-slate-700/30 rounded-lg p-4">
              <p className="text-slate-400 text-sm mb-2">Total Images</p>
              <p className="text-2xl font-bold text-white">2,870</p>
            </div>
            <div className="bg-slate-700/30 rounded-lg p-4">
              <p className="text-slate-400 text-sm mb-2">Training Samples</p>
              <p className="text-2xl font-bold text-blue-400">60%</p>
            </div>
            <div className="bg-slate-700/30 rounded-lg p-4">
              <p className="text-slate-400 text-sm mb-2">Validation Samples</p>
              <p className="text-2xl font-bold text-green-400">20%</p>
            </div>
            <div className="bg-slate-700/30 rounded-lg p-4">
              <p className="text-slate-400 text-sm mb-2">Test Samples</p>
              <p className="text-2xl font-bold text-orange-400">20%</p>
            </div>
          </div>

          <div className="mt-6 space-y-3">
            <h4 className="font-semibold text-white">Class Details</h4>
            {(insights?.classDistribution || defaultInsights.classDistribution).map((item, index) => (
              <div key={item.name} className="flex items-center justify-between p-3 bg-slate-700/30 rounded-lg">
                <div className="flex items-center gap-3">
                  <div 
                    className="w-4 h-4 rounded"
                    style={{ backgroundColor: COLORS[index % COLORS.length] }}
                  />
                  <span className="text-white">{item.name}</span>
                </div>
                <span className="text-slate-400">{item.value} images</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
