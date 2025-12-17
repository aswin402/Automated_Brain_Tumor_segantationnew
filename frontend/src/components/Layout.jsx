import { Link, useLocation } from 'react-router-dom'
import { Activity, Upload, BarChart3, Grid3X3, Database, History } from 'lucide-react'

export function Layout({ children }) {
  const location = useLocation()

  const navItems = [
    { path: '/', label: 'Dashboard', icon: Activity },
    { path: '/upload', label: 'Upload & Predict', icon: Upload },
    { path: '/performance', label: 'Model Performance', icon: BarChart3 },
    { path: '/confusion-matrix', label: 'Confusion Matrix', icon: Grid3X3 },
    { path: '/dataset', label: 'Dataset Insights', icon: Database },
    { path: '/history', label: 'History', icon: History }
  ]

  return (
    <div className="flex h-screen bg-primary">
      <aside className="w-64 bg-secondary border-r border-slate-700 overflow-y-auto">
        <div className="p-6 border-b border-slate-700">
          <h1 className="text-2xl font-bold text-white">Brain Tumor AI</h1>
          <p className="text-sm text-slate-400 mt-1">Classification Dashboard</p>
        </div>
        
        <nav className="p-4">
          {navItems.map((item) => {
            const Icon = item.icon
            const isActive = location.pathname === item.path
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center gap-3 px-4 py-3 rounded-lg mb-2 transition-colors ${
                  isActive
                    ? 'bg-blue-600 text-white'
                    : 'text-slate-300 hover:bg-slate-700'
                }`}
              >
                <Icon size={20} />
                <span>{item.label}</span>
              </Link>
            )
          })}
        </nav>
      </aside>

      <main className="flex-1 overflow-auto bg-primary">
        <div className="p-8">
          {children}
        </div>
      </main>
    </div>
  )
}
