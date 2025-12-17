import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Layout } from './components/Layout'
import { Dashboard } from './pages/Dashboard'
import { Upload } from './pages/Upload'
import { Performance } from './pages/Performance'
import { ConfusionMatrix } from './pages/ConfusionMatrix'
import { Dataset } from './pages/Dataset'
import { History } from './pages/History'

function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/upload" element={<Upload />} />
          <Route path="/performance" element={<Performance />} />
          <Route path="/confusion-matrix" element={<ConfusionMatrix />} />
          <Route path="/dataset" element={<Dataset />} />
          <Route path="/history" element={<History />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  )
}

export default App
