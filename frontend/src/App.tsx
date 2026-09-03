import { Routes, Route, Navigate } from 'react-router-dom'
import { AppShell } from './components/shared/AppShell'
import Dashboard from './pages/Dashboard'
import ModelDetail from './pages/ModelDetail'
import AlertCenter from './pages/AlertCenter'
import Reports from './pages/Reports'
import DriftAnalysis from './pages/DriftAnalysis'
import PerformanceOverview from './pages/PerformanceOverview'
import SettingsPage from './pages/SettingsPage'

function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/models/:id" element={<ModelDetail />} />
        <Route path="/drift" element={<DriftAnalysis />} />
        <Route path="/performance" element={<PerformanceOverview />} />
        <Route path="/alerts" element={<AlertCenter />} />
        <Route path="/reports" element={<Reports />} />
        <Route path="/settings" element={<SettingsPage />} />
      </Routes>
    </AppShell>
  )
}

export default App
