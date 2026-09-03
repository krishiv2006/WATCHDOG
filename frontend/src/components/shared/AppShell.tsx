import { useState, useEffect, type ReactNode } from "react"
import { Link, useLocation, useNavigate } from "react-router-dom"
import { Shield, LayoutDashboard, Database, Activity, TrendingUp, Bell, FileText, Settings, HelpCircle, User, Search, ExternalLink, Grid, X, AlertTriangle, AlertCircle, CheckCircle2 } from "lucide-react"
import { cn } from "@/lib/utils"
import { api, connectAlertsWebSocket } from "@/lib/api"
import { RegisterModelModal } from "@/components/modals/RegisterModelModal"
import { DocsModal } from "@/components/modals/DocsModal"
import { ApiModal } from "@/components/modals/ApiModal"
import { SupportModal } from "@/components/modals/SupportModal"
import { QuickToolsModal } from "@/components/modals/QuickToolsModal"
import { UserAccountModal } from "@/components/modals/UserAccountModal"

function Sidebar({ unacknowledgedCount }: { unacknowledgedCount: number }) {
  const location = useLocation()
  const navigation = [
    { name: 'Dashboard', href: '/dashboard', icon: LayoutDashboard },
    { name: 'Models', href: '/models/m-1', icon: Database },
    { name: 'Drift Analysis', href: '/drift', icon: Activity },
    { name: 'Performance', href: '/performance', icon: TrendingUp },
    { name: 'Alerts', href: '/alerts', icon: Bell, badge: unacknowledgedCount > 0 ? unacknowledgedCount : undefined },
    { name: 'Reports', href: '/reports', icon: FileText },
    { name: 'Settings', href: '/settings', icon: Settings },
  ]
  const bottomNav = [
    { name: 'Help Center', href: '#support', icon: HelpCircle },
    { name: 'Account', href: '#account', icon: User },
  ]

  return (
    <div className="fixed inset-y-0 left-0 z-50 flex w-[230px] flex-col border-r border-outline-variant bg-surface-lowest">
      <div className="flex h-16 shrink-0 items-center px-6 border-b border-outline-variant gap-3">
        <div className="bg-primary text-on-primary p-1.5 rounded-lg">
          <Shield className="h-5 w-5" />
        </div>
        <div className="flex flex-col">
          <span className="text-headline-sm font-semibold !text-[16px]">Watchdog AI</span>
          <span className="text-label-sm text-on-surface-variant uppercase tracking-widest text-[10px]">Enterprise ML Monitor</span>
        </div>
      </div>
      <div className="flex flex-1 flex-col overflow-y-auto px-3 py-4 gap-1">
        {navigation.map((item) => {
          const isActive = location.pathname.startsWith(item.href)
          return (
            <Link
              key={item.name}
              to={item.href}
              className={cn(
                "group flex items-center justify-between rounded-lg px-3 py-2 text-sm font-medium transition-colors relative",
                isActive
                  ? "bg-surface-low text-primary"
                  : "text-on-surface hover:bg-surface hover:text-on-surface"
              )}
            >
              <div className="flex items-center gap-3">
                {isActive && <div className="absolute left-0 top-1.5 bottom-1.5 w-[3px] bg-primary rounded-r-md" />}
                <item.icon className="h-4 w-4" />
                {item.name}
              </div>
              {item.badge && (
                <span className="bg-error text-on-primary text-[10px] font-bold px-1.5 py-0.5 rounded-full">
                  {item.badge}
                </span>
              )}
            </Link>
          )
        })}
      </div>
      <div className="mt-auto px-3 py-4 border-t border-outline-variant gap-1 flex flex-col">
        {bottomNav.map((item) => (
          <a
            key={item.name}
            href={item.href}
            onClick={(e) => {
              e.preventDefault()
              window.dispatchEvent(new CustomEvent('open-modal', { detail: item.name }))
            }}
            className="group flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium text-on-surface-variant hover:bg-surface hover:text-on-surface"
          >
            <item.icon className="h-4 w-4" />
            {item.name}
          </a>
        ))}
      </div>
    </div>
  )
}

function Topbar({
  onOpenDocs,
  onOpenApi,
  onOpenSupport,
  onOpenTools,
  onOpenAccount,
  user,
  recentAlerts,
  unacknowledgedCount,
  onResolveAlert
}: {
  onOpenDocs: () => void
  onOpenApi: () => void
  onOpenSupport: () => void
  onOpenTools: () => void
  onOpenAccount: () => void
  user: any
  recentAlerts: any[]
  unacknowledgedCount: number
  onResolveAlert: (id: string) => void
}) {
  const navigate = useNavigate()
  const [searchQuery, setSearchQuery] = useState('')
  const [models, setModels] = useState<any[]>([])
  const [showSearchDropdown, setShowSearchDropdown] = useState(false)
  const [showAlertsDropdown, setShowAlertsDropdown] = useState(false)

  useEffect(() => {
    api.getModels().then(setModels)
  }, [])

  const filteredModels = searchQuery.trim()
    ? models.filter(m => m.name.toLowerCase().includes(searchQuery.toLowerCase()) || m.algorithm.toLowerCase().includes(searchQuery.toLowerCase()))
    : []

  return (
    <div className="sticky top-0 z-40 flex h-16 shrink-0 items-center gap-x-4 border-b border-outline-variant bg-surface-lowest px-4 sm:gap-x-6 sm:px-6 lg:px-8">
      <h1 className="text-title-lg font-bold flex-1 cursor-pointer" onClick={() => navigate('/dashboard')}>ML Model Monitoring</h1>

      {/* Search Bar with live results dropdown */}
      <div className="flex flex-1 items-center justify-center max-w-md relative">
        <div className="relative w-full">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-on-surface-variant" />
          <input
            type="text"
            placeholder="Search models, alerts..."
            value={searchQuery}
            onChange={(e) => {
              setSearchQuery(e.target.value)
              setShowSearchDropdown(true)
            }}
            onFocus={() => setShowSearchDropdown(true)}
            onBlur={() => setTimeout(() => setShowSearchDropdown(false), 200)}
            className="h-10 w-full rounded-lg border border-outline-variant bg-surface px-10 text-sm focus:outline-none focus:ring-2 focus:ring-primary/20"
          />
          {searchQuery && (
            <button onClick={() => setSearchQuery('')} className="absolute right-3 top-1/2 -translate-y-1/2 text-on-surface-variant hover:text-on-surface">
              <X className="h-4 w-4" />
            </button>
          )}
        </div>

        {/* Live Search Results Overlay */}
        {showSearchDropdown && searchQuery.trim() && (
          <div className="absolute top-12 left-0 right-0 bg-surface-lowest border border-outline-variant rounded-xl shadow-2xl p-2 z-50 space-y-1">
            <div className="text-[10px] uppercase font-bold text-on-surface-variant px-3 py-1">Matching Models</div>
            {filteredModels.length === 0 ? (
              <div className="text-xs text-on-surface-variant px-3 py-2">No matching models found.</div>
            ) : (
              filteredModels.map(m => (
                <button
                  key={m.id}
                  onMouseDown={() => {
                    navigate(`/models/${m.id}`)
                    setShowSearchDropdown(false)
                  }}
                  className="w-full text-left px-3 py-2 rounded-lg hover:bg-surface flex items-center justify-between text-xs"
                >
                  <span className="font-semibold text-on-surface">{m.name} ({m.version})</span>
                  <span className="text-[10px] text-on-surface-variant uppercase">{m.algorithm}</span>
                </button>
              ))
            )}
          </div>
        )}
      </div>

      <div className="flex flex-1 items-center justify-end gap-5 text-sm text-on-surface-variant font-medium">
        <button onClick={onOpenDocs} className="hover:text-on-surface flex items-center gap-1 transition-colors">
          Docs <ExternalLink className="h-3 w-3"/>
        </button>
        <button onClick={onOpenApi} className="hover:text-on-surface transition-colors">
          API
        </button>
        <button onClick={onOpenSupport} className="hover:text-on-surface transition-colors">
          Support
        </button>
        <div className="h-4 w-px bg-outline-variant mx-1" />

        {/* Bell Icon & Notifications Dropdown */}
        <div className="relative">
          <button
            onClick={() => setShowAlertsDropdown(!showAlertsDropdown)}
            className="relative p-1.5 rounded-lg hover:bg-surface text-on-surface-variant hover:text-on-surface transition-colors"
          >
            <Bell className="h-5 w-5" />
            {unacknowledgedCount > 0 && (
              <span className="absolute top-1 right-1 h-2.5 w-2.5 rounded-full bg-error ring-2 ring-surface-lowest" />
            )}
          </button>

          {showAlertsDropdown && (
            <div className="absolute right-0 top-12 w-80 bg-surface-lowest border border-outline-variant rounded-xl shadow-2xl p-3 z-50 space-y-2">
              <div className="flex items-center justify-between border-b border-outline-variant pb-2">
                <span className="font-bold text-xs text-on-surface">Recent Notifications ({unacknowledgedCount})</span>
                <button onClick={() => navigate('/alerts')} className="text-[11px] text-primary hover:underline font-semibold">
                  View All
                </button>
              </div>

              <div className="space-y-1.5 max-h-64 overflow-y-auto">
                {recentAlerts.length === 0 ? (
                  <div className="text-xs text-on-surface-variant text-center py-4">No active alerts.</div>
                ) : (
                  recentAlerts.slice(0, 5).map(a => (
                    <div key={a.id} className="p-2.5 bg-surface rounded-lg border border-outline-variant text-xs space-y-1">
                      <div className="flex items-center justify-between font-bold">
                        <span className={`flex items-center gap-1 ${a.severity === 'critical' ? 'text-error' : 'text-warning-text'}`}>
                          {a.severity === 'critical' ? <AlertCircle className="h-3 w-3" /> : <AlertTriangle className="h-3 w-3" />}
                          {a.title}
                        </span>
                      </div>
                      <p className="text-[11px] text-on-surface-variant line-clamp-1">{a.description}</p>
                      <div className="flex justify-end pt-1">
                        <button
                          onClick={() => onResolveAlert(a.id)}
                          className="text-[10px] text-stable-text font-bold hover:underline flex items-center gap-0.5"
                        >
                          <CheckCircle2 className="h-3 w-3" /> Resolve
                        </button>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}
        </div>

        {/* Square Launcher Icon */}
        <button onClick={onOpenTools} className="p-1.5 rounded-lg hover:bg-surface text-on-surface-variant hover:text-on-surface transition-colors">
          <Grid className="h-5 w-5" />
        </button>

        {/* User Account Avatar (Google Sign-In Trigger) */}
        <button
          onClick={onOpenAccount}
          className="h-8 w-8 rounded-full bg-primary text-on-primary flex flex-col items-center justify-center font-bold text-xs ring-2 ring-primary/20 hover:scale-105 transition-transform"
        >
          {user ? user.initials : 'JS'}
        </button>
      </div>
    </div>
  )
}

export function AppShell({ children }: { children: ReactNode }) {
  const navigate = useNavigate()
  const [user, setUser] = useState<any>(() => {
    const saved = localStorage.getItem('watchdog_user')
    return saved ? JSON.parse(saved) : null
  })
  const [recentAlerts, setRecentAlerts] = useState<any[]>([])
  const [unacknowledgedCount, setUnacknowledgedCount] = useState(0)

  // Modals state
  const [isRegisterOpen, setIsRegisterOpen] = useState(false)
  const [isDocsOpen, setIsDocsOpen] = useState(false)
  const [isApiOpen, setIsApiOpen] = useState(false)
  const [isSupportOpen, setIsSupportOpen] = useState(false)
  const [isToolsOpen, setIsToolsOpen] = useState(false)
  const [isAccountOpen, setIsAccountOpen] = useState(false)

  const fetchAlerts = () => {
    api.getAlerts({ perPage: 10 }).then(res => {
      setRecentAlerts(res.items)
      const unack = res.items.filter(a => a.status === 'unacknowledged').length
      setUnacknowledgedCount(unack)
    }).catch(() => {})
  }

  useEffect(() => {
    fetchAlerts()
    const disconnect = connectAlertsWebSocket(() => {
      fetchAlerts()
    })
    return disconnect
  }, [])

  useEffect(() => {
    const handler = (e: any) => {
      const modalName = e.detail
      if (modalName === 'Help Center' || modalName === 'Support') setIsSupportOpen(true)
      if (modalName === 'Account') setIsAccountOpen(true)
      if (modalName === 'Register Model') setIsRegisterOpen(true)
    }
    window.addEventListener('open-modal', handler)
    return () => window.removeEventListener('open-modal', handler)
  }, [])

  const handleLogin = (userInfo: any) => {
    setUser(userInfo)
    localStorage.setItem('watchdog_user', JSON.stringify(userInfo))
  }

  const handleLogout = () => {
    setUser(null)
    localStorage.removeItem('watchdog_user')
  }

  const handleResolveAlert = async (id: string) => {
    await api.updateAlertStatus(id, 'resolved')
    fetchAlerts()
  }

  return (
    <div className="min-h-screen bg-background text-on-surface font-sans">
      <Sidebar unacknowledgedCount={unacknowledgedCount} />
      <div className="pl-[230px] flex flex-col min-h-screen">
        <Topbar
          onOpenDocs={() => setIsDocsOpen(true)}
          onOpenApi={() => setIsApiOpen(true)}
          onOpenSupport={() => setIsSupportOpen(true)}
          onOpenTools={() => setIsToolsOpen(true)}
          onOpenAccount={() => setIsAccountOpen(true)}
          user={user}
          recentAlerts={recentAlerts}
          unacknowledgedCount={unacknowledgedCount}
          onResolveAlert={handleResolveAlert}
        />
        <main className="flex-1 py-8 px-10">
          <div className="mx-auto max-w-7xl">
            {children}
          </div>
        </main>
      </div>

      {/* Global Modals */}
      <RegisterModelModal
        isOpen={isRegisterOpen}
        onClose={() => setIsRegisterOpen(false)}
        onSuccess={() => {
          setIsRegisterOpen(false)
          window.location.reload()
        }}
      />

      <DocsModal
        isOpen={isDocsOpen}
        onClose={() => setIsDocsOpen(false)}
      />

      <ApiModal
        isOpen={isApiOpen}
        onClose={() => setIsApiOpen(false)}
      />

      <SupportModal
        isOpen={isSupportOpen}
        onClose={() => setIsSupportOpen(false)}
      />

      <QuickToolsModal
        isOpen={isToolsOpen}
        onClose={() => setIsToolsOpen(false)}
        onOpenRegister={() => setIsRegisterOpen(true)}
        onNavigate={(path) => navigate(path)}
      />

      <UserAccountModal
        isOpen={isAccountOpen}
        onClose={() => setIsAccountOpen(false)}
        user={user}
        onLogin={handleLogin}
        onLogout={handleLogout}
      />
    </div>
  )
}
