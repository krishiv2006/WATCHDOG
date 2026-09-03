import { useState, useEffect } from 'react'
import { X, User, LogOut, Check, ShieldCheck, Mail, Key } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface Props {
  isOpen: boolean
  onClose: () => void
  user: any
  onLogin: (userInfo: any) => void
  onLogout: () => void
}

export function UserAccountModal({ isOpen, onClose, user, onLogin, onLogout }: Props) {
  const [googleEmail, setGoogleEmail] = useState('')
  const [googleName, setGoogleName] = useState('')
  const [isLoggingIn, setIsLoggingIn] = useState(false)

  if (!isOpen) return null

  const handleSimulatedGoogleLogin = (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoggingIn(true)

    setTimeout(() => {
      const loggedUser = {
        name: googleName || 'John Smith',
        email: googleEmail || 'john.smith@watchdog.ai',
        picture: 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=150&auto=format&fit=crop&q=80',
        initials: (googleName || 'John Smith').split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2),
        role: 'Lead ML Engineer',
        authProvider: 'Google OAuth 2.0',
      }
      onLogin(loggedUser)
      setIsLoggingIn(false)
      onClose()
    }, 800)
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4 backdrop-blur-sm animate-in fade-in">
      <div className="w-full max-w-md rounded-xl border border-outline-variant bg-surface-lowest p-6 shadow-2xl space-y-6">
        <div className="flex items-center justify-between border-b border-outline-variant pb-4">
          <div className="flex items-center gap-2.5">
            <div className="h-10 w-10 rounded-full bg-primary text-on-primary font-bold text-sm flex items-center justify-center ring-2 ring-primary/20">
              {user ? user.initials : 'JS'}
            </div>
            <div>
              <h2 className="text-title-lg font-bold text-on-surface">{user ? user.name : 'Account Profile'}</h2>
              <p className="text-body-sm text-on-surface-variant">{user ? user.email : 'Google Single Sign-On'}</p>
            </div>
          </div>
          <button onClick={onClose} className="text-on-surface-variant hover:text-on-surface p-1 rounded-md hover:bg-surface">
            <X className="h-5 w-5" />
          </button>
        </div>

        {user ? (
          <div className="space-y-4">
            <div className="p-4 bg-surface rounded-xl border border-outline-variant space-y-3">
              <div className="flex items-center justify-between text-xs">
                <span className="text-on-surface-variant">Authentication Status</span>
                <span className="bg-stable-bg/50 text-stable-text font-bold px-2 py-0.5 rounded-full flex items-center gap-1">
                  <ShieldCheck className="h-3 w-3" /> Authenticated
                </span>
              </div>
              <div className="text-xs space-y-1">
                <p className="text-on-surface font-semibold">Role: <span className="font-normal text-on-surface-variant">{user.role}</span></p>
                <p className="text-on-surface font-semibold">Provider: <span className="font-normal text-on-surface-variant">{user.authProvider}</span></p>
              </div>
            </div>

            <Button variant="destructive" className="w-full justify-center" onClick={() => { onLogout(); onClose(); }}>
              <LogOut className="h-4 w-4 mr-2" /> Sign Out
            </Button>
          </div>
        ) : (
          <form onSubmit={handleSimulatedGoogleLogin} className="space-y-4">
            <div className="p-4 bg-surface rounded-xl border border-outline-variant text-center space-y-3">
              <div className="flex justify-center">
                <div className="p-3 rounded-full bg-white shadow border border-gray-200">
                  <svg className="h-6 w-6" viewBox="0 0 24 24">
                    <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                    <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                    <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"/>
                    <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"/>
                  </svg>
                </div>
              </div>
              <h3 className="font-bold text-sm text-on-surface">Google Sign-In</h3>
              <p className="text-xs text-on-surface-variant">Sign in with your Google Workspace or corporate email to access Watchdog AI dashboards.</p>
            </div>

            <div>
              <label className="block text-label-md font-semibold text-on-surface-variant mb-1">Google Email</label>
              <input
                type="email"
                placeholder="john.smith@watchdog.ai"
                value={googleEmail}
                onChange={(e) => setGoogleEmail(e.target.value)}
                className="w-full h-10 px-3 rounded-lg border border-outline-variant bg-surface text-sm focus:outline-none focus:border-primary"
              />
            </div>

            <div>
              <label className="block text-label-md font-semibold text-on-surface-variant mb-1">Full Name</label>
              <input
                type="text"
                placeholder="John Smith"
                value={googleName}
                onChange={(e) => setGoogleName(e.target.value)}
                className="w-full h-10 px-3 rounded-lg border border-outline-variant bg-surface text-sm focus:outline-none focus:border-primary"
              />
            </div>

            <Button type="submit" className="w-full justify-center" disabled={isLoggingIn}>
              {isLoggingIn ? 'Authenticating with Google...' : 'Continue with Google Single Sign-On'}
            </Button>
          </form>
        )}

        <div className="flex justify-end border-t border-outline-variant pt-4">
          <Button variant="outline" onClick={onClose}>Close</Button>
        </div>
      </div>
    </div>
  )
}
