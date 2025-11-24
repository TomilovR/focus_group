import React from 'react';
import { cn } from '../lib/utils';



export function Sidebar({ className, activeTab }: { className?: string, activeTab: string }) {
  const scrollToSection = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <div className={cn("pb-12 w-64 bg-background text-foreground h-screen sticky top-0 flex flex-col z-20 border-r border-border/50", className)}>
      <div className="space-y-8 py-12">
        <div className="px-8 mb-8">
          <span className="text-foreground font-bold text-2xl tracking-tight font-serif lowercase">email ai</span>
        </div>
        <div className="px-6 space-y-4">
          <button
            className={cn("nav-link w-full flex items-center gap-3 px-2 text-left", activeTab === 'dashboard' && "nav-link-active")}
            onClick={() => scrollToSection('dashboard')}
          >
            дашборд
          </button>
          <button
            className={cn("nav-link w-full flex items-center gap-3 px-2 text-left", activeTab === 'audiences' && "nav-link-active")}
            onClick={() => scrollToSection('audiences')}
          >
            аудитории
          </button>
          <button
            className={cn("nav-link w-full flex items-center gap-3 px-2 text-left", activeTab === 'history' && "nav-link-active")}
            onClick={() => scrollToSection('history')}
          >
            история
          </button>
          <button
            className={cn("nav-link w-full flex items-center gap-3 px-2 text-left", activeTab === 'settings' && "nav-link-active")}
            onClick={() => scrollToSection('settings')}
          >
            настройки
          </button>
        </div>
      </div>
      <div className="mt-auto px-8 pb-12">
        <p className="text-xs text-muted font-medium">Pro Plan</p>
        <p className="text-[10px] text-muted-foreground mt-1">750 / 1000 credits</p>
      </div>
    </div>
  );
}

export function Layout({ children, activeTab }: { children: React.ReactNode, activeTab: string }) {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = React.useState(false);

  const scrollToSection = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
      setIsMobileMenuOpen(false);
    }
  };

  return (
    <div className="flex min-h-screen bg-background font-sans text-foreground">
      {/* Desktop Sidebar */}
      <div className="hidden lg:block">
        <Sidebar activeTab={activeTab} />
      </div>

      {/* Mobile Header */}
      <div className="lg:hidden fixed top-0 left-0 right-0 z-50 bg-background/80 backdrop-blur-md border-b border-border/40 px-6 h-16 flex items-center justify-between">
        <span className="text-foreground font-bold text-xl tracking-tight font-serif lowercase">email ai</span>
        <button onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)} className="p-2 -mr-2">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            {isMobileMenuOpen ? <path d="M18 6 6 18M6 6l12 12" /> : <path d="M3 12h18M3 6h18M3 18h18" />}
          </svg>
        </button>
      </div>

      {/* Mobile Menu Overlay */}
      {isMobileMenuOpen && (
        <div className="lg:hidden fixed inset-0 z-40 bg-background pt-24 px-6 animate-in slide-in-from-top-10 duration-200">
          <div className="flex flex-col gap-6 text-2xl font-serif font-medium">
            <button onClick={() => scrollToSection('dashboard')} className={cn("text-left", activeTab === 'dashboard' ? "text-foreground" : "text-muted-foreground")}>дашборд</button>
            <button onClick={() => scrollToSection('audiences')} className={cn("text-left", activeTab === 'audiences' ? "text-foreground" : "text-muted-foreground")}>аудитории</button>
            <button onClick={() => scrollToSection('history')} className={cn("text-left", activeTab === 'history' ? "text-foreground" : "text-muted-foreground")}>история</button>
            <button onClick={() => scrollToSection('settings')} className={cn("text-left", activeTab === 'settings' ? "text-foreground" : "text-muted-foreground")}>настройки</button>
          </div>
        </div>
      )}

      <main className="flex-1 px-4 lg:px-24 pt-20 lg:pt-0 transition-all duration-500 ease-in-out">
        <div className="mx-auto max-w-5xl w-full">
          {children}
        </div>
      </main>
    </div>
  );
}
