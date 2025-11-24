import { useState, useEffect } from 'react';
import { Layout } from './components/Layout';
import { EmailEditor } from './components/EmailEditor';
import { ResultsDashboard } from './components/ResultsDashboard';
import { Audiences } from './components/Audiences';
import { History } from './components/History';
import { Settings } from './components/Settings';
import type { SimulationResult } from './types';


function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [emailDraft, setEmailDraft] = useState({
    subject: 'Приглашение на вебинар: AI в маркетинге',
    body: 'Привет! Приглашаем вас на наш эксклюзивный вебинар...',
    cta: 'Зарегистрироваться',
    audience: 'marketing-managers',
    sample_size: 10
  });
  const [simulationResult, setSimulationResult] = useState<SimulationResult | null>(null);
  const [isSimulating, setIsSimulating] = useState(false);

  const [progress, setProgress] = useState(0);

  // Scroll Spy Logic
  useEffect(() => {
    const handleScroll = () => {
      const sections = ['dashboard', 'audiences', 'history', 'settings'];

      // Special case: if near top, set dashboard
      if (window.scrollY < 100) {
        setActiveTab('dashboard');
        return;
      }

      for (const section of sections) {
        const element = document.getElementById(section);
        if (element) {
          const rect = element.getBoundingClientRect();
          // If the section is in the upper half of the viewport
          if (rect.top >= 0 && rect.top <= window.innerHeight / 2) {
            setActiveTab(section);
            break;
          }
          // Or if we've scrolled past it but it's still the most relevant
          if (rect.top < 0 && rect.bottom > window.innerHeight / 2) {
            setActiveTab(section);
            break;
          }
        }
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleSimulate = async () => {
    setIsSimulating(true);
    setSimulationResult(null);
    setProgress(0);

    try {
      const response = await fetch('http://localhost:8000/api/simulate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(emailDraft),
      });

      if (!response.ok) {
        throw new Error('Simulation failed');
      }

      if (!response.body) throw new Error('No response body');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep the last incomplete line in buffer

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const event = JSON.parse(line);
            if (event.type === 'progress') {
              setProgress(event.current);
            } else if (event.type === 'result') {
              setSimulationResult(event.data);
              // Auto-scroll to results
              setTimeout(() => {
                const resultsElement = document.getElementById('results');
                if (resultsElement) {
                  resultsElement.scrollIntoView({ behavior: 'smooth' });
                }
              }, 100);
            } else if (event.type === 'error') {
              console.error('Backend error:', event.message);
              alert(`Simulation error: ${event.message}`);
            }
          } catch (e) {
            console.error('Error parsing JSON line:', line, e);
          }
        }
      }

    } catch (error) {
      console.error('Error simulating:', error);
      alert('Failed to run simulation. Please ensure the backend is running.');
    } finally {
      setIsSimulating(false);
    }
  };

  return (
    <Layout activeTab={activeTab}>
      <div className="w-full">
        <section id="dashboard" className="min-h-screen snap-start flex flex-col justify-center pt-8 pb-20 lg:py-24">
          <EmailEditor
            draft={emailDraft}
            setDraft={setEmailDraft}
            onSimulate={handleSimulate}
            isSimulating={isSimulating}
            currentProgress={progress}
          />
        </section>

        {simulationResult && (
          <section id="results" className="min-h-screen snap-start snap-always flex flex-col justify-center pt-8 pb-20 lg:py-24 border-t border-border/40">
            <div className="animate-in fade-in slide-in-from-bottom-8 duration-700">
              <ResultsDashboard result={simulationResult} />
            </div>
          </section>
        )}

        <section id="audiences" className="min-h-screen snap-start snap-always flex flex-col justify-center pt-8 pb-20 lg:py-24 border-t border-border/40">
          <Audiences />
        </section>

        <section id="history" className="min-h-screen snap-start snap-always flex flex-col justify-center pt-8 pb-20 lg:py-24 border-t border-border/40">
          <History />
        </section>

        <section id="settings" className="min-h-screen snap-start snap-always flex flex-col justify-center pt-8 pb-20 lg:py-24 border-t border-border/40">
          <Settings />
        </section>
      </div>
    </Layout>
  );
}

export default App;
