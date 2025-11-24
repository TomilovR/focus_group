import { useState, useEffect } from 'react';
import { Calendar, ArrowRight, Users, Trash2 } from 'lucide-react';
import { Button } from './ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from './ui/dialog';
import { ResultsDashboard } from './ResultsDashboard';
import type { SimulationResult } from '../types';

interface HistoryItem {
  id: string;
  timestamp: number;
  subject: string;
  metrics: {
    openRate: number;
    clickRate: number;
  };
  audience: string;
}

interface DetailedSimulationResult extends SimulationResult {
  subject: string;
  body: string;
  cta: string;
}

export function History() {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedResult, setSelectedResult] = useState<DetailedSimulationResult | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  useEffect(() => {
    fetchHistory();

    // Poll for updates every 5 seconds (simple way to keep history fresh)
    const interval = setInterval(fetchHistory, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchHistory = () => {
    fetch('http://localhost:8000/api/history')
      .then(res => res.json())
      .then(data => {
        setHistory(data);
        setLoading(false);
      })
      .catch(err => {
        console.error("Failed to fetch history:", err);
        setLoading(false);
      });
  };

  const handleClearHistory = async () => {
    if (!confirm('Вы уверены, что хотите очистить всю историю? Это действие нельзя отменить.')) {
      return;
    }

    try {
      await fetch('http://localhost:8000/api/history', {
        method: 'DELETE',
      });
      fetchHistory();
    } catch (err) {
      console.error("Failed to clear history:", err);
      alert("Не удалось очистить историю");
    }
  };

  const handleViewResult = (id: string) => {
    setDetailLoading(true);
    fetch(`http://localhost:8000/api/history/${id}`)
      .then(res => res.json())
      .then(data => {
        setSelectedResult(data);
        setDetailLoading(false);
      })
      .catch(err => {
        console.error("Failed to fetch details:", err);
        setDetailLoading(false);
      });
  };

  const formatDate = (timestamp: number) => {
    return new Date(timestamp).toLocaleString('ru-RU', {
      day: 'numeric',
      month: 'short',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  if (loading) {
    return <div className="p-12 text-center text-muted-foreground">Загрузка истории...</div>;
  }

  return (
    <div className="space-y-8 animate-in fade-in duration-700 h-full flex flex-col">
      <div className="flex items-end justify-between">
        <div>
          <h2 className="text-4xl font-serif font-bold tracking-tight text-foreground mb-2">История</h2>
          <p className="text-muted-foreground">Архив всех проведенных тестов.</p>
        </div>
        {history.length > 0 && (
          <Button
            variant="ghost"
            size="sm"
            onClick={handleClearHistory}
            className="text-muted-foreground hover:text-red-500 hover:bg-red-50"
          >
            <Trash2 className="w-4 h-4 mr-2" />
            Очистить
          </Button>
        )}
      </div>

      <div className="space-y-0 flex-1 overflow-y-auto max-h-[600px] pr-2 custom-scrollbar">
        {history.length === 0 ? (
          <div className="text-center py-12 text-muted-foreground">
            История пуста. Запустите первую симуляцию!
          </div>
        ) : (
          history.map((item) => (
            <div
              key={item.id}
              className="group py-6 flex items-center justify-between border-b border-border/40 cursor-pointer hover:opacity-60 transition-opacity"
              onClick={() => handleViewResult(item.id)}
            >
              <div className="flex items-center gap-8">
                <div className="text-3xl text-muted-foreground/30 font-light w-12">
                  #
                </div>
                <div>
                  <h3 className="text-xl font-semibold text-foreground mb-2">{item.subject}</h3>
                  <div className="flex items-center gap-6 text-sm text-muted-foreground">
                    <span className="flex items-center gap-2">
                      <Calendar className="h-4 w-4" />
                      {formatDate(item.timestamp)}
                    </span>
                    <span className="flex items-center gap-2">
                      <Users className="h-4 w-4" />
                      {item.audience}
                    </span>
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-12">
                <div className="text-right hidden md:block">
                  <div className="text-xs font-bold text-muted uppercase tracking-wider mb-1">Open Rate</div>
                  <div className="font-bold text-2xl text-foreground">{item.metrics.openRate}%</div>
                </div>
                <ArrowRight className="h-6 w-6 text-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
              </div>
            </div>
          ))
        )}
      </div>

      <Dialog open={!!selectedResult || detailLoading} onOpenChange={() => setSelectedResult(null)}>
        <DialogContent className="max-w-5xl h-[90vh] overflow-y-auto bg-background border-none shadow-2xl p-0">
          <DialogHeader className="p-8 pb-0">
            <DialogTitle className="text-2xl font-bold text-foreground">Детали симуляции</DialogTitle>
            <DialogDescription className="text-muted-foreground">Полный отчет о прогнозируемой эффективности.</DialogDescription>
          </DialogHeader>
          <div className="p-8 pt-4">
            {detailLoading ? (
              <div className="text-center py-12">Загрузка деталей...</div>
            ) : (
              selectedResult && (
                <div className="space-y-8">
                  <div className="bg-zinc-50 p-6 rounded-lg border border-border/40 space-y-4">
                    <div>
                      <div className="text-xs font-mono uppercase tracking-widest text-muted-foreground mb-1">Тема</div>
                      <div className="text-lg font-medium text-foreground">{selectedResult.subject}</div>
                    </div>
                    <div>
                      <div className="text-xs font-mono uppercase tracking-widest text-muted-foreground mb-1">Сообщение</div>
                      <div className="text-base text-foreground whitespace-pre-wrap font-serif leading-relaxed opacity-80 max-h-40 overflow-y-auto custom-scrollbar">
                        {selectedResult.body}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs font-mono uppercase tracking-widest text-muted-foreground mb-1">CTA</div>
                      <div className="text-sm font-medium text-foreground bg-white border border-border/20 px-3 py-1 rounded-md w-fit">
                        {selectedResult.cta}
                      </div>
                    </div>
                  </div>

                  <ResultsDashboard result={selectedResult} />
                </div>
              )
            )}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
