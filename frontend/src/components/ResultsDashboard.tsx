import { ArrowRight } from 'lucide-react';
import { cn } from '../lib/utils';
import { useState } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from './ui/dialog';
import { generateMockResult } from '../lib/mockData';
import type { SimulationResult } from '../types';

export function ResultsDashboard({ result: propResult }: { result?: SimulationResult | null }) {
  const [internalResult] = useState(() => generateMockResult({
    subject: 'Default Email',
    body: 'This is a default email body.',
    cta: 'Click here',
    audience: 'tech-leaders',
    sample_size: 10
  }));
  const result = propResult || internalResult;
  const [selectedPersona, setSelectedPersona] = useState<any>(null);

  return (
    <div className="space-y-8 lg:space-y-12 animate-in fade-in duration-700 h-full flex flex-col">
      <div className="grid gap-8 lg:gap-12 grid-cols-2 lg:grid-cols-4 shrink-0">
        <MetricItem
          label="Open Rate"
          value={`${result.metrics.openRate}%`}
          trend="+2.4%"
        />
        <MetricItem
          label="Click Rate"
          value={`${result.metrics.clickRate}%`}
          trend="+1.1%"
        />
        <MetricItem
          label="Reply Rate"
          value={`${result.metrics.replyRate}%`}
          trend="+0.5%"
        />
        <MetricItem
          label="Spam Score"
          value={`${result.metrics.spamRate}%`}
          trend="-0.2%"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12 flex-1 min-h-0">
        <div className="flex flex-col min-h-0">
          <h3 className="text-lg font-serif font-semibold mb-6 text-foreground">Реакции</h3>
          <div className="space-y-0 flex-1 overflow-y-auto max-h-[600px] pr-2">
            {result.responses.map((response, idx) => (
              <ResponseItem key={idx} response={response} onClick={() => setSelectedPersona(response)} />
            ))}
          </div>
        </div>

        <div className="flex flex-col min-h-0">
          <h3 className="text-lg font-serif font-semibold mb-6 text-foreground">Инсайты</h3>
          <div className="space-y-6 flex-1 overflow-y-auto">
            {result.insights.map((insight, idx) => (
              <div key={idx} className="group">
                <div className="flex items-baseline justify-between mb-1">
                  <h4 className="text-base font-medium text-foreground">{insight.title}</h4>
                  <span className={cn("text-xs font-medium px-2 py-0.5 rounded-full",
                    insight.type === 'positive' ? "bg-zinc-100 text-zinc-900" :
                      insight.type === 'warning' ? "bg-zinc-100 text-zinc-900" :
                        "bg-zinc-100 text-zinc-900"
                  )}>
                    {insight.type}
                  </span>
                </div>
                <p className="text-sm text-muted-foreground leading-relaxed">{insight.description}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      <Dialog open={!!selectedPersona} onOpenChange={() => setSelectedPersona(null)}>
        <DialogContent className="sm:max-w-[600px] bg-background border-none shadow-2xl p-8">
          <DialogHeader className="mb-6">
            <DialogTitle className="flex items-center gap-4">
              <span className="text-4xl">{selectedPersona?.persona.avatar}</span>
              <div>
                <span className="text-2xl font-bold block text-foreground">{selectedPersona?.persona.name}</span>
                <span className="text-base text-muted-foreground">{selectedPersona?.persona.role} • {selectedPersona?.persona.company}</span>
              </div>
            </DialogTitle>
            <DialogDescription className="hidden">Details</DialogDescription>
          </DialogHeader>
          <div className="space-y-8">
            <div className="grid grid-cols-2 gap-8">
              <div>
                <h4 className="text-xs font-bold text-muted uppercase tracking-wider mb-2">Психотип</h4>
                <p className="text-base text-foreground">{selectedPersona?.persona.psychographics}</p>
              </div>
              <div>
                <h4 className="text-xs font-bold text-muted uppercase tracking-wider mb-2">История</h4>
                <p className="text-base text-foreground">{selectedPersona?.persona.pastBehavior}</p>
              </div>
            </div>

            <div>
              <h4 className="text-xs font-bold text-muted uppercase tracking-wider mb-2">Мысли</h4>
              <p className="text-lg text-foreground italic leading-relaxed">
                "{selectedPersona?.detailedReasoning}"
              </p>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function MetricItem({ label, value, trend }: { label: string, value: string, trend: string }) {
  return (
    <div className="flex flex-col items-start gap-2">
      <span className="text-sm text-muted-foreground font-medium lowercase">{label}</span>
      <div className="flex flex-col items-start">
        <span className="text-6xl font-black tracking-tighter leading-none">{value}</span>
        <span className={cn("text-xs font-medium mt-2", trend.startsWith('+') ? "text-emerald-600" : "text-rose-600")}>
          {trend}
        </span>
      </div>
    </div>
  );
}

function translateAction(action: string) {
  const map: Record<string, string> = {
    'opened': 'Открыто',
    'clicked': 'Клик',
    'replied': 'Ответ',
    'ignored': 'Игнорировано',
    'spam': 'Спам'
  };
  return map[action.toLowerCase()] || action;
}

function ResponseItem({ response, onClick }: { response: any, onClick: () => void }) {
  return (
    <div
      onClick={onClick}
      className="group py-4 border-b border-border/40 cursor-pointer hover:opacity-60 transition-opacity flex items-start gap-4"
    >
      <div className="text-2xl shrink-0 grayscale opacity-100">{response.persona.avatar}</div>
      <div className="flex-1 min-w-0">
        <div className="flex items-baseline justify-between mb-1">
          <h4 className="text-base font-medium text-foreground truncate">
            {response.persona.name}
          </h4>
          <span className="text-xs font-medium text-muted-foreground">
            {translateAction(response.action)}
          </span>
        </div>
        <p className="text-sm text-muted-foreground line-clamp-2 italic">
          "{response.comment}"
        </p>
      </div>
      <ArrowRight className="w-4 h-4 text-foreground opacity-0 group-hover:opacity-100 transition-opacity mt-1" />
    </div>
  )
}
