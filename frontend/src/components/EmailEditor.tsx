import { useState, useEffect, useRef } from 'react';
import { Button } from './ui/button';
import { ChevronDown, Check } from 'lucide-react';
import type { EmailDraft } from '../types';

interface Audience {
  id: number;
  name: string;
  type: string;
  size: number;
}

interface EmailEditorProps {
  draft: EmailDraft;
  setDraft: (draft: EmailDraft) => void;
  onSimulate: () => void;
  isSimulating: boolean;
  currentProgress?: number;
}

export function EmailEditor({ draft, setDraft, onSimulate, isSimulating, currentProgress = 0 }: EmailEditorProps) {
  const [audiences, setAudiences] = useState<Audience[]>([]);
  const [isAudienceOpen, setIsAudienceOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch('http://localhost:8000/api/audiences')
      .then(res => res.json())
      .then(data => setAudiences(data))
      .catch(err => console.error("Failed to fetch audiences:", err));
  }, []);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsAudienceOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const selectedAudienceName = audiences.find(a => a.id.toString() === draft.audience)?.name || draft.audience || "Выберите аудиторию";

  return (
    <div className="h-full flex flex-col animate-in fade-in duration-700">
      <div className="mb-8">
        <h1 className="text-4xl font-serif font-bold tracking-tight text-foreground mb-2">Черновик</h1>
        <p className="text-muted-foreground">Создайте идеальное письмо для вашей аудитории.</p>
      </div>

      <div className="space-y-6 flex-1 flex flex-col min-h-0">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="group space-y-2 relative" ref={dropdownRef}>
            <label className="block text-[10px] font-mono uppercase tracking-widest text-muted-foreground group-focus-within:text-foreground transition-colors">Аудитория</label>

            <div
              onClick={() => setIsAudienceOpen(!isAudienceOpen)}
              className="w-full text-lg font-medium text-foreground border-b border-border/20 hover:border-foreground transition-colors py-2 cursor-pointer flex items-center justify-between"
            >
              <span>{selectedAudienceName}</span>
              <ChevronDown className={`w-4 h-4 text-muted-foreground transition-transform duration-200 ${isAudienceOpen ? 'rotate-180' : ''}`} />
            </div>

            {isAudienceOpen && (
              <div className="absolute top-full left-0 w-full z-50 mt-2 bg-background border border-border/10 shadow-2xl rounded-sm overflow-hidden animate-in fade-in slide-in-from-top-2 duration-200">
                {audiences.map((audience) => (
                  <div
                    key={audience.id}
                    onClick={() => {
                      setDraft({ ...draft, audience: audience.id.toString() });
                      setIsAudienceOpen(false);
                    }}
                    className="px-4 py-3 hover:bg-zinc-50 cursor-pointer flex items-center justify-between group transition-colors"
                  >
                    <div>
                      <div className="font-medium text-foreground">{audience.name}</div>
                      <div className="text-xs text-muted-foreground">{audience.size} участников • {audience.type}</div>
                    </div>
                    {draft.audience === audience.id.toString() && <Check className="w-4 h-4 text-foreground" />}
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="group space-y-2">
            <label className="block text-[10px] font-mono uppercase tracking-widest text-muted-foreground group-focus-within:text-foreground transition-colors">
              Размер выборки: {draft.sample_size}
            </label>
            <div className="py-2">
              <input
                type="range"
                min="5"
                max="50"
                step="5"
                value={draft.sample_size}
                onChange={(e) => setDraft({ ...draft, sample_size: parseInt(e.target.value) })}
                className="w-full h-1 bg-zinc-200 rounded-lg appearance-none cursor-pointer accent-foreground"
              />
            </div>
          </div>
        </div>

        <div className="group space-y-2">
          <label className="block text-[10px] font-mono uppercase tracking-widest text-muted-foreground group-focus-within:text-foreground transition-colors">Тема</label>
          <input
            placeholder="Введите тему..."
            value={draft.subject}
            onChange={(e) => setDraft({ ...draft, subject: e.target.value })}
            className="invisible-input w-full text-lg font-medium text-foreground placeholder:text-muted/20 py-2"
          />
        </div>

        <div className="group space-y-2 flex-1 flex flex-col min-h-0">
          <label className="block text-[10px] font-mono uppercase tracking-widest text-muted-foreground group-focus-within:text-foreground transition-colors">Сообщение</label>
          <textarea
            placeholder="Напишите текст письма..."
            className="invisible-input flex-1 w-full min-h-[300px] text-base font-sans leading-relaxed resize-none placeholder:text-muted/20 py-2 overflow-y-auto"
            value={draft.body}
            onChange={(e) => setDraft({ ...draft, body: e.target.value })}
          />
        </div>

        <div className="group space-y-2">
          <label className="block text-[10px] font-mono uppercase tracking-widest text-muted-foreground group-focus-within:text-foreground transition-colors">Call to Action</label>
          <input
            placeholder="Текст кнопки..."
            value={draft.cta}
            onChange={(e) => setDraft({ ...draft, cta: e.target.value })}
            className="invisible-input w-full text-lg font-medium text-foreground placeholder:text-muted/20 py-2"
          />
        </div>

        <div className="mt-12 flex items-center justify-between pt-8 border-t border-border/40">
          <div className="flex flex-col gap-4 w-full max-w-md">
            {isSimulating ? (
              <div className="space-y-2">
                <div className="flex justify-between text-xs font-mono uppercase tracking-widest text-muted-foreground">
                  <span>Генерация ответов...</span>
                  <span>{currentProgress} / {draft.sample_size}</span>
                </div>
                <div className="h-2 w-full bg-zinc-100 overflow-hidden">
                  <div
                    className="h-full bg-foreground transition-all duration-300 ease-out"
                    style={{ width: `${(currentProgress / draft.sample_size) * 100}%` }}
                  />
                </div>
              </div>
            ) : (
              <Button
                onClick={onSimulate}
                className="bg-foreground text-background hover:bg-foreground/90 rounded-none h-12 px-8 font-medium text-lg w-fit"
              >
                Запустить симуляцию
              </Button>
            )}
          </div>
          <span className="text-xs text-muted-foreground font-mono tracking-widest uppercase">
            {draft.body.length} / 1000
          </span>
        </div>
      </div>
    </div>
  );
}
