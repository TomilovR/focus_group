import { Input } from './ui/input';
import { Button } from './ui/button';
import { Save, Shield, Zap, Key } from 'lucide-react';

export function Settings() {
  return (
    <div className="space-y-16 animate-in fade-in duration-500 max-w-4xl">
      <div>
        <h2 className="text-4xl font-serif font-bold tracking-tight text-foreground mb-2">Настройки</h2>
        <p className="text-muted-foreground">Конфигурация параметров симуляции.</p>
      </div>

      <div className="space-y-12">
        <div className="space-y-8">
          <div className="flex items-center gap-4 pb-4 border-b border-border">
            <Zap className="h-6 w-6 text-foreground" />
            <h3 className="text-2xl font-serif font-bold text-foreground">Модель ИИ</h3>
          </div>

          <div className="space-y-6 max-w-xl">
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <label className="text-base font-medium text-foreground">Температура (Креативность)</label>
                <span className="text-sm font-bold text-foreground">0.7</span>
              </div>
              <Input type="range" min="0" max="100" defaultValue="70" className="cursor-pointer h-1 p-0 border-none bg-muted accent-foreground rounded-full" />
              <div className="flex justify-between text-xs text-muted font-bold uppercase tracking-wide">
                <span>Строгая логика</span>
                <span>Высокая креативность</span>
              </div>
            </div>
          </div>
        </div>

        <div className="space-y-8">
          <div className="flex items-center gap-4 pb-4 border-b border-border">
            <Shield className="h-6 w-6 text-foreground" />
            <h3 className="text-2xl font-serif font-bold text-foreground">Безопасность</h3>
          </div>

          <div className="space-y-6 max-w-xl">
            <div className="space-y-2">
              <label className="text-base font-medium text-foreground">API Key (OpenAI / Anthropic)</label>
              <div className="relative">
                <div className="absolute left-0 top-1/2 -translate-y-1/2 text-foreground">
                  <Key className="h-5 w-5" />
                </div>
                <input
                  type="password"
                  value="sk-........................"
                  readOnly
                  className="invisible-input pl-8 w-full text-lg font-mono text-foreground border-b border-border focus:border-foreground py-2"
                />
                <div className="absolute inset-y-0 right-0 flex items-center pointer-events-none">
                  <span className="text-xs font-bold text-muted-foreground flex items-center gap-1">
                    <Shield className="h-3 w-3" />
                    Encrypted
                  </span>
                </div>
              </div>
              <p className="text-xs text-muted-foreground pt-2">Ключ хранится локально в зашифрованном виде.</p>
            </div>
          </div>
        </div>

        <div className="pt-8">
          <Button size="lg" className="bg-foreground text-background hover:bg-foreground/90 rounded-none h-12 px-8 font-medium text-base">
            <Save className="mr-2 h-4 w-4" />
            Сохранить настройки
          </Button>
        </div>
      </div>
    </div>
  );
}
