import { useState, useEffect } from 'react';
import { Users, Plus, UploadCloud, UserPlus, Search, MoreHorizontal, ArrowLeft } from 'lucide-react';
import { Button } from './ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from './ui/dialog';

// Interface matching API response
interface Audience {
  id: number;
  name: string;
  type: string;
  size: number;
  lastUpdated: string;
  personas: any[];
}

export function Audiences() {
  const [audiences, setAudiences] = useState<Audience[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedAudience, setSelectedAudience] = useState<Audience | null>(null);
  const [selectedPersona, setSelectedPersona] = useState<any>(null);
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [newAudienceName, setNewAudienceName] = useState('');

  useEffect(() => {
    fetch('http://localhost:8000/api/audiences')
      .then(res => res.json())
      .then(data => {
        setAudiences(data);
        setLoading(false);
      })
      .catch(err => {
        console.error("Failed to fetch audiences:", err);
        setLoading(false);
      });
  }, []);

  const handleCreateAudience = () => {
    // TODO: Implement create API
    if (newAudienceName) {
      alert("Creation not implemented in backend yet.");
      setIsCreateModalOpen(false);
    }
  };

  if (loading) {
    return <div className="p-12 text-center text-muted-foreground">Загрузка аудиторий...</div>;
  }

  return (
    <div className="space-y-12 animate-in fade-in duration-700">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h2 className="text-4xl font-serif font-bold tracking-tight text-foreground mb-2">Аудитории</h2>
          <p className="text-muted-foreground">Управляйте сегментами для тестирования гипотез.</p>
        </div>
        <Button onClick={() => setIsCreateModalOpen(true)} className="bg-foreground text-background hover:bg-foreground/90 rounded-none h-12 px-8 font-medium">
          <Plus className="mr-2 h-4 w-4" />
          Создать аудиторию
        </Button>
      </div>

      <div className="grid gap-12 md:grid-cols-3">
        <div className="md:col-span-2 space-y-8">
          <div className="flex items-center justify-between pb-4 border-b border-border">
            <h3 className="text-xl font-serif font-semibold text-foreground">Ваши сегменты</h3>
            <div className="relative w-64">
              <Search className="absolute left-0 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <input
                placeholder="Поиск..."
                className="invisible-input pl-8 w-full text-foreground placeholder:text-muted-foreground/50"
              />
            </div>
          </div>

          <div className="space-y-0">
            {audiences.map((audience) => (
              <div
                key={audience.id}
                onClick={() => setSelectedAudience(audience)}
                className="group flex items-center justify-between py-6 border-b border-border/40 cursor-pointer hover:opacity-60 transition-opacity"
              >
                <div className="flex items-center gap-6">
                  <div className="text-3xl grayscale opacity-80">
                    <Users className="h-8 w-8 stroke-1" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-foreground mb-1">{audience.name}</h3>
                    <p className="text-sm text-muted-foreground">{audience.size.toLocaleString()} контактов • {audience.type}</p>
                  </div>
                </div>
                <div className="flex items-center gap-6">
                  <span className="text-xs text-muted-foreground font-medium">{audience.lastUpdated}</span>
                  <MoreHorizontal className="h-5 w-5 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="space-y-8">
          <div className="pb-4 border-b border-border">
            <h3 className="text-xl font-serif font-semibold text-foreground">Импорт</h3>
          </div>
          <div className="space-y-4">
            <button className="w-full flex items-center gap-4 p-4 text-left hover:bg-zinc-50 transition-colors group">
              <UploadCloud className="h-6 w-6 text-muted-foreground group-hover:text-foreground transition-colors" />
              <div>
                <div className="font-medium text-foreground">Загрузить CSV</div>
                <div className="text-xs text-muted-foreground">До 10MB</div>
              </div>
            </button>
            <button className="w-full flex items-center gap-4 p-4 text-left hover:bg-zinc-50 transition-colors group">
              <UserPlus className="h-6 w-6 text-muted-foreground group-hover:text-foreground transition-colors" />
              <div>
                <div className="font-medium text-foreground">Подключить CRM</div>
                <div className="text-xs text-muted-foreground">HubSpot, Salesforce</div>
              </div>
            </button>
          </div>
        </div>
      </div>

      {/* Audience Details Modal */}
      <Dialog open={!!selectedAudience} onOpenChange={() => { setSelectedAudience(null); setSelectedPersona(null); }}>
        <DialogContent className="sm:max-w-[600px] bg-background border-none shadow-2xl p-8 transition-all duration-300">
          {selectedPersona ? (
            <div className="animate-in fade-in slide-in-from-right-8 duration-300">
              <button
                onClick={() => setSelectedPersona(null)}
                className="mb-6 text-sm text-muted-foreground hover:text-foreground flex items-center gap-2 transition-colors"
              >
                <ArrowLeft className="w-4 h-4" />
                Назад к списку
              </button>

              <div className="flex items-start gap-6 mb-8">
                <span className="text-5xl grayscale opacity-80">{selectedPersona.avatar}</span>
                <div>
                  <h3 className="text-2xl font-serif font-bold text-foreground mb-1">{selectedPersona.name}</h3>
                  <p className="text-lg text-muted-foreground">{selectedPersona.role} at {selectedPersona.company}</p>
                </div>
              </div>

              <div className="space-y-8">
                <div>
                  <h4 className="text-xs font-bold text-muted uppercase tracking-wider mb-3">Психографический портрет</h4>
                  <p className="text-foreground leading-relaxed text-base">
                    {selectedPersona.psychographics || "Нет данных о психографии."}
                  </p>
                </div>

                <div>
                  <h4 className="text-xs font-bold text-muted uppercase tracking-wider mb-3">Поведенческие паттерны</h4>
                  <p className="text-foreground leading-relaxed text-base">
                    {selectedPersona.pastBehavior || "Нет данных о прошлом поведении."}
                  </p>
                </div>

                <div className="pt-4 border-t border-border/40 flex gap-8">
                  <div>
                    <span className="block text-xs font-bold text-muted uppercase tracking-wider mb-1">Open Rate</span>
                    <span className="text-xl font-mono text-foreground">{(Math.random() * 40 + 20).toFixed(1)}%</span>
                  </div>
                  <div>
                    <span className="block text-xs font-bold text-muted uppercase tracking-wider mb-1">Reply Rate</span>
                    <span className="text-xl font-mono text-foreground">{(Math.random() * 15 + 2).toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <>
              <DialogHeader className="mb-8">
                <DialogTitle className="flex items-center gap-4 text-3xl">
                  <Users className="h-8 w-8 stroke-1" />
                  <span className="text-foreground font-bold">{selectedAudience?.name}</span>
                </DialogTitle>
                <DialogDescription className="text-muted-foreground text-lg">
                  {selectedAudience?.size.toLocaleString()} участников • Тип: {selectedAudience?.type}
                </DialogDescription>
              </DialogHeader>

              <div className="py-4">
                <h4 className="text-xs font-bold text-muted uppercase tracking-wider mb-6">Профили</h4>
                <div className="space-y-2 max-h-[400px] overflow-y-auto pr-2 -mr-2">
                  {selectedAudience?.personas.map((member: any) => (
                    <div
                      key={member.id}
                      onClick={() => setSelectedPersona(member)}
                      className="flex items-center gap-4 p-3 -mx-3 rounded-lg hover:bg-zinc-50 cursor-pointer group transition-colors"
                    >
                      <span className="text-2xl grayscale opacity-80 group-hover:scale-110 transition-transform duration-200">{member.avatar}</span>
                      <div>
                        <p className="text-base font-medium text-foreground group-hover:text-black transition-colors">{member.name}</p>
                        <p className="text-sm text-muted-foreground">{member.role} at {member.company}</p>
                      </div>
                      <div className="ml-auto opacity-0 group-hover:opacity-100 transition-opacity">
                        <ArrowLeft className="w-4 h-4 rotate-180 text-muted-foreground" />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <DialogFooter className="mt-8">
                <Button variant="ghost" onClick={() => setSelectedAudience(null)} className="hover:bg-transparent hover:text-foreground/70">Закрыть</Button>
                <Button className="bg-foreground text-background hover:bg-foreground/90 rounded-none px-8">Редактировать</Button>
              </DialogFooter>
            </>
          )}
        </DialogContent>
      </Dialog>

      {/* Create Audience Modal */}
      <Dialog open={isCreateModalOpen} onOpenChange={setIsCreateModalOpen}>
        <DialogContent className="bg-background border-none shadow-2xl p-8">
          <DialogHeader className="mb-6">
            <DialogTitle className="text-2xl font-bold text-foreground">Новая аудитория</DialogTitle>
            <DialogDescription className="text-muted-foreground">
              Опишите параметры сегмента или загрузите данные.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-8 py-4">
            <div className="space-y-2">
              <label className="text-sm font-medium text-muted">Название</label>
              <input
                placeholder="напр. Новые пользователи"
                value={newAudienceName}
                onChange={(e) => setNewAudienceName(e.target.value)}
                className="invisible-input w-full text-xl font-medium text-foreground border-b border-border focus:border-foreground py-2 placeholder:text-muted/30"
              />
            </div>
            <div className="space-y-4">
              <label className="text-sm font-medium text-muted">Источник</label>
              <div className="grid grid-cols-2 gap-8">
                <div className="cursor-pointer group">
                  <Users className="h-8 w-8 text-foreground mb-3" />
                  <span className="text-base font-medium text-foreground block">Синтетическая</span>
                  <span className="text-xs text-muted-foreground">Сгенерировать AI</span>
                </div>
                <div className="cursor-pointer group opacity-50 hover:opacity-100 transition-opacity">
                  <UploadCloud className="h-8 w-8 text-foreground mb-3" />
                  <span className="text-base font-medium text-foreground block">Загрузка</span>
                  <span className="text-xs text-muted-foreground">CSV, CRM</span>
                </div>
              </div>
            </div>
          </div>
          <DialogFooter className="mt-8">
            <Button variant="ghost" onClick={() => setIsCreateModalOpen(false)} className="hover:bg-transparent hover:text-foreground/70">Отмена</Button>
            <Button onClick={handleCreateAudience} className="bg-foreground text-background hover:bg-foreground/90 rounded-none px-8">Создать</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
