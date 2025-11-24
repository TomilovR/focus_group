import type { EmailDraft, SimulationResult, Persona, Insight } from '../types';

export const generateMockResult = (draft: EmailDraft, overrideMetrics?: any): SimulationResult => {
  const isGoodSubject = draft.subject.length > 10 && !draft.subject.toLowerCase().includes('бесплатно');
  
  const baseOpenRate = isGoodSubject ? 45 : 15;
  const baseClickRate = isGoodSubject ? 12 : 2;

  let metrics;
  
  if (overrideMetrics) {
    metrics = {
      openRate: overrideMetrics.open,
      clickRate: Math.floor(overrideMetrics.open * 0.3), // Approximate click rate based on open rate
      replyRate: Math.floor(Math.random() * 5),
      spamRate: Math.floor(Math.random() * 5),
      ignoreRate: overrideMetrics.ignore,
      forwardRate: Math.floor(Math.random() * 3),
      readRate: overrideMetrics.read
    };
  } else {
    metrics = {
      openRate: Math.min(100, Math.floor(baseOpenRate + Math.random() * 10)),
      clickRate: Math.min(100, Math.floor(baseClickRate + Math.random() * 5)),
      replyRate: Math.floor(Math.random() * 5),
      spamRate: isGoodSubject ? Math.floor(Math.random() * 2) : Math.floor(10 + Math.random() * 20),
      ignoreRate: 0, // Calculated below
      forwardRate: Math.floor(Math.random() * 3),
      readRate: Math.floor(baseOpenRate * 0.6) // 60% of openers read attentively
    };
    metrics.ignoreRate = 100 - (metrics.openRate + metrics.spamRate);
  }

  const personas: Persona[] = [
    { 
      id: '1', name: 'Алексей Петров', role: 'CTO', company: 'ТехФлоу', avatar: '👨‍💻',
      psychographics: 'Прагматик, ценит краткость и техническую конкретику. Ненавидит маркетинговый булшит.',
      pastBehavior: 'Часто открывает письма с техническими заголовками, редко кликает.'
    },
    { 
      id: '2', name: 'Мария Иванова', role: 'VP Engineering', company: 'КлаудСкейл', avatar: '👩‍💼',
      psychographics: 'Ориентирована на рост команды и эффективность процессов. Ищет решения для масштабирования.',
      pastBehavior: 'Отвечает на персонализированные письма.'
    },
    { 
      id: '3', name: 'Михаил Сидоров', role: 'DevOps Lead', company: 'Стартап Инк', avatar: '👷',
      psychographics: 'Скептик. Ищет подвох. Любит open-source решения.',
      pastBehavior: 'Часто помечает "холодные" письма как спам.'
    },
    { 
      id: '4', name: 'Елена Смирнова', role: 'Product Manager', company: 'Саасифай', avatar: '👩‍🎨',
      psychographics: 'Визуал, ценит понятные презентации и кейсы. Ищет новые фичи для продукта.',
      pastBehavior: 'Кликает на ссылки с демо-версиями.'
    },
    { 
      id: '5', name: 'Дмитрий Козлов', role: 'Основатель', company: 'ИИ Лабс', avatar: '🤵',
      psychographics: 'Визионер, но очень занят. Читает только первые 2 строки.',
      pastBehavior: 'Игнорирует длинные письма.'
    },
  ];

  const responses = personas.map(p => {
    const rand = Math.random();
    let action: 'opened' | 'ignored' | 'clicked' | 'spam' | 'replied' = 'ignored';
    let comment = '';
    let detailedReasoning = '';

    if (rand > 0.8) {
      action = 'replied';
      comment = "Интересное предложение, но у нас уже заключен контракт. Напишите в 3 квартале?";
      detailedReasoning = "Письмо попало в текущую потребность компании, но тайминг неудачный. Персональный тон письма побудил к вежливому ответу вместо игнорирования.";
    } else if (rand > 0.6) {
      action = 'clicked';
      comment = "Кликнул, чтобы проверить цены. Кажется дороговато для нас сейчас.";
      detailedReasoning = "CTA был четким и обещал конкретную ценность. Пользователь заинтересовался деталями, но ценовое позиционирование вызвало сомнения.";
    } else if (rand > 0.3) {
      action = 'opened';
      comment = "Открыл, потому что тема письма зацепила, но текст слишком длинный.";
      detailedReasoning = "Тема письма (Subject Line) была релевантной, но тело письма (Body) оказалось перегруженным текстом, что привело к потере внимания.";
    } else {
      action = 'ignored';
      comment = "Проигнорировал. Выглядит как обычная рекламная рассылка.";
      detailedReasoning = "Письмо не прошло фильтр 'свой-чужой'. Слишком общие фразы и отсутствие персонализации заставили пользователя принять это за массовый спам.";
    }

    return {
      persona: p,
      action,
      sentiment: 'neutral',
      comment,
      detailedReasoning
    } as const;
  });

  const insights: Insight[] = [
    isGoodSubject ? {
      type: 'positive',
      title: 'Сильная тема письма',
      description: 'Заголовок короткий, емкий и не содержит стоп-слов. Это обеспечивает высокий Open Rate.'
    } : {
      type: 'negative',
      title: 'Слабая тема письма',
      description: 'Тема выглядит как спам или слишком общая. Рекомендуется добавить конкретики или персонализации.'
    },
    {
      type: 'warning',
      title: 'Риск игнорирования CTO',
      description: 'Технические директора (CTO) склонны игнорировать это письмо из-за недостатка технических деталей в первом абзаце.'
    },
    {
      type: 'positive',
      title: 'Хороший CTA',
      description: 'Призыв к действию четкий и понятный, что положительно влияет на Click Rate среди менеджеров.'
    }
  ];

  return {
    id: Date.now().toString(),
    timestamp: Date.now(),
    metrics,
    insights,
    responses
  };
};
