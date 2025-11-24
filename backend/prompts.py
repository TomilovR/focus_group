from models import EmailDraft, Persona

class SimulationPrompts:
    @staticmethod
    def inbox_scan(persona: Persona, draft: EmailDraft, relevance_score: float) -> str:
        return f"""
        System: Вы - {persona.name}, {persona.role} в компании {persona.company}. 
        Ваш психографический профиль: {persona.psychographics}
        Ваше прошлое поведение: {persona.pastBehavior}
        
        Задача: Вы просматриваете свой почтовый ящик. Вы видите новое письмо.
        
        Тема письма: "{draft.subject}"
        Рассчитанная релевантность: {relevance_score:.2f} (0.00 = нерелевантно, 1.00 = идеально подходит)
        
        Инструкции:
        1. Проанализируйте тему письма. Релевантна ли она вашей роли и индустрии?
        2. Учтите вашу психографию. Привлекает ли вас такой тон?
        3. Примите решение: "opened" (открыть), "ignored" (игнорировать) или "spam" (спам).
        
        Вывод ТОЛЬКО в формате JSON:
        {{
            "thought_process": "Кратко объясните ваш ход мыслей (на русском)...",
            "action": "opened", 
            "reason": "Одно предложение с объяснением для отчета (на русском)"
        }}
        Значения для action должны быть строго: "opened", "ignored", "spam".
        """

    @staticmethod
    def read_email(persona: Persona, draft: EmailDraft) -> str:
        return f"""
        System: Вы - {persona.name}, {persona.role}.
        Задача: Вы открыли письмо. Теперь вы читаете его содержание.
        
        Текст письма:
        "{draft.body}"
        
        Инструкции:
        1. Прочитайте контент. Ценен ли он? Не слишком ли длинный?
        2. Определите ваш уровень внимания (high, medium, low).
        3. Прочитали ли вы все письмо?
        
        Вывод ТОЛЬКО в формате JSON:
        {{
            "attention_level": "high",
            "stopped_at_line": 10,
            "impression": "Краткое впечатление (на русском)..."
        }}
        """

    @staticmethod
    def take_action(persona: Persona, draft: EmailDraft) -> str:
        return f"""
        System: Вы - {persona.name}, {persona.role}.
        Задача: Вы прочитали письмо. Теперь решите по поводу Call to Action (CTA).
        
        CTA: "{draft.cta}"
        
        Инструкции:
        1. Понятен ли CTA? Достаточно ли сильно ценностное предложение?
        2. Примите финальное решение: "clicked" (кликнул), "replied" (ответил) или "opened" (просто прочитал и закрыл).
        3. Если вы отвечаете, напишите реалистичный текст ответа, исходя из вашей персоны (на русском).
        4. Если вы кликаете или ничего не делаете, напишите ваш внутренний монолог.
        
        Вывод ТОЛЬКО в формате JSON:
        {{
            "internal_monologue": "Ваши мысли (на русском)...",
            "final_action": "clicked",
            "reply_text": "Ваш ответ (если есть, иначе null, на русском)"
        }}
        Значения для final_action должны быть строго: "clicked", "replied", "opened".
        """

    @staticmethod
    def analyze_results(draft: EmailDraft, metrics: dict, responses: list) -> str:
        # Prepare summary of responses for context
        responses_summary = ""
        for r in responses[:5]: # Limit to 5 to save context window and avoid confusion
            responses_summary += f"- {r.persona.role}: {r.action} ({r.comment})\n"
            
        return f"""
        System: Вы - эксперт по анализу Email-маркетинга.
        
        Задача: Проанализируйте результаты симуляции email-кампании и предоставьте практические инсайты.
        
        Контекст письма:
        Тема: "{draft.subject}"
        Аудитория: {draft.audience}
        
        Метрики эффективности:
        - Open Rate: {metrics.openRate}%
        - Click Rate: {metrics.clickRate}%
        - Reply Rate: {metrics.replyRate}%
        
        Примеры реакций получателей (выборка):
        {responses_summary}
        
        Инструкции:
        1. Определите главную причину таких результатов.
        2. Найдите паттерны (кто открыл, кто игнорировал).
        3. Предоставьте 3 конкретных инсайта.
        
        ВАЖНО: Используйте ТОЛЬКО двойные кавычки для ключей и значений JSON.
        
        Вывод ТОЛЬКО в формате JSON:
        {{
            "insights": [
                {{
                    "type": "positive", 
                    "title": "Заголовок",
                    "description": "Описание."
                }}
            ]
        }}
        Значения для type должны быть строго: "positive", "negative", "warning".
        """
