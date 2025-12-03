from models import EmailDraft, Persona

class SimulationPrompts:
    @staticmethod
    def inbox_scan(persona: Persona, draft: EmailDraft, relevance_score: float) -> str:
        return f"""
You are {persona.name}, a {persona.role} at {persona.company}. 
Your psychographic profile: {persona.psychographics}
Your past behavior: {persona.pastBehavior}

Task: You are checking your inbox. You see a new email.

Email Subject: "{draft.subject}"
Relevance Score: {relevance_score:.2f} (0.00 = irrelevant, 1.00 = perfect match)

Analyze the subject line and decide:
1. Is it relevant to your role and industry?
2. Does the tone appeal to your psychographic profile?
3. Make a decision: "opened", "ignored", or "spam".

Respond ONLY with valid JSON (no extra text, no markdown):
{{
    "thought_process": "Brief explanation in Russian",
    "action": "opened",
    "reason": "One sentence explanation in Russian"
}}

Valid action values: "opened", "ignored", "spam"
"""

    @staticmethod
    def read_email(persona: Persona, draft: EmailDraft) -> str:
        return f"""
You are {persona.name}, a {persona.role}.
You have opened the email. Now read its content.

Email Body:
"{draft.body}"

Analyze:
1. Is the content valuable? Too long?
2. Determine your attention level (high, medium, low).
3. Did you read the entire email?

Respond ONLY with valid JSON (no extra text, no markdown):
{{
    "attention_level": "high",
    "stopped_at_line": 10,
    "impression": "Brief impression in Russian"
}}
"""

    @staticmethod
    def take_action(persona: Persona, draft: EmailDraft) -> str:
        return f"""
You are {persona.name}, a {persona.role}.
You have read the email. Now decide about the Call to Action (CTA).

CTA: "{draft.cta}"

Analyze:
1. Is the CTA clear? Is the value proposition strong enough?
2. Make a final decision: "clicked" (clicked the CTA), "replied" (sent a reply), or "opened" (just read and closed).
3. If replying, write a realistic response text in Russian matching your persona.
4. If clicking or doing nothing, write your internal thoughts.

Respond ONLY with valid JSON (no extra text, no markdown):
{{
    "internal_monologue": "Your thoughts in Russian",
    "final_action": "clicked",
    "reply_text": "Your reply if applicable, otherwise null, in Russian"
}}

Valid final_action values: "clicked", "replied", "opened"
"""

    @staticmethod
    def analyze_results(draft: EmailDraft, metrics: dict, responses: list) -> str:
        # Prepare summary of responses for context
        responses_summary = ""
        for r in responses[:5]: # Limit to 5 to save context window
            responses_summary += f"- {r.persona.role}: {r.action} ({r.comment})\n"
            
        return f"""
You are an Email Marketing expert analyzing campaign simulation results.

Email Context:
Subject: "{draft.subject}"
Audience: {draft.audience}

Performance Metrics:
- Open Rate: {metrics.openRate}%
- Click Rate: {metrics.clickRate}%
- Reply Rate: {metrics.replyRate}%
- Spam Rate: {metrics.spamRate}%
- Ignore Rate: {metrics.ignoreRate}%

Sample Recipient Reactions:
{responses_summary}

Task:
1. Identify the main reason for these results.
2. Find patterns (who opened, who ignored).
3. Provide 3 specific actionable insights in Russian.

IMPORTANT: Use ONLY double quotes for JSON keys and string values. No single quotes.

Respond ONLY with valid JSON (no extra text, no markdown, no code blocks):
{{
    "insights": [
        {{
            "type": "positive",
            "title": "Заголовок на русском",
            "description": "Описание на русском."
        }},
        {{
            "type": "negative",
            "title": "Другой заголовок",
            "description": "Другое описание."
        }},
        {{
            "type": "warning",
            "title": "Третий заголовок",
            "description": "Третье описание."
        }}
    ]
}}

Valid type values: "positive", "negative", "warning"
"""
