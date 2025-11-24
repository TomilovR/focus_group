from pydantic import BaseModel
from typing import List, Optional, Literal

class EmailDraft(BaseModel):
    subject: str
    body: str
    cta: str
    audience: str
    sample_size: int = 10

class Persona(BaseModel):
    id: str
    name: str
    role: str
    company: str
    avatar: str
    psychographics: str
    pastBehavior: str

class Response(BaseModel):
    persona: Persona
    action: Literal['opened', 'ignored', 'clicked', 'spam', 'replied']
    sentiment: Literal['positive', 'neutral', 'negative']
    comment: str
    detailedReasoning: str

class Insight(BaseModel):
    type: Literal['positive', 'negative', 'warning']
    title: str
    description: str

class Metrics(BaseModel):
    openRate: int
    clickRate: int
    replyRate: int
    spamRate: int
    ignoreRate: int
    forwardRate: int
    readRate: int

class SimulationResult(BaseModel):
    id: str
    timestamp: int
    metrics: Metrics
    insights: List[Insight]
    responses: List[Response]
