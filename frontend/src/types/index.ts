export interface EmailDraft {
  subject: string;
  body: string;
  cta: string;
  audience: string;
  sample_size: number;
}

export interface Persona {
  id: string;
  name: string;
  role: string;
  company: string;
  avatar: string;
  // New detailed fields
  psychographics: string;
  pastBehavior: string;
}

export interface SimulationResponse {
  persona: Persona;
  action: 'opened' | 'ignored' | 'clicked' | 'spam' | 'replied';
  sentiment: 'positive' | 'neutral' | 'negative';
  comment: string;
  detailedReasoning: string; // Why they took this action
}

export interface Insight {
  type: 'positive' | 'negative' | 'warning';
  title: string;
  description: string;
}

export interface SimulationMetrics {
  openRate: number;
  clickRate: number;
  replyRate: number;
  spamRate: number;
  ignoreRate: number;
  forwardRate: number;
  readRate: number; // Attentive reading
}

export interface SimulationResult {
  id: string;
  timestamp: number;
  metrics: SimulationMetrics;
  insights: Insight[]; // Changed from string[]
  responses: SimulationResponse[];
}
