import json
import time
from typing import List
from models import EmailDraft, Persona, SimulationResult, Response, Metrics, Insight
from llm_service import BaseLLM, MockLLM, OpenAILLM, EmbeddingService
from profiles import generate_personas
from prompts import SimulationPrompts

class Simulator:
    def __init__(self, llm: BaseLLM = None):
        if llm:
            self.llm = llm
        else:
            try:
                self.llm = OpenAILLM()
            except ImportError:
                print("OpenAI library not found, falling back to MockLLM")
                self.llm = MockLLM()
        
        self.embedding_service = EmbeddingService()

    def run_simulation_stream(self, draft: EmailDraft):
        personas = generate_personas(draft.sample_size, audience_id=draft.audience)
        responses = []
        
        open_count = 0
        click_count = 0
        reply_count = 0
        spam_count = 0
        ignore_count = 0
        read_count = 0
        forward_count = 0

        total = len(personas)

        for i, p in enumerate(personas):
            response = self._simulate_single_persona(draft, p)
            responses.append(response)
            
            # Update counts
            if response.action == 'opened': open_count += 1
            if response.action == 'clicked': click_count += 1
            if response.action == 'replied': reply_count += 1
            if response.action == 'spam': spam_count += 1
            if response.action == 'ignored': ignore_count += 1
            
            # Heuristics for other metrics based on action
            if response.action in ['opened', 'clicked', 'replied']:
                read_count += 1
            
            # Random forward
            if response.action == 'clicked' and hash(p.id) % 5 == 0:
                forward_count += 1
            
            # Yield progress
            yield {
                "type": "progress",
                "current": i + 1,
                "total": total
            }

        # Calculate metrics
        metrics = Metrics(
            openRate=int((open_count / total) * 100) if total > 0 else 0,
            clickRate=int((click_count / total) * 100) if total > 0 else 0,
            replyRate=int((reply_count / total) * 100) if total > 0 else 0,
            spamRate=int((spam_count / total) * 100) if total > 0 else 0,
            ignoreRate=int((ignore_count / total) * 100) if total > 0 else 0,
            forwardRate=int((forward_count / total) * 100) if total > 0 else 0,
            readRate=int((read_count / total) * 100) if total > 0 else 0
        )

        insights = self._generate_insights(draft, metrics, responses)

        result = SimulationResult(
            id=str(int(time.time())),
            timestamp=int(time.time() * 1000),
            metrics=metrics,
            insights=insights,
            responses=responses
        )
        
        yield {
            "type": "result",
            "data": result.dict()
        }

    def _simulate_single_persona(self, draft: EmailDraft, persona: Persona) -> Response:
        # Calculate relevance score
        persona_context = f"{persona.role} {persona.company} {persona.psychographics} {persona.pastBehavior}"
        relevance_score = self.embedding_service.get_similarity(draft.subject, persona_context)
        
        # Phase A: Inbox Scan
        prompt_a = SimulationPrompts.inbox_scan(persona, draft, relevance_score)
        
        try:
            res_a_str = self.llm.predict(prompt_a)
            res_a_str = res_a_str.replace("```json", "").replace("```", "").strip()
            res_a = json.loads(res_a_str)
        except:
            res_a = {"action": "ignored", "reason": "Failed to parse decision", "thought_process": "Error parsing LLM response"}
        
        action = res_a.get("action", "ignored").lower()
        if action not in ['opened', 'ignored', 'spam']:
            action = 'ignored'
            
        reason = res_a.get("reason", "Not relevant")
        detailed_reasoning = res_a.get("thought_process", reason)
        comment = reason
        
        if action == "opened":
            # Phase B: Reading (Optional, currently just for internal state)
            # prompt_b = SimulationPrompts.read_email(persona, draft)
            # res_b = ... 
            
            # Phase C: Action
            prompt_c = SimulationPrompts.take_action(persona, draft)
            
            try:
                res_c_str = self.llm.predict(prompt_c)
                res_c_str = res_c_str.replace("```json", "").replace("```", "").strip()
                res_c = json.loads(res_c_str)
            except:
                res_c = {"final_action": "opened", "internal_monologue": "Undecided"}
            
            final_action = res_c.get("final_action", "opened").lower()
            if final_action in ['clicked', 'replied']:
                action = final_action
            
            comment = res_c.get("reply_text") if action == 'replied' else res_c.get("internal_monologue", reason)
            detailed_reasoning = res_c.get("internal_monologue", reason)

        return Response(
            persona=persona,
            action=action,
            sentiment='neutral', # TODO: Analyze sentiment
            comment=comment or "No comment",
            detailedReasoning=detailed_reasoning
        )

    def _generate_insights(self, draft: EmailDraft, metrics: Metrics, responses: list) -> List[Insight]:
        insights = []
        
        # Try to get smart insights from LLM
        try:
            prompt = SimulationPrompts.analyze_results(draft, metrics, responses)
            llm_response = self.llm.predict(prompt)
            print(f"DEBUG: LLM Raw Response for Insights: {llm_response}") # Debug print
            
            import json
            import re
            
            # Use regex to find the JSON object
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            
            if json_match:
                cleaned_response = json_match.group(0)
                try:
                    data = json.loads(cleaned_response)
                except json.JSONDecodeError:
                    print("DEBUG: JSON decode failed, trying ast.literal_eval")
                    import ast
                    data = ast.literal_eval(cleaned_response)
                
                if "insights" in data:
                    for item in data["insights"]:
                        insight_type = item.get("type", "warning").lower()
                        if insight_type == 'issue':
                            insight_type = 'negative'
                        elif insight_type not in ['positive', 'negative', 'warning']:
                            insight_type = 'warning'

                        insights.append(Insight(
                            type=insight_type,
                            title=item.get("title", "Insight"),
                            description=item.get("description", "")
                        ))
                    print("Generated smart insights via LLM.")
                    return insights
            else:
                print("DEBUG: No JSON found in LLM response.")
                
        except Exception as e:
            print(f"Failed to generate smart insights: {e}. Falling back to heuristics.")
            import traceback
            traceback.print_exc()
        
        # Fallback Heuristics
        if metrics.openRate < 20:
            insights.append(Insight(
                type='negative',
                title='Низкий Open Rate',
                description='Тема письма недостаточно привлекательна для этой аудитории.'
            ))
        elif metrics.openRate > 40:
            insights.append(Insight(
                type='positive',
                title='Высокий Open Rate',
                description='Тема письма работает отлично.'
            ))
            
        if metrics.spamRate > 10:
             insights.append(Insight(
                type='warning',
                title='Высокий риск спама',
                description='Многие получатели отметили письмо как спам. Проверьте стоп-слова.'
            ))
            
        return insights
