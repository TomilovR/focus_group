import json
import time
import re
from typing import List
from models import EmailDraft, Persona, SimulationResult, Response, Metrics, Insight
from llm_service import BaseLLM, MockLLM, OpenAILLM, EmbeddingService, LLMError
from profiles import generate_personas
from prompts import SimulationPrompts
from config import logger

class Simulator:
    def __init__(self, llm: BaseLLM = None):
        if llm:
            self.llm = llm
        else:
            try:
                self.llm = OpenAILLM()
                logger.info("Initialized OpenAI LLM for simulation")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI LLM: {e}, falling back to MockLLM")
                self.llm = MockLLM()
        
        self.embedding_service = EmbeddingService()

    def _parse_llm_json(self, llm_response: str, fallback: dict = None) -> dict:
        """
        Robust JSON parser for LLM responses.
        Handles markdown code blocks, extra text, and malformed JSON.
        """
        if not llm_response:
            logger.warning("Empty LLM response received")
            return fallback or {}
        
        try:
            # Remove markdown code blocks
            cleaned = llm_response.strip()
            cleaned = re.sub(r'^```json\s*', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'^```\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
            cleaned = cleaned.strip()
            
            # Try direct JSON parse first
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass
            
            # Extract JSON object using regex
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            
            # Try to find JSON array
            array_match = re.search(r'\[[^\[\]]*(?:\{[^{}]*\}[^\[\]]*)*\]', cleaned, re.DOTALL)
            if array_match:
                return {"data": json.loads(array_match.group(0))}
            
            logger.error(f"Could not extract valid JSON from: {llm_response[:200]}...")
            return fallback or {}
            
        except Exception as e:
            logger.error(f"JSON parsing failed: {e}. Response: {llm_response[:200]}...")
            return fallback or {}

    def run_simulation_stream(self, draft: EmailDraft):
        logger.info(f"Starting simulation for audience: {draft.audience}")
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
        logger.info(f"Simulating {total} personas")

        for i, p in enumerate(personas):
            try:
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
                
            except Exception as e:
                logger.error(f"Error simulating persona {p.name}: {e}")
                # Add fallback response
                responses.append(Response(
                    persona=p,
                    action='ignored',
                    sentiment='neutral',
                    comment='Simulation error occurred',
                    detailedReasoning=f'Error: {str(e)}'
                ))
                ignore_count += 1
            
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

        logger.info(f"Simulation metrics: {metrics.dict()}")
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
        logger.info("Simulation completed successfully")

    def _simulate_single_persona(self, draft: EmailDraft, persona: Persona) -> Response:
        # Calculate relevance score
        persona_context = f"{persona.role} {persona.company} {persona.psychographics} {persona.pastBehavior}"
        relevance_score = self.embedding_service.get_similarity(draft.subject, persona_context)
        
        # Phase A: Inbox Scan
        prompt_a = SimulationPrompts.inbox_scan(persona, draft, relevance_score)
        
        try:
            res_a_str = self.llm.predict(prompt_a)
            res_a = self._parse_llm_json(res_a_str, fallback={
                "action": "ignored", 
                "reason": "Unable to parse response",
                "thought_process": "LLM response parsing failed"
            })
        except LLMError as e:
            logger.error(f"LLM error in Phase A for {persona.name}: {e}")
            res_a = {
                "action": "ignored", 
                "reason": "LLM unavailable",
                "thought_process": str(e)
            }
        except Exception as e:
            logger.error(f"Unexpected error in Phase A for {persona.name}: {e}")
            res_a = {
                "action": "ignored", 
                "reason": "Processing error",
                "thought_process": str(e)
            }
        
        action = res_a.get("action", "ignored").lower()
        if action not in ['opened', 'ignored', 'spam']:
            action = 'ignored'
            
        reason = res_a.get("reason", "Not relevant")
        detailed_reasoning = res_a.get("thought_process", reason)
        comment = reason
        
        if action == "opened":
            # Phase C: Action
            prompt_c = SimulationPrompts.take_action(persona, draft)
            
            try:
                res_c_str = self.llm.predict(prompt_c)
                res_c = self._parse_llm_json(res_c_str, fallback={
                    "final_action": "opened",
                    "internal_monologue": "Read but no action taken"
                })
            except LLMError as e:
                logger.error(f"LLM error in Phase C for {persona.name}: {e}")
                res_c = {"final_action": "opened", "internal_monologue": "LLM unavailable"}
            except Exception as e:
                logger.error(f"Unexpected error in Phase C for {persona.name}: {e}")
                res_c = {"final_action": "opened", "internal_monologue": "Processing error"}
            
            final_action = res_c.get("final_action", "opened").lower()
            if final_action in ['clicked', 'replied']:
                action = final_action
            
            comment = res_c.get("reply_text") if action == 'replied' else res_c.get("internal_monologue", reason)
            detailed_reasoning = res_c.get("internal_monologue", reason)

        return Response(
            persona=persona,
            action=action,
            sentiment='neutral',
            comment=comment or "No comment",
            detailedReasoning=detailed_reasoning or "No detailed reasoning"
        )

    def _generate_insights(self, draft: EmailDraft, metrics: Metrics, responses: list) -> List[Insight]:
        insights = []
        
        # Try to get smart insights from LLM
        try:
            prompt = SimulationPrompts.analyze_results(draft, metrics, responses)
            llm_response = self.llm.predict(prompt)
            logger.debug(f"LLM insights response length: {len(llm_response)}")
            
            data = self._parse_llm_json(llm_response)
            
            if data and "insights" in data:
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
                    
                if insights:
                    logger.info(f"Generated {len(insights)} smart insights via LLM")
                    return insights
            
            logger.warning("LLM response did not contain valid insights, using heuristics")
                
        except Exception as e:
            logger.warning(f"Failed to generate smart insights: {e}. Falling back to heuristics.")
        
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
        
        if metrics.clickRate > 15:
            insights.append(Insight(
                type='positive',
                title='Хороший Click Rate',
                description='CTA эффективен и побуждает к действию.'
            ))
            
        logger.info(f"Generated {len(insights)} heuristic insights")
        return insights
