from abc import ABC, abstractmethod
import random
import json

class BaseLLM(ABC):
    @abstractmethod
    def predict(self, prompt: str) -> str:
        pass

class MockLLM(BaseLLM):
    def predict(self, prompt: str) -> str:
        # Simple heuristic-based mock response
        # In a real scenario, this would call OpenAI/Gemini
        
        if "Phase A" in prompt: # Open Rate
            return json.dumps({
                "action": random.choice(["opened", "opened", "ignored", "spam"]),
                "reason": "Subject line was catchy" if random.random() > 0.5 else "Subject looked generic"
            })
        
        if "Phase B" in prompt: # Read Rate
            return json.dumps({
                "attention_level": random.choice(["high", "low"]),
                "stopped_at_line": random.randint(3, 10)
            })
            
        if "Phase C" in prompt: # Action
            return json.dumps({
                "final_action": random.choice(["clicked", "replied", "ignored"]),
                "reply_text": "Спасибо, интересно." if random.random() > 0.5 else "",
                "internal_monologue": "I need this solution right now."
            })
            
        return "{}"

import os
from dotenv import load_dotenv

load_dotenv()

class OpenAILLM(BaseLLM):
    def __init__(self, base_url: str = None, api_key: str = None):
        from openai import OpenAI
        
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234/v1")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "lm-studio")
        
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        
    def predict(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="local-model", # LM Studio usually ignores this, but it's required
                messages=[
                    {"role": "system", "content": "You are a helpful assistant simulating a specific persona. Always respond in valid JSON when requested."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return "{}"

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.use_fallback = False
            print(f"Loaded embedding model: {model_name}")
        except ImportError:
            print("sentence-transformers not found, using fallback.")
            self.use_fallback = True
        except Exception as e:
            print(f"Error loading embedding model: {e}, using fallback.")
            self.use_fallback = True
        
    def get_similarity(self, text1: str, text2: str) -> float:
        """
        Calculates similarity between two texts.
        Returns a score between 0.0 and 1.0.
        """
        if self.use_fallback:
            return self._keyword_similarity(text1, text2)

        try:
            # Generate embeddings
            embeddings = self.model.encode([text1, text2])
            vec1 = embeddings[0]
            vec2 = embeddings[1]
            
            return self._cosine_similarity(vec1, vec2)
        except Exception as e:
            print(f"Embedding calculation failed: {e}")
            return self._keyword_similarity(text1, text2)

    def _cosine_similarity(self, vec1, vec2) -> float:
        import numpy as np
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    def _keyword_similarity(self, text1: str, text2: str) -> float:
        # Simple Jaccard similarity on words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Remove common stop words (very basic list)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
