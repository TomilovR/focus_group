from abc import ABC, abstractmethod
import random
import json
import time
from typing import Optional

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
from config import LLM_BASE_URL, LLM_API_KEY, LLM_TIMEOUT, LLM_MAX_RETRIES, logger

load_dotenv()

class LLMError(Exception):
    """Custom exception for LLM-related errors"""
    pass

class LLMTimeoutError(LLMError):
    """Exception for LLM timeout errors"""
    pass

class OpenAILLM(BaseLLM):
    def __init__(self, base_url: str = None, api_key: str = None):
        from openai import OpenAI
        
        self.base_url = base_url or LLM_BASE_URL
        self.api_key = api_key or LLM_API_KEY
        self.timeout = LLM_TIMEOUT
        self.max_retries = LLM_MAX_RETRIES
        
        try:
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)
            logger.info(f"OpenAI LLM initialized with base_url: {self.base_url}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise LLMError(f"Failed to initialize LLM client: {str(e)}")
        
    def predict(self, prompt: str) -> str:
        """Predict with retry logic and error handling"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"LLM predict attempt {attempt + 1}/{self.max_retries}")
                
                response = self.client.chat.completions.create(
                    model="local-model",  # LM Studio usually ignores this, but it's required
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant simulating a specific persona. Always respond in valid JSON when requested."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    timeout=self.timeout
                )
                
                content = response.choices[0].message.content
                logger.debug(f"LLM response received: {len(content)} chars")
                return content
                
            except Exception as e:
                error_type = type(e).__name__
                last_error = e
                
                # Check for timeout errors
                if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                    logger.warning(f"LLM timeout on attempt {attempt + 1}: {str(e)}")
                    if attempt < self.max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"LLM timeout after {self.max_retries} attempts")
                        raise LLMTimeoutError(f"LLM request timed out after {self.max_retries} attempts")
                
                # Check for connection errors
                elif "connection" in str(e).lower() or "network" in str(e).lower():
                    logger.warning(f"LLM connection error on attempt {attempt + 1}: {str(e)}")
                    if attempt < self.max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"LLM connection failed after {self.max_retries} attempts")
                        raise LLMError(f"Failed to connect to LLM service: {str(e)}")
                
                # For other errors, log and raise
                else:
                    logger.error(f"LLM error ({error_type}): {str(e)}")
                    raise LLMError(f"LLM prediction failed: {str(e)}")
        
        # If all retries failed
        logger.error(f"LLM failed after {self.max_retries} attempts. Last error: {str(last_error)}")
        raise LLMError(f"LLM prediction failed after {self.max_retries} attempts")
    
    def predict_with_fallback(self, prompt: str, fallback_response: Optional[str] = None) -> str:
        """Predict with a fallback response if LLM fails"""
        try:
            return self.predict(prompt)
        except LLMError as e:
            logger.warning(f"LLM failed, using fallback: {str(e)}")
            if fallback_response:
                return fallback_response
            # Return a generic fallback JSON
            return json.dumps({
                "action": "ignored",
                "reason": "Unable to generate response",
                "error": True
            })

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.use_fallback = False
            logger.info(f"Loaded embedding model: {model_name}")
        except ImportError:
            logger.warning("sentence-transformers not found, using fallback.")
            self.use_fallback = True
        except Exception as e:
            logger.warning(f"Error loading embedding model: {e}, using fallback.")
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
            logger.error(f"Embedding calculation failed: {e}")
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
