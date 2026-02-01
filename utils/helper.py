import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import hashlib
import logging
from transformers import pipeline

class helper:
    def __init__(self):
        self.parse_mistral_response()
        self.deduplicate_texts()
        self.get_retry_session()
        
    def parse_mistral_response(response_data):
        """
        Safely parse mistral API response with multiple fallback strategies.
        
        Args:
            response_data: JSON response from mistral API
        
        Returns:
            Extracted text content or error message
        """
        try:
            # Strategy 1: Standard OpenAI-compatible format
            if "choices" in response_data and response_data["choices"]:
                message = response_data["choices"][0].get("message", {})
                content = message.get("content")
                if content:
                    return content.strip()
            
            # Strategy 2: Direct output_text field
            if "output_text" in response_data:
                return response_data["output_text"].strip()
            
            # Strategy 3: Text field
            if "text" in response_data:
                return response_data["text"].strip()
            
            # Strategy 4: Nested content structures
            if "result" in response_data:
                result = response_data["result"]
                if isinstance(result, dict) and "content" in result:
                    return result["content"].strip()
                elif isinstance(result, str):
                    return result.strip()
            
            # No valid content found
            logging.warning(f"Unexpected mistral response structure: {list(response_data.keys())}")
            return "No response text available"
            
        except Exception as e:
            logging.error(f"Error parsing mistral response: {e}")
            return f"Response parsing error: {str(e)}"

    def deduplicate_texts(texts, min_length=20):
        """
        Deduplicate texts using hash-based comparison.
        Also filters out texts shorter than min_length.
        
        Args:
            texts: List of text strings
            min_length: Minimum text length to keep
        
        Returns:
            List of unique, valid texts
        """
        seen_hashes = set()
        unique_texts = []
        
        for text in texts:
            text = text.strip()
            if len(text) < min_length:
                continue
            
            # Create hash of normalized text (lowercase, no extra spaces)
            normalized = ' '.join(text.lower().split())
            text_hash = hashlib.md5(normalized.encode()).hexdigest()
            
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_texts.append(text)
        
        return unique_texts

    def get_retry_session(retries=3, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504)):
        """
        Create a requests session with retry logic
        
        Args:
            retries: Number of retry attempts
            backoff_factor: Multiplier for exponential backoff (wait = {backoff_factor} * (2 ** retry_number))
            status_forcelist: HTTP status codes to retry on
        """
        session = requests.Session()
        retry = Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]  # Updated from method_whitelist
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    @st.cache_resource
    def _load_finbert_model():
        """
        Load FinBERT model once and cache it globally.
        This prevents reloading on every Streamlit rerun.
        """
        try:
            logging.info("Loading FinBERT model...")
            model = pipeline('sentiment-analysis', model='yiyanghkust/finbert-tone')
            logging.info("FinBERT model loaded successfully")
            return model
        except Exception as e:
            logging.warning(f"FinBERT load failed: {e}, using default sentiment model")
            return pipeline('sentiment-analysis')