import os
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import hashlib
import logging
from transformers import pipeline


def setup_logging(level: str | None = None) -> None:
    """
    Configure root logger to emit human-readable logs with timestamp, level, and message.

    Safe to call multiple times; it will not duplicate handlers.
    """
    root = logging.getLogger()
    if getattr(root, "_streamlit_logging_configured", False):
        return

    log_level_name = (level or os.environ.get("LOG_LEVEL") or "INFO").upper()
    try:
        log_level = getattr(logging, log_level_name, logging.INFO)
    except Exception:
        log_level = logging.INFO

    root.setLevel(log_level)

    # Remove existing basic handlers (e.g. basicConfig)
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler()
    # Classic: timestamp, level, module:line, message
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(module)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)

    root._streamlit_logging_configured = True  # type: ignore[attr-defined]
    root.info("Logging initialized")


class helper:
    """
    Utility namespace. Methods are intentionally static-like;
    avoid relying on instance state.
    """

    def __init__(self):
        # Kept for backward-compat if instantiated somewhere;
        # real usage is via static methods.
        pass

    @staticmethod
    def parse_mistral_response(response_data):
        """
        Safely parse Mistral API response (OpenAI-compatible format).
        Args:
            response_data: JSON response from Mistral API
        Returns:
            Extracted text content or error message
        """
        try:
            if "choices" in response_data and response_data["choices"]:
                message = response_data["choices"][0].get("message", {})
                content = message.get("content")
                if content:
                    return content.strip()
            logging.warning(
                "Unexpected Mistral response structure",
                extra={"error_category": "api"},
            )
            return "No response text available"
        except Exception as e:
            logging.error(
                f"Error parsing Mistral response: {e}",
                extra={"error_category": "logic"},
            )
            return f"Response parsing error: {str(e)}"

    @staticmethod
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
            text = (text or "").strip()
            if len(text) < min_length:
                continue

            # Create hash of normalized text (lowercase, no extra spaces)
            normalized = " ".join(text.lower().split())
            text_hash = hashlib.md5(normalized.encode()).hexdigest()

            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_texts.append(text)

        return unique_texts

    @staticmethod
    def get_retry_session(
        retries=3,
        backoff_factor=2,
        status_forcelist=(429, 500, 502, 503, 504),
    ):
        """
        Create a requests session with retry logic.

        Args:
            retries: Number of retry attempts
            backoff_factor: Multiplier for exponential backoff
            status_forcelist: HTTP status codes to retry on
        """
        session = requests.Session()
        retry = Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    @staticmethod
    @st.cache_resource
    def _load_finbert_model():
        """
        Load FinBERT model once and cache it globally.
        This prevents reloading on every Streamlit rerun.
        """
        try:
            logging.info(
                "Loading FinBERT model...",
                extra={"pipeline_step": "model_load"},
            )
            model = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")
            logging.info(
                "FinBERT model loaded successfully",
                extra={"pipeline_step": "model_load"},
            )
            return model
        except Exception as e:
            logging.warning(
                f"FinBERT load failed: {e}, using default sentiment model",
                extra={"error_category": "model"},
            )
            return pipeline("sentiment-analysis")