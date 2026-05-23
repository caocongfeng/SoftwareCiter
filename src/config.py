"""Centralized configuration for the Software Citation Agent.

Manages OpenAI model assignments per module, API keys, search settings,
and other pipeline parameters in one place.
"""

import os
from dotenv import load_dotenv


def load_config() -> dict:
    """Load configuration from environment variables and .env file.

    Returns:
        dict with all configuration values, including per-module model assignments.
    """
    load_dotenv()

    config = {
        # ── OpenAI API ──
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),

        # ── Per-Module Model Assignments ──
        # Each module can use a different model for cost/quality optimization.
        # Override via environment variables or modify defaults below.
        "model_parser": os.getenv("MODEL_PARSER", "gpt-5-mini"),        # XML→text structuring (high volume, simple task)
        #  "model_parser": os.getenv("MODEL_PARSER", "gpt-4o-mini"),        # XML→text structuring (high volume, simple task)
        "model_extractor": os.getenv("MODEL_EXTRACTOR", "gpt-5-mini"),    # Software mention extraction (core task, needs quality)
        "model_searcher": os.getenv("MODEL_SEARCHER", "gpt-5-mini"),      # Search result synthesis
        "model_citation_builder": os.getenv("MODEL_CITATION", "gpt-5-mini"),  # Citation formatting
        "model_verifier": os.getenv("MODEL_VERIFIER", "gpt-5-mini"),      # Citation verification & correction

        # ── LLM Parameters ──
        "llm_temperature": float(os.getenv("LLM_TEMPERATURE", "0")),

        # ── Search Settings ──
        "search_max_results": int(os.getenv("SEARCH_MAX_RESULTS", "10")),

        # ── Parser Settings ──
        "parser_max_chars": int(os.getenv("PARSER_MAX_CHARS", "100000")),  # Max chars to send to LLM parser
        "parser_use_llm": os.getenv("PARSER_USE_LLM", "true").lower() == "true",  # Use LLM for structuring

        # ── Output ──
        "output_format": os.getenv("OUTPUT_FORMAT", "text"),  # text, json, bibtex
    }

    if not config["openai_api_key"]:
        raise ValueError(
            "OPENAI_API_KEY is required. "
            "Set it in .env or as an environment variable."
        )

    return config


def get_model_for_module(config: dict, module: str) -> str:
    """Get the OpenAI model name for a specific module.

    Args:
        config: Configuration dict from load_config().
        module: Module name ('parser', 'extractor', 'searcher',
                'citation_builder', 'verifier').

    Returns:
        Model name string (e.g., 'gpt-5-mini').
    """
    key = f"model_{module}"
    return config.get(key, config.get("model_extractor", "gpt-5-mini"))
