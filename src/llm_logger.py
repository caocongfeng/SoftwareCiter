"""LLM Interaction Logger — records every LLM API call for debugging.

Writes a JSONL file with one entry per LLM call, including:
  - module (extractor, searcher, builder, verifier)
  - step (extract_chunk_1, synthesis, build, verify, aggregate, ...)
  - input messages and output text (truncated for readability)
  - publication ID and software name for cross-referencing

Usage:
    from src.llm_logger import LLMLogger
    logger = LLMLogger(Path("eval_LLM_log.jsonl"))
    logger.log("searcher", "synthesis", "PMC123", "SPSS", messages, response)
"""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class LLMLogger:
    """Thread-safe JSONL logger for LLM API interactions."""

    def __init__(self, log_path: Path, max_content_chars: int = 3000):
        """Initialize logger.

        Args:
            log_path: Path to the JSONL log file.
            max_content_chars: Max chars to store per input/output field.
        """
        self.log_path = Path(log_path)
        self.max_content_chars = max_content_chars
        self._call_count = 0
        self._lock = threading.Lock()

        # Ensure parent directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        module: str,
        step: str,
        pub_id: str,
        sw_name: str,
        input_messages: list,
        output: str,
    ):
        """Log a single LLM API call.

        Args:
            module: Module name (extractor, searcher, builder, verifier).
            step: Step within the module (e.g., extract_chunk_1, synthesis).
            pub_id: Publication identifier.
            sw_name: Software name being processed.
            input_messages: List of LangChain message objects.
            output: Raw LLM output text.
        """
        with self._lock:
            self._call_count += 1
            call_id = self._call_count
        max_c = self.max_content_chars

        # Format input messages
        formatted_input = []
        for m in input_messages:
            role = getattr(m, "type", "unknown")
            content = getattr(m, "content", str(m))
            formatted_input.append({
                "role": role,
                "content": content[:max_c] + ("..." if len(content) > max_c else ""),
            })

        entry = {
            "timestamp": datetime.now().isoformat(),
            "call_id": call_id,
            "module": module,
            "step": step,
            "pub_id": pub_id,
            "software": sw_name,
            "input": formatted_input,
            "output": output[:max_c] + ("..." if len(output) > max_c else ""),
        }

        try:
            line = json.dumps(entry, ensure_ascii=False) + "\n"
            with self._lock:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(line)
        except Exception as e:
            logger.warning(f"Failed to write LLM log: {e}")

    def get_stats(self) -> dict:
        """Return basic statistics about logged calls."""
        return {
            "total_calls": self._call_count,
            "log_file": str(self.log_path),
        }
