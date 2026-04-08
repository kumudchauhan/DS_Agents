"""LLM routing for the DA Agent: Ollama (local) or OpenRouter (cloud)."""

from __future__ import annotations

import urllib.request
import urllib.error


def _ollama_available(base_url: str = "http://localhost:11434") -> bool:
    """Ping the Ollama endpoint to check if it's running."""
    try:
        req = urllib.request.Request(base_url, method="GET")
        with urllib.request.urlopen(req, timeout=3):
            return True
    except (urllib.error.URLError, OSError):
        return False


def get_da_llm(
    local_mode: bool = True,
    model_name: str | None = None,
    temperature: float = 0.2,
):
    """Return an LLM instance for the DA Agent.

    Parameters
    ----------
    local_mode : bool
        True  -> ChatOllama on localhost:11434
        False -> OpenRouter via existing get_llm()
    model_name : str | None
        Override the default model name.
    temperature : float
        Sampling temperature (default 0.2 for more deterministic output).
    """
    if not local_mode:
        from app.llm.llm_provider import get_llm
        return get_llm(
            model_name=model_name or "mistralai/mistral-small-3.1-24b-instruct",
            temperature=temperature,
        )

    # --- Ollama (local) ---
    base_url = "http://localhost:11434"

    if not _ollama_available(base_url):
        raise RuntimeError(
            "Ollama is not running at http://localhost:11434.\n"
            "Install Ollama: https://ollama.com/download\n"
            "Then run: ollama pull llama3.1:8b && ollama serve"
        )

    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        raise RuntimeError(
            "langchain-ollama is not installed.\n"
            "Run: pip install langchain-ollama>=0.3.0"
        )

    default_model = model_name or "llama3.1:8b"

    return ChatOllama(
        model=default_model,
        base_url=base_url,
        temperature=temperature,
    )
