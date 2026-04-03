import os

from langchain_openai import ChatOpenAI


def get_llm(
    model_name: str = "mistralai/mistral-small-3.1-24b-instruct",
    temperature: float = 0.3,
):
    """Return a ChatOpenAI instance pointing at OpenRouter.

    Set the OPENROUTER_API_KEY environment variable before use.
    You can optionally override the model via the OPENROUTER_MODEL env var.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. "
            "Get a key at https://openrouter.ai/keys and run:\n"
            "  export OPENROUTER_API_KEY='your-key-here'"
        )

    model_name = os.environ.get("OPENROUTER_MODEL", model_name)

    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
    )
