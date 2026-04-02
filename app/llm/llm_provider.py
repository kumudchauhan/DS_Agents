from langchain_ollama import ChatOllama


def get_llm(model_name: str = "mistral", temperature: float = 0.3):
    """Return a ChatOllama instance pointing at a local Ollama server."""
    return ChatOllama(model=model_name, temperature=temperature)
