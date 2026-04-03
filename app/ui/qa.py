"""Natural-language Q&A over the uploaded dataset using Ollama / Mistral."""

import pandas as pd

from app.llm.llm_provider import get_llm


def ask_question_about_data(question: str, df: pd.DataFrame) -> str:
    """Send *question* along with dataframe context to the local LLM.

    Context includes the schema, descriptive statistics, and five sample rows
    so the model can reason about the data without receiving the full dataset.

    Returns the LLM's answer as a string, or a user-friendly error message if
    the Ollama server is unreachable.
    """
    schema_info = "\n".join(
        f"  {col}: {dtype}" for col, dtype in df.dtypes.items()
    )
    describe_stats = df.describe(include="all").to_string()
    sample_rows = df.head(5).to_string(index=False)

    prompt = (
        "You are a helpful data analyst assistant. A user uploaded a dataset "
        "and wants to ask questions about it.\n\n"
        f"**Column schema:**\n{schema_info}\n\n"
        f"**Descriptive statistics:**\n{describe_stats}\n\n"
        f"**Sample rows (first 5):**\n{sample_rows}\n\n"
        f"**User question:** {question}\n\n"
        "Answer concisely using the information above. If you cannot answer "
        "from the provided context, say so clearly."
    )

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        return response.content
    except Exception as exc:
        return (
            f"Could not reach the Ollama server.\n\n"
            f"Make sure Ollama is running (`ollama serve`) and the Mistral "
            f"model is pulled (`ollama pull mistral`).\n\n"
            f"Error details: {exc}"
        )
