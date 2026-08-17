import os

from dotenv import load_dotenv

load_dotenv(override=True)


def _get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing in backend/.env")

    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError("groq is not installed") from exc

    return Groq(api_key=api_key)


def generate_answer(question: str, context: str):
    client = _get_client()
    model = os.getenv("GROQ_CHAT_MODEL", "qwen/qwen3.6-27b")

    prompt = f"""
You are an AI Resume Assistant.
Answer the user's question using only the information
provided in the resume context.

Resume Context: {context}
Question: {question}

If the answer is not available in the resume context,
say that the information is not available in the resume.
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an AI Resume Assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
    except Exception as exc:
        raise RuntimeError(f"Answer generation failed: {exc}") from exc

    return response.choices[0].message.content or ""
