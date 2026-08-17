import os

from dotenv import load_dotenv

load_dotenv()


def _get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError("google-genai is not installed") from exc

    return genai.Client(api_key=api_key)


def generate_answer(question: str, context: str):
    client = _get_client()
    prompt = f"""
You are an AI Resume Assistant.
Answer the user's question using only the information
provided in the resume context.

Resume Context: {context}
Question: {question}

If the answer is not available in the resume context,
say that the information is not available in the resume.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text
