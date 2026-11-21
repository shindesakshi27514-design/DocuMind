from google import genai
from google.genai import types


def answer_question(question: str, retrieved_chunks: list[dict],
                    api_key: str, model_name: str = "gemini-3.8-flash") -> str:
    if not retrieved_chunks:
        return "I could not find relevant information in the uploaded documents."

    context_parts = []
    for item in retrieved_chunks:
        context_parts.append(
            f"Source: {item['source']} | Page: {item['page']}\n{item['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
You are DocuMind AI, a careful document question-answering assistant.

Answer the user's question using ONLY the supplied document context.
- If the context does not contain the answer, say: "I couldn't find that information in the uploaded documents."
- Do not invent facts or claim that the documents say something they do not.
- Give a clear, useful answer.
- Mention relevant source filenames and page numbers in your answer when appropriate.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
"""
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=1200
        )
    )
    answer = getattr(response, "text", None)
    if not answer:
        raise RuntimeError("The AI service returned an empty response.")
    return answer.strip()
