from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.groq import Groq

NO_ANSWER_PHRASE = "I don't have information about this in the provided notes."

SYSTEM_PROMPT = (
    "Answer only using the provided context from the user's notes. "
    f"If the context does not contain the answer, respond with exactly: \"{NO_ANSWER_PHRASE}\" and nothing else. "
    "When the answer includes mathematical equations, format them using LaTeX: "
    "wrap inline math in single dollar signs like $x^2$ and display equations in double dollar signs like $$E=mc^2$$. "
    "After your answer, list the source note titles you used, unless you gave the no-information response above."
)


def get_query_engine(index: VectorStoreIndex, top_k: int = 4) -> RetrieverQueryEngine:
    llm = Groq(model="openai/gpt-oss-20b", system_prompt=SYSTEM_PROMPT)
    return index.as_query_engine(llm=llm, similarity_top_k=top_k)


def ask(query_engine: RetrieverQueryEngine, question: str):
    response = query_engine.query(question)
    answer = str(response)

    if NO_ANSWER_PHRASE in answer:
        return NO_ANSWER_PHRASE, []

    sources = sorted({node.metadata.get("title", "unknown") for node in response.source_nodes})
    return answer, sources