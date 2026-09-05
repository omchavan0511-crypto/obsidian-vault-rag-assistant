from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.groq import Groq

SYSTEM_PROMPT = (
    "Answer only using the provided context from the user's notes. "
    "If the context does not contain the answer, say so plainly. "
    "After your answer, list the source note titles you used."
)


def get_query_engine(index: VectorStoreIndex, top_k: int = 4) -> RetrieverQueryEngine:
    llm = Groq(model="openai/gpt-oss-20b", system_prompt=SYSTEM_PROMPT)
    return index.as_query_engine(llm=llm, similarity_top_k=top_k)


def ask(query_engine: RetrieverQueryEngine, question: str):
    response = query_engine.query(question)
    sources = sorted({node.metadata.get("title", "unknown") for node in response.source_nodes})
    return str(response), sources
