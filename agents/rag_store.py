"""
Vector store using ChromaDB with default embeddings (no PyTorch needed).
Seeds a knowledge base on first run.
"""
import os
import logging
import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

CHROMA_DIR = os.getenv("CHROMA_DIR", "/tmp/chroma_db")

SEED_DOCUMENTS = [
    {"id": "py1", "text": "Python is a high-level, interpreted programming language created by Guido van Rossum, first released in 1991. It emphasizes readability and simplicity.", "topic": "python"},
    {"id": "py2", "text": "Python supports multiple paradigms: procedural, object-oriented, and functional. It has a vast standard library and active community.", "topic": "python"},
    {"id": "py3", "text": "Python was named after Monty Python's Flying Circus, not the snake. Guido van Rossum created it in the late 1980s.", "topic": "python"},
    {"id": "ml1", "text": "Machine learning is a subset of AI that enables systems to learn from data without being explicitly programmed. It includes supervised, unsupervised, and reinforcement learning.", "topic": "ml"},
    {"id": "ml2", "text": "Deep learning uses neural networks with many layers. It excels at image recognition, NLP, and speech recognition tasks.", "topic": "ml"},
    {"id": "ml3", "text": "Supervised learning trains on labeled data. Examples: linear regression, decision trees, neural networks, SVMs.", "topic": "ml"},
    {"id": "dk1", "text": "Docker is an open-source platform for containerizing applications. Containers package code and dependencies together for consistent deployment.", "topic": "docker"},
    {"id": "dk2", "text": "Docker Compose defines multi-container applications using a YAML file to configure services, networks, and volumes.", "topic": "docker"},
    {"id": "fa1", "text": "FastAPI is a modern Python web framework for building APIs with automatic OpenAPI docs generation, type hints, and async support.", "topic": "fastapi"},
    {"id": "fa2", "text": "FastAPI is one of the fastest Python frameworks, comparable to NodeJS and Go, built on Starlette and Pydantic.", "topic": "fastapi"},
    {"id": "llm1", "text": "Large Language Models (LLMs) are neural networks trained on massive text corpora. They generate human-like text and perform language tasks.", "topic": "llm"},
    {"id": "llm2", "text": "Prompt engineering optimizes LLM inputs. Techniques: chain-of-thought, few-shot, role assignment, RAG.", "topic": "llm"},
    {"id": "rag1", "text": "RAG (Retrieval-Augmented Generation) combines information retrieval with generation. It fetches relevant documents and uses them as context for answers.", "topic": "rag"},
    {"id": "geo1", "text": "The Eiffel Tower is located in Paris, France. Built by Gustave Eiffel for the 1889 World's Fair, standing 330 meters tall.", "topic": "geography"},
    {"id": "sci1", "text": "Water boils at 100 degrees Celsius (212°F) at standard atmospheric pressure (1 atm). At higher altitudes it boils at lower temperatures.", "topic": "science"},
    {"id": "db1", "text": "PostgreSQL is an open-source object-relational database with 35+ years of development. Supports advanced data types and ACID compliance.", "topic": "database"},
    {"id": "db2", "text": "Redis is an in-memory data structure store used as database, cache, and message broker. Supports strings, hashes, lists, sets.", "topic": "database"},
    {"id": "web1", "text": "Server-Sent Events (SSE) enables servers to push updates to clients over HTTP. It is one-directional: server to client only.", "topic": "web"},
    {"id": "ai1", "text": "Multi-agent systems consist of multiple interacting intelligent agents. Agents can cooperate, compete, or negotiate to achieve goals.", "topic": "ai"},
    {"id": "ctx1", "text": "Context window management handles LLM token limits via summarization, chunking, and sliding window techniques.", "topic": "llm"},
]

_client = None
_collection = None


def get_collection():
    global _client, _collection
    if _collection is not None:
        return _collection

    logger.info("[rag_store] Initializing ChromaDB...")
    _client = chromadb.PersistentClient(path=CHROMA_DIR)
    ef = embedding_functions.DefaultEmbeddingFunction()

    try:
        _collection = _client.get_collection("mega_ai_kb", embedding_function=ef)
        count = _collection.count()
        if count == 0:
            raise ValueError("Empty")
        logger.info(f"[rag_store] Loaded ChromaDB with {count} docs")
    except Exception:
        logger.info("[rag_store] Seeding ChromaDB knowledge base...")
        _collection = _client.get_or_create_collection("mega_ai_kb", embedding_function=ef)
        _collection.add(
            ids=[d["id"] for d in SEED_DOCUMENTS],
            documents=[d["text"] for d in SEED_DOCUMENTS],
            metadatas=[{"topic": d["topic"]} for d in SEED_DOCUMENTS],
        )
        logger.info(f"[rag_store] Seeded {len(SEED_DOCUMENTS)} documents")

    return _collection


def similarity_search(query: str, k: int = 4) -> list:
    col = get_collection()
    results = col.query(query_texts=[query], n_results=k)
    docs = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    ids = results["ids"][0]
    return [
        {
            "id": ids[i],
            "content": docs[i],
            "source": f"chromadb:{ids[i]}",
            "topic": metadatas[i].get("topic", "general"),
            "score": distances[i],
        }
        for i in range(len(docs))
    ]
