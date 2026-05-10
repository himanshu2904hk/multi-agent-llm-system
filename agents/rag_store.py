"""
Vector store setup using ChromaDB + LangChain.
Seeds a knowledge base of documents on first run.
"""
import os
import logging
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logger = logging.getLogger(__name__)

CHROMA_DIR = os.getenv("CHROMA_DIR", "/tmp/chroma_db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

# Knowledge base documents seeded into ChromaDB
SEED_DOCUMENTS = [
    Document(page_content="Python is a high-level, interpreted programming language created by Guido van Rossum. It was first released in 1991. Python emphasizes code readability and simplicity.", metadata={"source": "python_overview", "topic": "python"}),
    Document(page_content="Python supports multiple programming paradigms including procedural, object-oriented, and functional programming. It has a large standard library and active community.", metadata={"source": "python_paradigms", "topic": "python"}),
    Document(page_content="Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that access data and use it to learn for themselves.", metadata={"source": "ml_overview", "topic": "machine_learning"}),
    Document(page_content="Supervised learning is the most common type of machine learning. The algorithm learns from labeled training data and makes predictions. Examples include linear regression, decision trees, and neural networks.", metadata={"source": "ml_supervised", "topic": "machine_learning"}),
    Document(page_content="Deep learning is a subset of machine learning that uses neural networks with many layers (deep neural networks). It is particularly effective for image recognition, natural language processing, and speech recognition.", metadata={"source": "deep_learning", "topic": "machine_learning"}),
    Document(page_content="Docker is an open-source platform for developing, shipping, and running applications in containers. Containers allow developers to package an application with all its dependencies into a standardized unit.", metadata={"source": "docker_overview", "topic": "docker"}),
    Document(page_content="Docker Compose is a tool for defining and running multi-container Docker applications. With Compose, you use a YAML file to configure your application's services, networks, and volumes.", metadata={"source": "docker_compose", "topic": "docker"}),
    Document(page_content="FastAPI is a modern, fast web framework for building APIs with Python based on standard Python type hints. It is one of the fastest Python frameworks available, comparable to NodeJS and Go.", metadata={"source": "fastapi_overview", "topic": "fastapi"}),
    Document(page_content="FastAPI automatically generates OpenAPI documentation. It supports async programming with Python asyncio, making it highly performant for I/O-bound operations.", metadata={"source": "fastapi_features", "topic": "fastapi"}),
    Document(page_content="Large Language Models (LLMs) are neural networks trained on massive text datasets. They can generate human-like text, answer questions, write code, and perform many other language tasks.", metadata={"source": "llm_overview", "topic": "llm"}),
    Document(page_content="Prompt engineering is the practice of designing and optimizing input prompts to get the best outputs from language models. Techniques include chain-of-thought, few-shot prompting, and role assignment.", metadata={"source": "prompt_engineering", "topic": "llm"}),
    Document(page_content="RAG (Retrieval-Augmented Generation) combines information retrieval with language generation. It retrieves relevant documents from a knowledge base and uses them as context for generating answers.", metadata={"source": "rag_overview", "topic": "rag"}),
    Document(page_content="The Eiffel Tower is located in Paris, France. It was built by Gustave Eiffel for the 1889 World's Fair. It stands 330 meters tall and is one of the most visited monuments in the world.", metadata={"source": "eiffel_tower", "topic": "geography"}),
    Document(page_content="Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure (1 atm). At higher altitudes, water boils at lower temperatures due to reduced air pressure.", metadata={"source": "water_boiling", "topic": "science"}),
    Document(page_content="Python was created by Guido van Rossum and was first released on February 20, 1991. It was named after the British comedy series Monty Python's Flying Circus, not the snake.", metadata={"source": "python_history", "topic": "python"}),
    Document(page_content="PostgreSQL is a powerful, open-source object-relational database system with over 35 years of active development. It supports advanced data types and performance optimization features.", metadata={"source": "postgresql_overview", "topic": "database"}),
    Document(page_content="Redis is an in-memory data structure store used as a database, cache, and message broker. It supports data structures such as strings, hashes, lists, sets, sorted sets.", metadata={"source": "redis_overview", "topic": "database"}),
    Document(page_content="Server-Sent Events (SSE) is a server push technology enabling clients to receive automatic updates from a server via HTTP connection. It is one-directional from server to client.", metadata={"source": "sse_overview", "topic": "web"}),
    Document(page_content="Multi-agent systems consist of multiple interacting intelligent agents. Each agent perceives its environment and takes actions. Agents can cooperate, compete, or negotiate with each other.", metadata={"source": "multi_agent", "topic": "ai"}),
    Document(page_content="Context window management in LLMs refers to handling the maximum token limit of a model. Techniques include summarization, chunking, and sliding window approaches to fit information within limits.", metadata={"source": "context_window", "topic": "llm"}),
]

_vectorstore = None


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    logger.info("[rag_store] Initializing ChromaDB vector store...")
    embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)

    # Check if already seeded
    try:
        _vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name="mega_ai_kb",
        )
        count = _vectorstore._collection.count()
        if count == 0:
            raise ValueError("Empty collection")
        logger.info(f"[rag_store] Loaded existing ChromaDB with {count} documents")
    except Exception:
        logger.info("[rag_store] Seeding ChromaDB with knowledge base documents...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        chunks = splitter.split_documents(SEED_DOCUMENTS)
        _vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR,
            collection_name="mega_ai_kb",
        )
        logger.info(f"[rag_store] Seeded {len(chunks)} chunks into ChromaDB")

    return _vectorstore


def similarity_search(query: str, k: int = 4) -> list:
    """Return top-k relevant documents for a query."""
    store = get_vectorstore()
    results = store.similarity_search_with_score(query, k=k)
    return [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "topic": doc.metadata.get("topic", "general"),
            "score": float(score),
        }
        for doc, score in results
    ]
