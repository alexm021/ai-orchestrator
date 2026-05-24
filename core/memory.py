# =============================================================================
# core/memory.py — MemoryManager: Persistent Vector Memory
# =============================================================================
#
# PURPOSE:
# Give the orchestrator a long-term memory — every completed task is stored
# as a vector embedding in ChromaDB. Future tasks can retrieve semantically
# relevant past interactions.
#
# HOW IT WORKS:
# 1. SAVE: When a task completes, we create a text document from the task
#    message + output, embed it with Gemini text-embedding-004, and store
#    the embedding + metadata in ChromaDB.
#
# 2. RETRIEVE: When the router sets requires_memory=True (e.g., "What did
#    we discuss last time?"), the RetrievalAgent calls memory.search(),
#    which embeds the query and finds the most similar past tasks.
#
# WHY VECTOR EMBEDDINGS:
# Traditional search = keyword matching ("email" matches "email")
# Vector search = semantic matching ("cold outreach" matches "sales email")
#
# The embedding model (text-embedding-004) converts text into a 768-dimension
# vector where semantically similar texts are geometrically close.
# "Write a Python function" and "Code a Python script" will have vectors
# with cosine similarity > 0.8, even though they share no words.
#
# WHY CHROMADB:
# - Runs locally, no external service needed
# - Persistent storage (data survives restarts)
# - Fast vector similarity search (uses HNSW index)
# - Simple Python API
#
# WHY COSINE SIMILARITY:
# For text embeddings, cosine similarity measures the ANGLE between vectors,
# not their magnitude. This works better than Euclidean distance because
# it's invariant to text length — a short and long version of the same
# idea will still be "close" to each other.
# =============================================================================

import os
from datetime import datetime, timezone
from typing import Optional

import chromadb
from google import genai

from core.config import settings
from core.logger import get_logger


class MemoryManager:
    """
    Persistent vector memory backed by ChromaDB + Gemini embeddings.

    Usage:
        memory = get_memory_manager()
        await memory.save_task(task_id, message, output, agents)
        results = await memory.search("what was that email about Anthropic?")
    """

    def __init__(self):
        self.logger = get_logger("MemoryManager")

        # Ensure storage directory exists before ChromaDB tries to use it
        os.makedirs(settings.chroma_persist_dir, exist_ok=True)

        # PersistentClient: data is written to disk and survives restarts
        # vs. EphemeralClient: in-memory only, lost on restart
        self._chroma = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
        )

        # Get or create the collection (like a table in SQL)
        # hnsw:space = cosine tells ChromaDB to use cosine similarity
        # instead of the default L2 (Euclidean) distance
        self._collection = self._chroma.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Gemini client for generating embeddings
        self._gemini = genai.Client(api_key=settings.gemini_api_key)

        self.logger.info(
            "memory_initialized",
            collection=settings.chroma_collection_name,
            persist_dir=settings.chroma_persist_dir,
            existing_memories=self._collection.count(),
        )

    # =========================================================================
    # EMBEDDING
    # =========================================================================

    async def _embed(self, text: str) -> list[float]:
        """
        Convert text to a 768-dimensional vector using Gemini text-embedding-004.

        WHY text-embedding-004 (not a sentence-transformers model):
        - Consistent with our Gemini-first architecture
        - No additional dependencies or model downloads
        - Strong semantic understanding — trained on massive text corpus
        - 768 dimensions gives good precision without being too large
        """
        result = await self._gemini.aio.models.embed_content(
            model="models/gemini-embedding-001",
            contents=text,
        )
        return list(result.embeddings[0].values)

    # =========================================================================
    # SAVE
    # =========================================================================

    async def save_task(
        self,
        task_id: str,
        task_message: str,
        primary_output: str,
        agents_used: list[str],
        status: str = "complete",
    ) -> None:
        """
        Save a completed task to long-term memory.

        Called automatically from node_aggregate after every successful task.
        Over time this builds up a rich memory of what the orchestrator has done.

        The document we embed contains BOTH the input AND output:
        "Task: Research Anthropic...  Output: Anthropic is an AI safety..."
        This allows matching on either side — future queries about Anthropic
        OR about cold email writing will both find this memory.

        Memory failures are silently swallowed — they are NON-CRITICAL.
        A failed memory write should never crash the orchestration.

        Args:
            task_id: Unique ID (used as ChromaDB document ID — must be unique)
            task_message: The original user request
            primary_output: The main output text (first 800 chars)
            agents_used: Which agents handled this task
            status: Only "complete" tasks are saved
        """
        if status != "complete":
            return  # Don't pollute memory with failed tasks

        try:
            # Build rich document text combining input + output
            # WHY both: we want the memory to match future queries about
            # either the topic (input) or the kind of result (output)
            doc_text = (
                f"Task: {task_message}\n"
                f"Agents used: {', '.join(agents_used)}\n"
                f"Result: {primary_output[:800]}"
            )

            embedding = await self._embed(doc_text)

            self._collection.add(
                ids=[task_id],
                embeddings=[embedding],
                documents=[doc_text],
                metadatas=[{
                    "task_id": task_id,
                    "agents_used": ",".join(agents_used),
                    "status": status,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    # Store truncated versions for fast retrieval without re-embedding
                    "task_message": task_message[:500],
                    "primary_output": primary_output[:500],
                }],
            )

            self.logger.info(
                "memory_saved",
                task_id=task_id,
                agents=agents_used,
                total_memories=self._collection.count(),
            )

        except Exception as e:
            # Memory is BEST-EFFORT — log but never raise
            self.logger.error(
                "memory_save_failed",
                task_id=task_id,
                error=str(e),
            )

    # =========================================================================
    # RETRIEVE
    # =========================================================================

    async def search(
        self,
        query: str,
        n_results: int = 5,
        min_relevance: float = 0.3,
    ) -> list[dict]:
        """
        Find past tasks semantically similar to the query.

        Returns memories ranked by relevance, highest first.
        Low-relevance results (< min_relevance) are filtered out to avoid
        injecting irrelevant context.

        HOW RELEVANCE WORKS:
        ChromaDB returns cosine distance (0 = identical, 2 = opposite).
        We convert to similarity: sim = 1 - (distance / 2)
        So sim=1.0 means exact match, sim=0.0 means completely unrelated.
        We default to min_relevance=0.3 — anything less is noise.

        Args:
            query: The search string (usually the current task message)
            n_results: Maximum memories to return
            min_relevance: Similarity threshold (0 to 1)

        Returns:
            List of memory dicts, sorted by relevance descending
        """
        total = self._collection.count()
        if total == 0:
            self.logger.info("memory_search_empty", query_length=len(query))
            return []

        try:
            query_embedding = await self._embed(query)

            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, total),
                include=["documents", "metadatas", "distances"],
            )

            memories = []
            for i, meta in enumerate(results["metadatas"][0]):
                # Convert ChromaDB cosine distance to similarity score
                # distance=0 → sim=1.0 (identical)
                # distance=1 → sim=0.5 (orthogonal / unrelated)
                # distance=2 → sim=0.0 (opposite)
                distance = results["distances"][0][i]
                similarity = round(1.0 - (distance / 2.0), 3)

                if similarity < min_relevance:
                    continue

                memories.append({
                    "task_message": meta.get("task_message", ""),
                    "primary_output": meta.get("primary_output", ""),
                    "agents_used": meta.get("agents_used", "").split(","),
                    "timestamp": meta.get("timestamp", ""),
                    "relevance_score": similarity,
                })

            self.logger.info(
                "memory_search_completed",
                query_length=len(query),
                results_found=len(memories),
                total_memories=total,
            )

            return memories

        except Exception as e:
            self.logger.error("memory_search_failed", error=str(e))
            return []  # Fail gracefully — return empty list, not an exception

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def count(self) -> int:
        """Return total number of stored memories."""
        return self._collection.count()

    def clear(self) -> None:
        """
        Delete ALL memories and recreate the empty collection.
        USE WITH CAUTION — this is permanent.
        """
        self._chroma.delete_collection(settings.chroma_collection_name)
        self._collection = self._chroma.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.logger.warning("memory_cleared_all_data_deleted")


# =============================================================================
# SINGLETON — One MemoryManager for the whole application
# =============================================================================
#
# WHY SINGLETON:
# ChromaDB holds an open connection to the vector store on disk.
# Creating multiple MemoryManager instances would open multiple connections
# to the same store, which can cause write conflicts and inconsistencies.
#
# The singleton pattern ensures one connection, thread-safe by design
# (Python's GIL handles the global assignment).
# =============================================================================

_instance: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """
    Return the application-wide singleton MemoryManager.

    Creates it on the first call, returns the same instance thereafter.
    Safe to call from any agent or node.
    """
    global _instance
    if _instance is None:
        _instance = MemoryManager()
    return _instance
