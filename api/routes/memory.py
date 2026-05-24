# =============================================================================
# api/routes/memory.py — GET /memories + POST /memories/search
# =============================================================================
#
# These endpoints expose the vector memory store to API clients.
#
# WHY EXPOSE MEMORY VIA API:
# - Dashboard: show users what the system has learned
# - Debugging: check if a task was saved correctly
# - Search: find relevant past tasks programmatically
# - Admin: clear old memories when needed
# =============================================================================

from fastapi import APIRouter, Depends, Query, HTTPException, Request, status
from api.auth import require_api_key
from api.limiter import limiter
from api.schemas import MemoriesResponse, MemoryRecord, MemorySearchRequest
from core.memory import get_memory_manager
from core.logger import get_logger

router = APIRouter()
logger = get_logger("api.memory")


@router.get(
    "/memories",
    response_model=MemoriesResponse,
    summary="List stored memories",
    description=(
        "Returns memories from the ChromaDB vector store. "
        "These are past tasks that were auto-saved after completion. "
        "Useful for debugging, auditing, and understanding what the system has learned."
    ),
)
@limiter.limit("30/minute")
async def list_memories(
    request: Request,
    limit: int = Query(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of memories to return",
    ),
    _: None = Depends(require_api_key),
) -> MemoriesResponse:
    """
    List recent memories from ChromaDB.

    Returns the most recently stored task memories up to `limit`.
    Note: ChromaDB doesn't support native pagination — we fetch all and slice.
    For large stores, use POST /memories/search with a relevant query instead.
    """
    try:
        memory = get_memory_manager()
        total = memory.count()

        if total == 0:
            return MemoriesResponse(total=0, memories=[])

        # ChromaDB doesn't have a "list all" with sorting out of the box.
        # We use a workaround: get all items and return them.
        # For production, we'd add pagination via offset IDs.
        collection = memory._collection
        results = collection.get(
            limit=limit,
            include=["metadatas"],
        )

        memories = []
        for meta in results.get("metadatas", []):
            agents_raw = meta.get("agents_used", "")
            agents = agents_raw.split(",") if agents_raw else []
            memories.append(MemoryRecord(
                task_message=meta.get("task_message", ""),
                primary_output=meta.get("primary_output", ""),
                agents_used=agents,
                timestamp=meta.get("timestamp", ""),
                relevance_score=None,  # Not a search result, no relevance score
            ))

        # Sort by timestamp descending (newest first)
        memories.sort(key=lambda m: m.timestamp, reverse=True)

        logger.info("api_memories_listed", total=total, returned=len(memories))

        return MemoriesResponse(total=total, memories=memories)

    except Exception as e:
        logger.error("api_memories_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "memory_error", "message": str(e)},
        )


@router.post(
    "/memories/search",
    response_model=MemoriesResponse,
    summary="Semantic search over memories",
    description=(
        "Find past tasks semantically similar to your query. "
        "Uses vector similarity — 'cold email to tech company' will match "
        "'outreach to Tesla' even with no shared keywords."
    ),
)
@limiter.limit("20/minute")
async def search_memories(request: Request, body: MemorySearchRequest, _: None = Depends(require_api_key)) -> MemoriesResponse:
    """
    Semantic similarity search over stored memories.

    Embeds the query using Gemini, then finds the nearest vectors in ChromaDB.
    Returns results ranked by relevance (highest first).
    """
    try:
        memory = get_memory_manager()
        total = memory.count()

        results = await memory.search(
            query=body.query,
            n_results=body.n_results,
        )

        memories = [
            MemoryRecord(
                task_message=r["task_message"],
                primary_output=r["primary_output"],
                agents_used=r["agents_used"],
                timestamp=r["timestamp"],
                relevance_score=r["relevance_score"],
            )
            for r in results
        ]

        logger.info(
            "api_memory_search",
            query_length=len(request.query),
            results=len(memories),
        )

        return MemoriesResponse(total=total, memories=memories)

    except Exception as e:
        logger.error("api_memory_search_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "search_error", "message": str(e)},
        )
