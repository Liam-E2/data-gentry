from typing import Type, Callable

from sqlalchemy import Select, select, func

from data_gentry.retrieval.join import JoinedScores


Fuser = Callable[[Type[JoinedScores]], Select]


def weighted_normalization(
    limit: int,
    fts_weight: float
) -> Fuser:
    """
    Combine vector and FTS results using weighted normalization.

    Args:
        limit: Maximum number of results to return
        fts_weight: Weight for FTS scores (0.0 to 1.0)

    Returns:
        Executable SqlAlchemy select statement
    """
    if fts_weight > 1.0 or fts_weight < 0.0:
        raise ValueError("Invalid fts_weight: must satisfy 0.0 <= fts_weight <= 1.0")


    def _weighted_normalization(joined: Type[JoinedScores]) -> Select:
        bm25_normed = (joined.fts_score / func.max(joined.fts_score).over()).label("bm25_normed")
        cosine_normed = ((joined.vector_score + 1) / func.max(joined.vector_score + 1).over()).label("cosine_normed")

        combined = select(
            joined.chunk_id,
            joined.content,
            joined.fts_score.label("bm25_raw"),
            bm25_normed,
            joined.vector_score.label("cosine_raw"),
            cosine_normed
        ).select_from(joined).order_by(
            ((fts_weight * bm25_normed) + ((1-fts_weight)* cosine_normed)).desc()
        ).limit(limit)

        return combined

    return _weighted_normalization
