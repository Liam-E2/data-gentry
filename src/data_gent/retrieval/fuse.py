from dataclasses import dataclass
from typing import Type, Callable, TypeVar, Concatenate, ParamSpec
from functools import wraps

from sqlalchemy import Engine, Connection, Select, select, func

from data_gent.embeddings import EmbeddingSource
from data_gent.retrieval.join import outer_join_scores, Joiner, JoinedScores
from data_gent.retrieval.score import fts_search, vector_search, VectorScorer, FullTextScorer
from data_gent.retrieval.util import with_requested_kwargs


@dataclass
class RetrievalResult:
    chunk_id: int
    content: str
    bm25_score: float
    bm25_score_normed: float
    cosine_similarity_score: float
    cosine_similarity_score_normed: float
    rank: int


Fuser = Callable[[Type[JoinedScores]], Select]


def weighted_normalization(
    joined: Type[JoinedScores],
    limit: int = 200,
    fts_weight: float = 0.8
) -> Select:
    """
    Combine vector and FTS results using weighted normalization.

    Args:
        conn: SQLAlchemy Connection
        vector_table: Name of temp table with vector search results
        fts_table: Name of temp table with FTS search results
        limit: Maximum number of results to return
        fts_weight: Weight for FTS scores (0.0 to 1.0)

    Returns:
        Executably SqlAlchemy select statement
    """
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


def retrieve(
        engine: Engine,
        query: str,
        embedding_source: EmbeddingSource,
        vector_scorer: VectorScorer = vector_search,
        fulltext_scorer: FullTextScorer = fts_search,
        joiner: Joiner = outer_join_scores,
        fuser: Fuser = weighted_normalization,
        **kwargs) -> list[RetrievalResult]:
    """
    Retrieve top-n records based on bm25 + cosine similarity.
    """
    query_embedding = embedding_source.get_embedding(query)

    with engine.begin() as conn:
        FullTextTable = with_requested_kwargs(fulltext_scorer)(conn, query, **kwargs)
        VectorTable = with_requested_kwargs(vector_scorer)(conn, query_embedding, **kwargs)
        JoinedScores = with_requested_kwargs(joiner)(FullTextTable, VectorTable, **kwargs)
        result = conn.execute(with_requested_kwargs(fuser)(JoinedScores, **kwargs)).fetchall()

    return [
        RetrievalResult(row[0], row[1], row[2], row[3], row[4], row[5], i)
        for i, row in enumerate(result)
    ]
