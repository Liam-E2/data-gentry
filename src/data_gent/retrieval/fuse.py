from dataclasses import dataclass
from typing import Type

from sqlalchemy import Engine, Connection, select, func

from data_gent.embeddings import EmbeddingSource
from data_gent.retrieval.join import outer_join_scores
from data_gent.retrieval.score import fts_search, vector_search, ScoredChunks


@dataclass
class RetrievalResult:
    chunk_id: int
    content: str
    bm25_score: float
    bm25_score_normed: float
    cosine_similarity_score: float
    cosine_similarity_score_normed: float
    rank: int


def combine_weighted(
    conn: Connection,
    vector_table: Type[ScoredChunks],
    fts_table: Type[ScoredChunks],
    limit: int,
    fts_weight: float
) -> list[tuple]:
    """
    Combine vector and FTS results using weighted normalization.

    Args:
        conn: SQLAlchemy Connection
        vector_table: Name of temp table with vector search results
        fts_table: Name of temp table with FTS search results
        limit: Maximum number of results to return
        fts_weight: Weight for FTS scores (0.0 to 1.0)

    Returns:
        List of tuples: (chunk_id, content, bm25_raw, bm25_normed, cosine_raw, cosine_normed)
    """
    joined = outer_join_scores(fts_table, vector_table)
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

    results = conn.execute(combined, {
        "limit": limit,
        "ftsweight": fts_weight
    }).fetchall()

    return [tuple(row) for row in results]


def retrieve(
        engine: Engine,
        query: str,
        embedding_source: EmbeddingSource,
        limit: int,
        cosine_limit: int = 200,
        fts_limit: int = 200,
        fts_weight: float = 0.8) -> list[RetrievalResult]:
    """
    Retrieve top-n records based on bm25 + cosine similarity.

    NOTE: to actually hit the index, has to be computed seperately - the duckdb extension
    doesn't recognize subqueries/window functions/etc. that could be accelerated.
    """
    query_embedding = embedding_source.get_embedding(query)

    with engine.begin() as conn:
        vector_table = vector_search(conn, query_embedding, cosine_limit)
        fts_table = fts_search(conn, query, fts_limit)
        result = combine_weighted(conn, vector_table, fts_table, limit, fts_weight)

    return [
        RetrievalResult(row[0], row[1], row[2], row[3], row[4], row[5], i)
        for i, row in enumerate(result)
    ]
