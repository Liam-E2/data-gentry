from dataclasses import dataclass
from typing import Callable, Protocol, Type
from functools import wraps

from sqlalchemy import Engine, Connection, text, Table, MetaData, Column, Integer, Float, Text, select, func, Select
from sqlalchemy.orm import Mapped, outerjoin, mapped_column, declarative_base, DeclarativeBase
from sqlalchemy.sql._typing import _HasClauseElement

from .utils import sanitized_uuid
from .embeddings import EmbeddingSource
from .settings import settings


@dataclass
class RetrievalResult:
    chunk_id: int
    content: str
    bm25_score: float
    bm25_score_normed: float
    cosine_similarity_score: float
    cosine_similarity_score_normed: float
    rank: int


class ScoredChunks(Protocol):
    __tablename__: str
    chunk_id: Mapped[int]
    content: Mapped[str]
    score: Mapped[float]


class JoinedScores(Protocol):
    __tablename__: str
    chunk_id: Mapped[int]
    content: Mapped[str]
    vector_score: Mapped[float]
    fts_score: Mapped[float]


class Base(DeclarativeBase):
    pass


def scores_chunks(fn: Callable[..., str]) -> Callable[..., Type[ScoredChunks]]:
    """
    Given a function that creates a scored temp table with signature ... -> table_name, returns a function
    ... -> Table, a SqlAlchemy table with columns chunk_id, content, and score.
    """
    @wraps(fn)
    def _inner(*args, **kwargs):
        table_name = fn(*args, **kwargs)
        attrs = {
            "__tablename__": table_name,
            "chunk_id": mapped_column(Integer, primary_key=True),
            "content": mapped_column(Text),
            "score": mapped_column(Float),
        }

        return type(table_name, (Base,), attrs)

    return _inner


def joins_scores(fn: Callable[[Type[ScoredChunks], Type[ScoredChunks]], Select]) -> Callable[[Type[ScoredChunks], Type[ScoredChunks]], Type[JoinedScores]]:
    @wraps(fn)
    def _inner(fts_table: Type[ScoredChunks], vector_table: Type[ScoredChunks]):
        result = fn(fts_table, vector_table).subquery()
        class Joined(Base):
            __table__ = result
            __mapper_args__ = {
                "primary_key": [result.c.chunk_id]
            }
            chunk_id: Mapped[int]
            content: Mapped[str]
            vector_score: Mapped[float]
            fts_score: Mapped[float]

        return Joined
    
    return _inner


@scores_chunks
def fts_search(
    conn: Connection,
    query: str,
    limit: int = 200,
) -> str:
    """
    Perform BM25 full-text search and store results in a temporary table.

    Args:
        conn: SQLAlchemy Connection from an active transaction/connection context
        query: Query string for BM25 search
        limit: Maximum number of results to retrieve

    Returns:
        Name of temporary table containing results (str)

    Temp table schema:
        - chunk_id (int): Primary identifier for the chunk
        - content (text): The chunk text content
        - score (float): Raw BM25 score (NOT normalized)
    """
    table_name = "fts_" + sanitized_uuid()

    query_sql = text(f"""
    CREATE TEMPORARY TABLE {table_name} AS
    SELECT
        chunk_id,
        content,
        fts_main_document_chunks.match_bm25(
            chunk_id,
            :query,
            fields := 'content'
        ) AS score
    FROM document_chunks
    ORDER BY score DESC
    LIMIT :limit;
    """)

    conn.execute(query_sql, {
        "query": query,
        "limit": limit
    })

    return table_name


@scores_chunks
def vector_search(
    conn: Connection,
    query_embedding: list[float],
    limit: int = 200,
) -> str:
    """
    Perform HNSW vector similarity search and store results in a temporary table.

    Args:
        conn: SQLAlchemy Connection from an active transaction/connection context
        query_embedding: Pre-computed embedding vector for the query
        limit: Maximum number of results to retrieve

    Returns:
        Name of temporary table containing results (str)

    Temp table schema:
        - chunk_id (int): Primary identifier for the chunk
        - content (text): The chunk text content
        - score (float): Raw cosine similarity score (NOT normalized)
    """
    table_name = "vss_" + sanitized_uuid()

    # Critical: Type casting to FLOAT[N] to hit HNSW index
    query = text(f"""
    CREATE TEMPORARY TABLE {table_name} AS
    SELECT
        chunk_id,
        content,
        array_cosine_similarity(
            embedding,
            :queryvec\\:\\:FLOAT[{settings.vec_size}]
        ) as score
    FROM document_chunks
    ORDER BY array_cosine_similarity(
        embedding,
        :queryvec\\:\\:FLOAT[{settings.vec_size}]
    ) DESC
    LIMIT :limit;
    """)

    conn.execute(query, {
        "queryvec": query_embedding,
        "limit": limit
    })

    return table_name


@joins_scores
def outer_join_scores(fts_table: Type[ScoredChunks], vector_table: Type[ScoredChunks]) -> Select:
    return select(
            func.coalesce(vector_table.chunk_id, fts_table.chunk_id).label("chunk_id"),
            func.coalesce(vector_table.content, fts_table.content).label("content"),
            func.coalesce(fts_table.score, 0.0001).label("fts_score"),
            func.coalesce(vector_table.score, -0.9999).label("vector_score")
        ).select_from(
            outerjoin(fts_table, vector_table, fts_table.chunk_id == vector_table.chunk_id, full=True)
        )


def _combine_weighted(
    conn: Connection,
    vector_table: Type[ScoredChunks],
    fts_table: Type[ScoredChunks],
    limit: int,
    fts_weight: float
) -> list[tuple]:
    """
    Combine vector and FTS results using weighted normalization.

    Internal helper function - not part of public API.

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
        result = _combine_weighted(conn, vector_table, fts_table, limit, fts_weight)

    return [
        RetrievalResult(row[0], row[1], row[2], row[3], row[4], row[5], i)
        for i, row in enumerate(result)
    ]
