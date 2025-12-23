"""
Defines interfaces and implementations for:
  - Scoring: rank document chunks based on an input query => ScoredChunks temp table
  - Joining: Join together 2 ScoredChunks
"""
from dataclasses import dataclass
from typing import Callable, Protocol, Type, ParamSpec
from functools import wraps

from sqlalchemy import Engine, Connection, text, Integer, Float, Text
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase

from data_gent.utils import sanitized_uuid
from data_gent.settings import settings


P = ParamSpec("P")


class Base(DeclarativeBase):
    pass


class ScoredChunks(Protocol):
    """
    Protocol class defining a temp table holding scoring results
    (for compatibility with SQLAlchemy/duckdb extension quirks)
    """
    __tablename__: str
    chunk_id: Mapped[int]
    content: Mapped[str]
    score: Mapped[float]


@dataclass(frozen=True)
class VectorScoreConfig:
    limit: int = 100
    

@dataclass(frozen=True)
class FullTextScoreConfig:
    limit: int = 100
    conjunctive: bool = False # All terms must be present to match
    # bm25 params
    b: float = 0.75
    k: float = 1.2 


VectorScorer = Callable[[Connection, list[float], VectorScoreConfig], Type[ScoredChunks]]
FullTextScorer = Callable[[Connection, str, FullTextScoreConfig], Type[ScoredChunks]]


def scores_chunks(fn: Callable[P, str]) -> Callable[P, Type[ScoredChunks]]:
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


@scores_chunks
def fts_search(
    conn: Connection,
    query: str,
    config: FullTextScoreConfig,
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
            fields := 'content',
            conjunctive := :conjunctive,
            b := :b,
            k := :k
        ) AS score
    FROM document_chunks
    ORDER BY score DESC
    LIMIT :limit;
    """)

    conn.execute(query_sql, {
        "query": query,
        "limit": config.limit,
        "conjunctive": config.conjunctive,
        "b": config.b,
        "k": config.k
    })

    return table_name


@scores_chunks
def vector_search(
    conn: Connection,
    query_vec: list[float],
    config: VectorScoreConfig
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

    # Nasty hack: Type casting to FLOAT[N] and using temp table to hit HNSW index
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
        "queryvec": query_vec,
        "limit": config.limit
    })

    return table_name
