from dataclasses import dataclass
from sqlalchemy import Engine, Connection, text

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


def _combine_weighted(
    conn: Connection,
    vector_table: str,
    fts_table: str,
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
    combine_query = text(f"""
    WITH combined AS (
        SELECT
            COALESCE(v.chunk_id, f.chunk_id) AS chunk_id,
            COALESCE(v.content, f.content) AS content,
            COALESCE(f.score, 0.0001) AS fts_score,
            COALESCE(v.score, -0.999) AS vector_score
        FROM {fts_table} f
        FULL OUTER JOIN {vector_table} v USING (chunk_id)
    )
    SELECT
        chunk_id,
        content,
        fts_score as bm25_raw,
        fts_score / MAX(fts_score) OVER () as bm25_normed,
        vector_score as cosine_raw,
        (vector_score + 1) / MAX(vector_score + 1) OVER () as cosine_normed
    FROM combined
    ORDER BY :ftsweight * bm25_normed + (1 - :ftsweight) * cosine_normed DESC
    LIMIT :limit;
    """)

    results = conn.execute(combine_query, {
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
