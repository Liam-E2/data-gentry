from dataclasses import dataclass

from sqlalchemy import Engine, text

from .embeddings import EmbeddingSource
from .settings import settings


@dataclass
class RetrievalResult:
    chunk_id: int
    content: str
    bm25_score_normed: float
    cosine_similarity_score_normed: float
    rank: int


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

    # Ugly hack to pass FLOAT[N] & actually hit the HNSW index
    hnsw_query_text = text(f"""
    CREATE TEMPORARY TABLE hnsw_results AS
    SELECT
        chunk_id,
        content,
        array_cosine_similarity(embedding, :queryvec\\:\\:FLOAT[{settings.vec_size}]) as cosine_similarity
    FROM document_chunks
    ORDER BY array_cosine_similarity(embedding, :queryvec\\:\\:FLOAT[{settings.vec_size}])
    LIMIT :cosine_limit;
    """)

    final_query_text = text("""
    WITH bm25 AS (
        SELECT
            chunk_id,
            content,
            fts_main_document_chunks.match_bm25(chunk_id, :query, fields := 'content') AS bm25_score
        FROM document_chunks
        ORDER BY bm25_score
        LIMIT :fts_limit
    ), combined as (
        SELECT
            COALESCE(b.chunk_id, v.chunk_id) AS chunk_id,
            COALESCE(b.content, v.content) AS content,
            coalesce(b.bm25_score, 0) AS bm25_score,
            coalesce(v.cosine_similarity, 0) as cosine_similarity
        FROM bm25 b
        FULL OUTER JOIN hnsw_results v USING (chunk_id)
    )
    SELECT 
        chunk_id,
        content,
        bm25_score / MAX(bm25_score) OVER () as bm25_normed,
        cosine_similarity / MAX(cosine_similarity) OVER () as cosine_normed
    FROM combined
    ORDER BY :ftsweight * bm25_normed + (1 - :ftsweight) * cosine_normed
    DESC
    LIMIT :limit;
    """)

    with engine.begin() as conn:
        conn.execute(hnsw_query_text, {"queryvec": embedding_source.get_embedding(query), "cosine_limit": cosine_limit}).fetchall()
        result = conn.execute(final_query_text, {"query": query, "limit": limit, "ftsweight": fts_weight, "fts_limit": fts_limit}).fetchall()

    return [
        RetrievalResult(row[0], row[1], row[2], row[3], i)
        for i, row in enumerate(result)
    ]
