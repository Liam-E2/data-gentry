from dataclasses import dataclass
from typing import Optional

from sqlalchemy import Engine

from data_gentry.embeddings import EmbeddingSource
from data_gentry.retrieval.join import outer_join_scores, Joiner
from data_gentry.retrieval.score import fts_search, vector_search, VectorScoreConfig, FullTextScoreConfig
from data_gentry.retrieval.fuse import weighted_normalization, Fuser


@dataclass
class RetrievalResult:
    chunk_id: int
    content: str
    bm25_score: float
    bm25_score_normed: float
    cosine_similarity_score: float
    cosine_similarity_score_normed: float
    rank: int


def retrieve(
        engine: Engine,
        query: str,
        query_embedder: EmbeddingSource,
        vector_score_config: Optional[VectorScoreConfig] = None,
        fulltext_score_config: Optional[FullTextScoreConfig] = None,
        joiner: Optional[Joiner] = None,
        fuser: Optional[Fuser] = None
        ) -> list[RetrievalResult]:
    """
    Retrieve top-n records based on hybrid bm25 + cosine similarity.
    Args:
        engine: Sqlalchemy engine.
        query: String to search for.
        query_embedder: EmbeddingSource instance, used to get embedding for input query.
        vector_score_config: Optional, config for semantic search.
        fulltext_score_config: Optional, config for full-text search.
        joiner: Optional, strategy for joining search results. Default: outer join.
        fuser: Optional, strategy for generating final scores. Default: weighted normalization(limit=200, fts_weight=0.8).
    """
    # Defaults
    vector_score_config = vector_score_config or VectorScoreConfig()
    fulltext_score_config = fulltext_score_config or FullTextScoreConfig()
    joiner = joiner or outer_join_scores
    fuser = fuser or weighted_normalization(limit=200, fts_weight=0.8)

    query_embedding = query_embedder.get_embedding(query)

    with engine.begin() as conn:
        FullTextTable = fts_search(conn, query, fulltext_score_config)
        VectorTable = vector_search(conn, query_embedding, vector_score_config)
        JoinedScores = joiner(FullTextTable, VectorTable)
        result = conn.execute(fuser(JoinedScores)).fetchall()

    return [
        RetrievalResult(row[0], row[1], row[2], row[3], row[4], row[5], i)
        for i, row in enumerate(result)
    ]
