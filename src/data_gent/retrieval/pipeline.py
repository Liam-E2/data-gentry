from dataclasses import dataclass

from sqlalchemy import Engine

from data_gent.embeddings import EmbeddingSource
from data_gent.retrieval.join import outer_join_scores, Joiner
from data_gent.retrieval.score import fts_search, vector_search, VectorScoreConfig, FullTextScoreConfig
from data_gent.retrieval.fuse import weighted_normalization, Fuser


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
        embedding_source: EmbeddingSource,
        vector_score_config: VectorScoreConfig = VectorScoreConfig(),
        fulltext_score_config: FullTextScoreConfig = FullTextScoreConfig(),
        joiner: Joiner = outer_join_scores,
        fuser: Fuser = weighted_normalization(limit=200, fts_weight=0.8)
        ) -> list[RetrievalResult]:
    """
    Retrieve top-n records based on bm25 + cosine similarity.
    """
    query_embedding = embedding_source.get_embedding(query)

    with engine.begin() as conn:
        FullTextTable = fts_search(conn, query, fulltext_score_config)
        VectorTable = vector_search(conn, query_embedding, vector_score_config)
        JoinedScores = joiner(FullTextTable, VectorTable)
        result = conn.execute(fuser(JoinedScores)).fetchall()

    return [
        RetrievalResult(row[0], row[1], row[2], row[3], row[4], row[5], i)
        for i, row in enumerate(result)
    ]
