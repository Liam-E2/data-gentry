from typing import Callable, Protocol, Type
from functools import wraps

from sqlalchemy import select, func, Select
from sqlalchemy.orm import Mapped, outerjoin, DeclarativeBase

from data_gent.retrieval.score import ScoredChunks


class Base(DeclarativeBase):
    pass


class JoinedScores(Protocol):
    __tablename__: str
    chunk_id: Mapped[int]
    content: Mapped[str]
    vector_score: Mapped[float]
    fts_score: Mapped[float]


Joiner = Callable[[Type[ScoredChunks], Type[ScoredChunks]], Type[JoinedScores]]


def joins_scores(fn: Callable[[Type[ScoredChunks], Type[ScoredChunks]], Select]) -> Joiner:
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
