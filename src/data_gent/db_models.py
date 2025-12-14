from datetime import datetime
from dataclasses import dataclass

from duckdb_engine.datatypes import Struct, BigInteger
from sqlalchemy import Sequence, Text, TIMESTAMP, ARRAY, func, ForeignKey, Float
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column


class BaseTable(DeclarativeBase):
    pass


class Documents(BaseTable):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column("id", BigInteger(), Sequence("document_id_sequence"), primary_key=True)
    content: Mapped[str] = mapped_column("content", Text())
    created_at: Mapped[datetime] = mapped_column("created_at", TIMESTAMP(), server_default=func.current_timestamp())


class DocumentChunks(BaseTable):
    __tablename__ = "document_chunks"

    chunk_id: Mapped[int] = mapped_column(BigInteger(), Sequence("chunk_id"), primary_key=True)
    document_id: Mapped[int] = mapped_column("document_id", BigInteger(), ForeignKey("documents.id"))
    content: Mapped[str] = mapped_column("content", Text())
    start_pos: Mapped[int] = mapped_column("start_pos", BigInteger())
    end_pos: Mapped[int] = mapped_column("end_pos", BigInteger())
    embedding: Mapped[list[float]] = mapped_column("embedding", ARRAY(Float))
