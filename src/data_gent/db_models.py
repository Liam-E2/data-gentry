from datetime import datetime

from duckdb_engine.datatypes import BigInteger
from sqlalchemy import Sequence, Text, TIMESTAMP, func, ForeignKey, DDL
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.types import UserDefinedType

from .settings import settings


class FloatArray(UserDefinedType):
    def __init__(self, n):
        self.n = n

    def get_col_spec(self, **kw):
        return f"FLOAT[{self.n}]"


class BaseTable(DeclarativeBase):
    pass


class Documents(BaseTable):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column("id", BigInteger(), Sequence("document_id_sequence"), primary_key=True)
    table: Mapped[str] = mapped_column("table", Text(), server_default=None, nullable=True)
    content: Mapped[str] = mapped_column("content", Text())
    created_at: Mapped[datetime] = mapped_column("created_at", TIMESTAMP(), server_default=func.current_timestamp())


class DocumentChunks(BaseTable):
    __tablename__ = "document_chunks"

    chunk_id: Mapped[int] = mapped_column(BigInteger(), Sequence("chunk_id"), primary_key=True)
    document_id: Mapped[int] = mapped_column("document_id", BigInteger(), ForeignKey("documents.id"))
    content: Mapped[str] = mapped_column("content", Text())
    embedding: Mapped[tuple[float]] = mapped_column("embedding", FloatArray(settings.vec_size))


INDEX_DDL = DDL("""
                DROP INDEX IF EXISTS embeddings_hnsw_index;
                CREATE INDEX embeddings_hnsw_index ON document_chunks 
                USING HNSW (embedding)
                WITH (metric = 'cosine');

                PRAGMA create_fts_index('document_chunks', 'chunk_id', 'content', overwrite=1);
                """)
