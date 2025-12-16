import os
from sqlalchemy import insert, Engine, text, select
from enum import StrEnum, auto

from .db_models import Documents, DocumentChunks, INDEX_DDL
from .embeddings import EmbeddingSource
from .chunking import Chunker
from .settings import settings


def create_chunk_indices(engine: Engine):
    with engine.connect() as conn:
        conn.execute(INDEX_DDL)
        conn.commit()
        conn.close()


def load_document(
        engine: Engine,
        embedding_source: EmbeddingSource,
        chunker: Chunker,
        file: str,
        table_name: str | None = None):
    """
    Load a text document into duckdb, break it into chunks, 
    create an embedding vector for each chunk, then
    create a HNSW index on the embedding vectors.
    """
    with open(file, "r") as f:
        text = f.read()

    with engine.begin() as conn:
        existing_doc = conn.execute(
            select(Documents.id)
            .where(Documents.content == text)
            .where(Documents.table == table_name)
        ).fetchone()

        if existing_doc:
            doc_id = existing_doc.id
        else:
            stmt = (
                insert(Documents)
                .values(content=text, table=table_name)
                .returning(Documents.id)
            )
            doc_id = conn.execute(stmt).scalar_one()

        # Only insert new chunks that don't already exist for this document
        existing_chunks = conn.execute(
            select(DocumentChunks.content)
            .where(DocumentChunks.document_id == doc_id)
        ).fetchall()
        existing_content = {row.content for row in existing_chunks}

        chunks = []
        for chunk in chunker.chunk(text):
            if chunk not in existing_content:
                chunks.append({
                    "document_id": doc_id,
                    "content": chunk,
                    "embedding": embedding_source.get_embedding(chunk)
                })

        if chunks:
            conn.execute(insert(DocumentChunks).values(chunks))
    
    create_chunk_indices(engine)


class InputType(StrEnum):
    CSV = auto()
    PARQUET = auto()
    JSON = auto()


def load_data(
        engine: Engine, 
        path: str, 
        input_type: InputType | str | None = None,
        table_name: str | None = None,
        opts: dict = dict()) -> str:
    """
    Load data file into duckdb, returning the created table name.
    opts corresponds directly to duckdb's read_{type} arguments.
    """
    if table_name is None:
        table_name = os.path.basename(path).split(".")[0]
    
    if input_type == None:
        for t in InputType:
            if path.endswith(t):
                input_type = t
                break
        else:
            raise ValueError(f"Unknown input type; must specify one of {InputType._member_names_}")
    
    opts_string = ", " + ", ".join(f"{k} = {v}" for k, v in opts.items()) if opts else ""
    match input_type:
        case InputType.PARQUET: select = f"SELECT * From read_parquet(:path{opts_string});"
        case InputType.CSV: select = f"SELECT * From read_csv(:path{opts_string});"
        case InputType.JSON: select = f"SELECT * FROM read_json(:path{opts_string});"
        case _: raise ValueError(f"Unknown input type; must specify one of {InputType._member_names_}")
    
    stmt = text(f"CREATE TABLE IF NOT EXISTS {table_name} AS {select}")
    with engine.begin() as conn:
        conn.execute(stmt, {"path": path})
    
    return table_name
