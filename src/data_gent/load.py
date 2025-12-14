from sqlalchemy import insert, Engine

from .db_models import Documents, DocumentChunks, INDEX_DDL
from .embeddings import EmbeddingSource
from .chunking import Chunker
from .config import settings


def create_vss_index(engine: Engine):
    with engine.connect() as conn:
        conn.execute(INDEX_DDL)
        conn.commit()
        conn.close()


def load_document(
        engine: Engine,
        embedding_source: EmbeddingSource,
        chunker: Chunker,
        file: str):
    """
    Load a text document into duckdb, break it into chunks, 
    create an embedding vector for each chunk, then
    create a HNSW index on the embedding vectors.
    """
    with open(file, "r") as f:
        text = f.read()

    stmt = (
        insert(Documents)
        .values(content=text)
        .returning(Documents.id)
    )

    with engine.begin() as conn:
        doc_id = conn.execute(stmt).scalar_one()
        chunks = [
            {
                "document_id": doc_id,
                "content": chunk,
                "embedding": embedding_source.get_embedding(chunk)
            }

            for chunk in chunker.chunk(text)
            ]

        conn.execute(insert(DocumentChunks).values(chunks))
    
    create_vss_index(engine)
