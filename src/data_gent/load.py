import duckdb
from sqlalchemy import insert, Engine


from .db_models import Documents, DocumentChunks
from .embeddings import EmbeddingSource
from .chunking import Chunker
from .config import settings


def create_vss_index(db_path: str, config: dict = {}, install: bool = False):
    config["hnsw_enable_experimental_persistence"] = True
    conn = duckdb.connect(db_path, config=config)
    if install:
        conn.execute("INSTALL vss;")
    conn.execute("LOAD vss;")

    conn.execute(f"DROP INDEX IF EXISTS embeddings_hnsw_index;")
    conn.execute(f"CREATE INDEX embeddings_hnsw_index ON document_chunks USING HNSW (embedding);")
    conn.commit()
    conn.close()


def load_document(
        engine: Engine,
        embedding_source: EmbeddingSource,
        chunker: Chunker,
        file: str):
    
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

        stmt = insert(DocumentChunks).values(chunks)
        conn.execute(stmt)
        conn.commit()
    
    create_vss_index(settings.db_path, install=True)
