from sqlalchemy import select

from src.data_gent.connection import get_sqlalchemy_engine
from src.data_gent.config import settings
from src.data_gent.load import load_document
from src.data_gent.embeddings import TestEmbeddingSource
from src.data_gent.chunking import SemchunkChunker
from src.data_gent.db_models import DocumentChunks


def test_load_db(db_session):
    eng = get_sqlalchemy_engine()

    load_document(eng, TestEmbeddingSource(), SemchunkChunker(), "/home/liam/src/data-gent/Dockerfile")

    with eng.connect() as conn:
        rows = conn.execute(select(DocumentChunks.content)).fetchall()
        print([row.content for row in rows])
        
    assert False
