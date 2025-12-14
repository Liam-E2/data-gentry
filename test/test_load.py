from tempfile import NamedTemporaryFile

from sqlalchemy import select
from pytest import fixture

from src.data_gent.connection import get_sqlalchemy_engine
from src.data_gent.config import settings
from src.data_gent.load import load_document
from src.data_gent.embeddings import TestEmbeddingSource
from src.data_gent.chunking import SemchunkChunker
from src.data_gent.db_models import DocumentChunks


@fixture
def docfile():
    data = b"Arrange: set up data. Act: Act on data. Assert: validate behavior of action."
    file = NamedTemporaryFile()
    with open(file.name, 'wb') as f:
        f.write(data)
    return file


def test_load_db(docfile):
    eng = get_sqlalchemy_engine()

    load_document(eng, TestEmbeddingSource(), SemchunkChunker(chunk_size=8, overlap=0.0), docfile.name)

    with eng.connect() as conn:
        rows: list[DocumentChunks] = conn.execute(select(DocumentChunks)).fetchall()
        for row in rows:
            assert isinstance(row.chunk_id, int)
            assert isinstance(row.document_id, int)
            assert isinstance(row.content, str)
            assert isinstance(row.embedding, tuple)
            assert isinstance(row.embedding[0], float)
            assert len(row.embedding) == settings.vec_size
