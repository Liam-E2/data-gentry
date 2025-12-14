from datetime import datetime

from src.data_gent.db_models import Documents,  DocumentChunks

from src.data_gent.config import settings

def test_create_documents_table(db_session):
    db_session.add(Documents(content="test"))
    db_session.commit()

    first = db_session.query(Documents).one()
    assert first.id == 1
    assert first.content == "test"
    assert isinstance(first.created_at, datetime)

    db_session.add(Documents(content="test2"))
    db_session.commit()

    second = db_session.query(Documents).filter(Documents.id == 2).one()
    assert second.id == 2
    assert second.content == "test2"
    assert second.created_at > first.created_at


def test_chunks_table(db_session):
    db_session.add(Documents(content="test"))
    db_session.commit()

    test_embedding = {
        "document_id": 1,
        "content": "test",
        "embedding": [1.0 for i in range(settings.vec_size)]
    }
    db_session.add(DocumentChunks(**test_embedding))
    db_session.commit()

    chunk = db_session.query(DocumentChunks).one()
    assert chunk.chunk_id == 1
    assert chunk.document_id == 1
    assert chunk.content == "test"
    assert chunk.embedding == tuple([1.0 for i in range(settings.vec_size)])
