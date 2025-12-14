from datetime import datetime

from src.data_gent.db import Documents,  DocumentChunks


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
        "embedding": [1.0, 2.0, 3.0],
        "start_pos": 0,
        "end_pos": 3
    }
    db_session.add(DocumentChunks(**test_embedding))
    db_session.commit()

    chunk = db_session.query(DocumentChunks).one()
    assert chunk.chunk_id == 1
    assert chunk.document_id == 1
    assert chunk.content == "test"
    assert chunk.embedding == [1.0, 2.0, 3.0]
    assert chunk.start_pos == 0
    assert chunk.end_pos == 3
