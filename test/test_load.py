from tempfile import NamedTemporaryFile
import os
from csv import DictWriter

from sqlalchemy import select, text
from pytest import fixture

from src.data_gent.connection import get_sqlalchemy_engine
from src.data_gent.settings import settings
from src.data_gent.load import load_document, load_data
from src.data_gent.embeddings import TestEmbeddingSource
from src.data_gent.chunking import SemchunkChunker
from src.data_gent.db_models import DocumentChunks


@fixture
def docfile():
    data = b"Arrange: set up data. Act: Act on data. Assert: validate behavior of action."
    file = NamedTemporaryFile()
    with open(file.name, 'wb') as f:
        f.write(data)
    
    with open(file.name, "rb") as f:
        yield f


@fixture
def csvfile():
    file = NamedTemporaryFile(suffix=".csv")
    with open(file.name, "w") as f:
        writer = DictWriter(f, ("c1", "c2", "c3"))
        writer.writeheader()
        writer.writerow({"c1": "a", "c2": 2, "c3": 3.1})

    return file
    

def test_load_db(docfile):
    eng = get_sqlalchemy_engine()
    try:
        load_document(eng, TestEmbeddingSource(), SemchunkChunker(chunk_size=8, overlap=0.0), docfile)

        with eng.connect() as conn:
            rows: list[DocumentChunks] = conn.execute(select(DocumentChunks)).fetchall()
            for row in rows:
                assert isinstance(row.chunk_id, int)
                assert isinstance(row.document_id, int)
                assert isinstance(row.content, str)
                assert isinstance(row.embedding, tuple)
                assert isinstance(row.embedding[0], float)
                assert len(row.embedding) == settings.vec_size
    except Exception:
        raise
    finally:
        os.remove(settings.db_path)


def test_load_csv(csvfile):
    eng = get_sqlalchemy_engine()

    table = load_data(eng, csvfile.name, "csv")

    with eng.begin() as conn:
        result = conn.execute(text(f"select * from {table}"))
        assert set(result.keys()) == {"c1", "c2", "c3"}
        
        data = result.fetchall()
        assert data[0][0] == "a"
        assert data[0][1] == 2
        assert data[0][2] == 3.1
   
    os.remove(settings.db_path)


def test_load_csv_with_opts(csvfile):
    eng = get_sqlalchemy_engine()
    try:
        table = load_data(eng, csvfile.name, "csv", opts={"header": True, "all_varchar": True})

        with eng.begin() as conn:
            result = conn.execute(text(f"select * from {table}"))
            assert set(result.keys()) == {"c1", "c2", "c3"}
            
            data = result.fetchall()
            assert data[0][0] == "a"
            assert data[0][1] == "2"
            assert data[0][2] == "3.1"
        
        os.remove(settings.db_path)

        table = load_data(eng, csvfile.name, "csv", opts={"columns": {"c1": "text", "c2": "float", "c3": "float"}})
        with eng.begin() as conn:
            result = conn.execute(text(f"select * from {table}"))
            assert set(result.keys()) == {"c1", "c2", "c3"}

            data = result.fetchall()
            assert data[0][0] == "a"
            assert isinstance(data[0][1], float)
            assert isinstance(data[0][2], float)
    
    except Exception:
        raise
    finally:
        os.remove(settings.db_path)


def test_filetype_inference(csvfile):
    eng = get_sqlalchemy_engine()
    try:
        name = load_data(eng, csvfile.name)
        with eng.begin() as conn:
            result = conn.execute(text(f"select * from {name}"))
            assert set(result.keys()) == {"c1", "c2", "c3"}

            data = result.fetchall()
            assert data[0][0] == "a"
            assert data[0][1] == 2
            assert data[0][2] == 3.1
    except Exception:
        raise
    finally:
        os.remove(settings.db_path)


def test_load_document_deduplicates_chunks(docfile):
    """Test that re-loading the same document doesn't insert duplicate chunks."""
    eng = get_sqlalchemy_engine()

    try:
        # First load - should insert all chunks
        load_document(eng, TestEmbeddingSource(), SemchunkChunker(chunk_size=8, overlap=0.0), docfile)

        with eng.connect() as conn:
            first_load_chunks = conn.execute(select(DocumentChunks)).fetchall()
            first_count = len(first_load_chunks)
            doc_id = first_load_chunks[0].document_id

        # Verify we actually got some chunks
        assert first_count > 0

        # Second load of same document - should skip all chunks
        load_document(eng, TestEmbeddingSource(), SemchunkChunker(chunk_size=8, overlap=0.0), docfile)

        with eng.connect() as conn:
            second_load_chunks = conn.execute(select(DocumentChunks)).fetchall()
            second_count = len(second_load_chunks)

            # Verify no new chunks were added
            assert second_count == first_count

            # Verify no duplicate content for the same document_id
            chunks_for_doc = [c for c in second_load_chunks if c.document_id == doc_id]
            contents = [c.content for c in chunks_for_doc]
            assert len(contents) == len(set(contents))
    except Exception:
        raise
    finally:
        os.remove(settings.db_path)


def test_httpfs_load():
    engine = get_sqlalchemy_engine({"httpfs"})

    try:
        load_data(engine, "https://jsonplaceholder.typicode.com/todos/1", "json", "test")
        with engine.connect() as conn:
            results = conn.execute(text("SELECT * FROM test;")).fetchall()
            assert len(results) == 1
            assert results[0][0] == 1
    except Exception:
        raise
    finally:
        os.remove(settings.db_path)
