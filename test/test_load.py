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
    return file


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
