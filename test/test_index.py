from sqlalchemy import select

from src.data_gent.connection import get_sqlalchemy_engine
from src.data_gent.config import settings
from src.data_gent.load import load_document
from src.data_gent.embeddings import TestEmbeddingSource
from src.data_gent.db_models import DocumentChunks


# This workaround works, but you have to be very careful not to open 2 conns at once with diff configs...
# If a duckdb cnxn is ever opened at the sametime as a sqlachemy cnxn, cnxn 2 will fail.
# This can sometimes be retried, but will require transaction rollbacks in cases...

def test_config_db(db_session):
    eng = get_sqlalchemy_engine()
    load_document(eng, TestEmbeddingSource(), "/home/liam/src/data-gent/test.txt")

    with eng.connect() as conn:
        print(conn.execute(select(DocumentChunks)).fetchall())
        
    assert False