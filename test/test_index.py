import duckdb
from pytest import raises
from sqlalchemy.exc import OperationalError

from src.data_gent.config import settings
from src.data_gent.load import config_db, create_vss_index
from src.data_gent.db_models import Documents, DocumentChunks


# This workaround works, but you have to be very careful not to open 2 conns at once with diff configs...
# If a duckdb cnxn is ever opened at the sametime as a sqlachemy cnxn, cnxn 2 will fail.
# This can sometimes be retried, but will require transaction rollbacks in cases...

def test_config_db(db_session):
    PATH = "./test_config_db.duckdb"
    
    db_session.add(Documents(content="test"))
    db_session.commit()

    test_embedding = {
        "document_id": 1,
        "content": "test",
        "embedding": [1.0 for i in range(settings.vec_size)],
        "start_pos": 0,
        "end_pos": 3
    }
    db_session.add(DocumentChunks(**test_embedding))
    db_session.commit()

    conn = config_db(PATH, config= {"hnsw_enable_experimental_persistence": True}, install=True)
    print(conn.execute("SELECT * FROM INFORMATION_SCHEMA.COLUMNS;").fetchall())
    create_vss_index(conn, "test_index")

    with raises(OperationalError):
        db_session.query(DocumentChunks).one()
    
    conn.close()
    db_session.query(DocumentChunks).one()

