import duckdb

from src.data_gent.load import config_db, create_vss_index

def test_config_db(db_session):
    PATH = "./test_config_db.duckdb"
    
    conn = config_db(PATH, config= {"hnsw_enable_experimental_persistence": True}, install=True)
    print(conn.execute("SELECT * FROM INFORMATION_SCHEMA.COLUMNS;").fetchall())
    create_vss_index(conn, "test_index")