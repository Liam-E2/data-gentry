import duckdb


def config_db(db_path: str, config: dict, install: bool = False):
    conn = duckdb.connect(db_path, config=config)
    if install:
        conn.execute("INSTALL vss;")
    conn.execute("LOAD vss;")
    return conn


def create_vss_index(conn: duckdb.DuckDBPyConnection, idx_name: str):
    conn.execute(f"CREATE INDEX {idx_name} ON document_chunks USING HNSW (embedding);")
    conn.commit()
