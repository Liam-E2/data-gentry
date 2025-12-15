from sqlalchemy import create_engine, Engine, NullPool
import duckdb

from .settings import settings
from .db_models import BaseTable


def get_sqlalchemy_engine() -> Engine:
    """
    Returns a sqlalchemy engine with duckdb extensions vss and fts pre-loaded.
    """
    conn = duckdb.connect(":memory:")
    conn.execute("INSTALL vss; INSTALL fts;")

    eng = create_engine("duckdb:///" + settings.db_path, poolclass=NullPool, connect_args={
        "preload_extensions": ["vss", "fts"],
        "config": {"hnsw_enable_experimental_persistence": True}
    })
        
    BaseTable.metadata.create_all(eng)
    return eng
