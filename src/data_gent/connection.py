from sqlalchemy import create_engine, Engine, NullPool
import duckdb

from .settings import settings
from .db_models import BaseTable


BASE_EXTENSIONS = {"vss", "fts"}


def get_sqlalchemy_engine(preload_extensions: set[str] = set()) -> Engine:
    """
    Returns a sqlalchemy engine with duckdb extensions vss and fts pre-loaded.
    """
    exts = BASE_EXTENSIONS.union(preload_extensions)
    conn = duckdb.connect(":memory:")
    for s in exts:
        conn.execute(f"INSTALL {s}")

    conn.close()
    
    eng = create_engine("duckdb:///" + settings.db_path, poolclass=NullPool, connect_args={
        "preload_extensions": list(exts),
        "config": {"hnsw_enable_experimental_persistence": True}
    })
        
    BaseTable.metadata.create_all(eng)
    return eng
