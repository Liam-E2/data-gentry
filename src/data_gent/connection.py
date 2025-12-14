from sqlalchemy import create_engine, Engine, NullPool
from sqlalchemy.orm import Session
import duckdb

from .config import settings
from .db_models import BaseTable


def get_sqlalchemy_engine():
    eng = create_engine("duckdb:///" + settings.db_path, poolclass=NullPool, connect_args={
        "preload_extensions": ["vss"],
        "config": {"hnsw_enable_experimental_persistence": True}
    })
    BaseTable.metadata.create_all(eng)
    return eng
