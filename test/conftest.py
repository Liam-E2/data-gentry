from pytest import fixture
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import Session
import duckdb

from src.data_gent.db import BaseTable


@fixture(scope="function")
def db_session():
    eng = create_engine("duckdb:///:memory:")
    BaseTable.metadata.create_all(eng)
    sess = Session(bind=eng)
    yield sess
    sess.close()
    eng.dispose()
