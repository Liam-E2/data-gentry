import os

from pytest import fixture
from sqlalchemy import create_engine, Engine, NullPool
from sqlalchemy.orm import Session
import duckdb

from src.data_gent.db_models import BaseTable


@fixture(scope="function")
def db_session(request):
    PATH = f"./{request.node.name}.duckdb"
    eng = create_engine("duckdb:///" + PATH, poolclass=NullPool)
    BaseTable.metadata.create_all(eng)
    sess = Session(bind=eng)
    yield sess
    sess.close()
    eng.dispose()

    os.remove(PATH)
