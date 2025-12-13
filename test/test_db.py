from sqlalchemy import create_engine, Engine
from sqlalchemy.orm.session import Session
from pytest import fixture


from src.data_gent.db import Documents, BaseTable


@fixture
def db_session():
    eng = create_engine("duckdb:///:memory:")
    BaseTable.metadata.create_all(eng)
    sess = Session(bind=eng)
    yield sess
    sess.close()
    eng.dispose()


def test_create_documents_table(db_session):
    db_session.add(Documents(content="test"))
    db_session.commit()

    first = db_session.query(Documents).one()
    assert first.id == 1
    assert first.content == "test"
