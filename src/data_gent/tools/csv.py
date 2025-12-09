from dataclasses import dataclass
import duckdb
from strands import tool


@dataclass
class DuckDBColumn():
    name: str
    type: str
    null: str
    key: str
    default: str
    extra: str


@tool
def read_csv(file_path: str) -> list[DuckDBColumn]:
    con = duckdb.connect()
    rows = con.execute(f"""
        DESCRIBE SELECT * FROM read_csv_auto('{file_path}')
    """).fetchall()

    return [DuckDBColumn(*row) for row in rows]
