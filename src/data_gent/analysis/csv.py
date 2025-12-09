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


class DuckDBTools:
    def __init__(self, connection_string: str = ":memory:"):
        self.connection = duckdb.connect(connection_string)



    @tool
    def describe_csv(self, file_path: str) -> list[DuckDBColumn]:
        """
        Describe the schema of a CSV file.
        """
        con = duckdb.connect()
        rows = con.execute(f"""
            DESCRIBE SELECT * FROM read_csv_auto('{file_path}')
        """).fetchall()

        return [DuckDBColumn(*row) for row in rows]
