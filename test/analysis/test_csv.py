import pytest
import tempfile
import csv

from src.data_gent.analysis.csv import DuckDBTools, DuckDBColumn


SAMPLE_CSV = [
    ["id", "name", "score"],
    [1, "Alice", 95.5],
    [2, "Bob", 82.0],
]

def test_describe_csv():
    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False) as tmp:
        writer = csv.writer(tmp)
        writer.writerows(SAMPLE_CSV)
        tmp_path = tmp.name

    # Instantiate your class
    agent = DuckDBTools()

    # Call the method
    schema = agent.describe_csv(tmp_path)

    # Check that the return type is correct
    assert isinstance(schema, list)
    assert all(isinstance(col, DuckDBColumn) for col in schema)

    # Check that the columns match expected names
    expected_columns = ["id", "name", "score"]
    returned_columns = [col.name for col in schema]
    assert returned_columns == expected_columns

    # Check that types are reasonable
    types = [col.type for col in schema]
    assert "BIGINT" in types
    assert "VARCHAR" in types or "STRING" in types
    assert "DOUBLE" in types or "FLOAT" in types
