import pytest
import tempfile
import csv

from src.data_gent.analysis.csv import DuckDBTools, DuckDBColumn, SampleData


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


def test_get_sample_data():
    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False) as tmp:
        writer = csv.writer(tmp)
        writer.writerows(SAMPLE_CSV)
        tmp_path = tmp.name

    tools = DuckDBTools()

    # Call the method requesting all columns, limit 2 rows
    sample_data: SampleData = tools.get_sample_data(
        file_path=tmp_path,
        columns=["id", "name", "score"],
        n_rows=2
    )

    # Validate the returned object is a SampleData instance
    assert isinstance(sample_data, SampleData)

    # Validate columns
    assert sample_data.columns == ["id", "name", "score"]

    # Validate rows count
    assert len(sample_data.rows) == 2

    # Validate row types and content
    for row in sample_data.rows:
        assert isinstance(row, dict)
        assert set(row.keys()) == set(sample_data.columns)
        assert isinstance(row["id"], int)
        assert isinstance(row["name"], str)
        assert isinstance(row["score"], float)

    # validate exact values
    expected_first_row = {"id": 1, "name": "Alice", "score": 95.5}
    assert sample_data.rows[0] == expected_first_row
