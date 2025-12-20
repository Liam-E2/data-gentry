# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DataGent is a Python library for creating efficient file-specific agents and RAG (Retrieval Augmented Generation) systems using DuckDB. It packages together document loading, chunking, embedding, and hybrid (BM-25 + HNSW) retrieval into a lightweight, vendor-neutral semantic layer.

**Status**: Proof-of-concept/playing around phase

## Development Environment

**Package Manager**: UV (uv.lock present)
**Python Version**: >=3.14 (see .python-version)
**Database**: DuckDB with VSS (vector similarity search) and FTS (full-text search) extensions

### Key Commands

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest

# Run specific test
uv run pytest test/test_load.py::test_load_db

# Type checking
uv run mypy src/

# Run the package
uv run python -m data_gent
```

### Environment Variables

Configure via `DATAGENT_` prefixed environment variables (pydantic-settings):

- `DATAGENT_DB_PATH`: Path to DuckDB database file (default: `./db.duckdb`)
- `DATAGENT_VEC_SIZE`: Embedding vector size (default: `1024`)
- `DATAGENT_DOCS_DIR`: Documents directory (default: `/data-gent/load/docs`)

Test environment uses `pytest-env` plugin with overrides in `pyproject.toml`:
- `DATAGENT_DB_PATH=./db.duckdb`
- `DATAGENT_VEC_SIZE=3`

## Architecture

### Core Components

**Database Schema** (`db_models.py`):
- `Documents`: Stores raw documents with optional table association
- `DocumentChunks`: Stores text chunks with embeddings and foreign key to parent document
- Uses SQLAlchemy ORM with DuckDB engine via `duckdb-engine`
- Custom `FloatArray(n)` type for fixed-size embedding vectors
- `INDEX_DDL`: Creates HNSW index on embeddings and FTS index on chunk content

**Data Loading** (`load.py`):
- `load_document()`: Loads strings, chunks them, generates embeddings, creates indices
- `load_data()`: Loads structured data files (CSV, JSON, Parquet) into DuckDB tables
  - Supports auto-detection of file type from extension
  - Accepts DuckDB-specific read options via `opts` parameter
  - Returns created table name

**Retrieval** (`retrieval.py`):
- `retrieve()`: Hybrid search combining BM-25 (full-text) and HNSW (vector similarity)
  - Creates temporary `hnsw_results` table to hit HNSW index (DuckDB limitation)
  - Cleans up temp table after retrieval
  - Returns normalized and ranked `RetrievalResult` objects
  - Uses weighted combination: `fts_weight * bm25 + (1-fts_weight) * cosine_similarity`

**Embeddings** (`embeddings.py`):
- Abstract `EmbeddingSource` interface with `get_embedding(text: str) -> List[float]`
- `BedrockEmbeddingSource`: AWS Bedrock implementation (amazon.titan-embed-text-v2:0)
- `TestEmbeddingSource`: Returns constant vectors for testing (length `DATAGENT_VEC_SIZE`)

**Chunking** (`chunking.py`):
- Abstract `Chunker` interface with `chunk(text: str) -> Iterable[str]`
- `SemchunkChunker`: Uses semchunk library with configurable tokenizer, chunk size, and overlap
- `ParagraphChunker`: Naively chunks by splitting on `\n\n`

**Connection** (`connection.py`):
- `get_sqlalchemy_engine()`: Creates DuckDB engine with VSS and FTS extensions pre-loaded
- Uses NullPool (no connection pooling)
- Enables experimental HNSW persistence via config
- Auto-creates tables via SQLAlchemy metadata
- Can provide additional extensions to be loaded

**Preprocessors** (`preprocessors.py`):
- Processes binary files into chunkable strings.
- Eg. parsing a PDF to markdown

### Critical Implementation Details

**HNSW Index Query Limitation**: DuckDB's VSS extension cannot accelerate HNSW queries with subqueries, window functions, or complex expressions. The retrieval implementation works around this by:
1. Creating a temporary `hnsw_results` table with just the vector similarity computation
2. Running the main hybrid query that joins this temp table with BM-25 results
3. The temp table ensures the HNSW index is actually used

**Index Creation**: Indices are recreated after every document load via `create_chunk_indices()`. This drops and recreates both the HNSW vector index and the FTS index.

**Type Casting for Embeddings**: HNSW queries require explicit type casting to `FLOAT[N]` where N matches `DATAGENT_VEC_SIZE`. This is hardcoded in the query string (see `retrieval.py:39-40`).

**Test Database Cleanup**: Tests use function-scoped fixtures that create unique DuckDB files per test and clean them up afterward. Main code tests also manually call `os.remove(settings.db_path)` in finally blocks.

## Testing Strategy

- Tests use `NamedTemporaryFile` for test data
- `conftest.py` provides `db_session` fixture with per-test database cleanup
- Test files verify:
  - Document loading and chunking (test_load.py)
  - Hybrid retrieval ranking and temp table cleanup (test_retrieval.py)
  - Data file loading with various options (test_load.py)
- Test that temporary HNSW table is deleted after retrieval (test_retrieval.py:191-193)

## Dependencies

**Core**:
- `duckdb` + `duckdb-engine`: Database and SQLAlchemy integration
- `sqlalchemy`: ORM and database toolkit
- `boto3` (with types): AWS Bedrock embeddings
- `semchunk`: Document chunking
- `strands-agents`: Agent framework integration
- `pytz`: Timezone handling

**Dev**:
- `pytest` + `pytest-env`: Testing with environment configuration
- `mypy`: Static type checking
