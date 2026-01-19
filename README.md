# DataGentry
🎩  
🧐  🦆

A small library for creating efficient file-specific agents / RAG systems with duckdb.

## Overview

Data Gentry packages together:
  - Loading data files and data documentation into a duckdb database with pre-built vector and full-text indices
on the data dictionary's contents. 
  - Simple interfaces for chunking + embedding documents and loading data, allowing the user to customize how the duckdb artifact is created.
    - Out-of-the-box chunking: Semchunk
    - Out-of-the-box embedding: Bedrock
  - Hybrid BM-25 / HNSW retrieval on the generated database.

The project is currently in a "proof-of-concept/playing around" phase, but in my mind could help to solve the problem that existing semantic layers are often tightly-coupled to vendors like Databricks or Snowflake, increasing vendor lock-in and coupling to spark workloads that are often overkill for the size of the data in question.

## TODO:
  - Support vector similarity metrics other than cosine similarity
  - Implement a set of tools to allow an agent to work with the artifact
  - Convenience functionality to auto-load from fs (/httpfs)?
