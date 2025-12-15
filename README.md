# DataGent
🎩  
🧐  🦆

A small library for creating efficient file-specific agents / RAG systems with duckdb.

## Overview

Data Gent packages together data files and data dictionaries into a duckdb database with pre-built vector and full-text indices
on the data dictionary's contents. The library also provides simple interfaces for loading data & documentation, allowing the user to customize how the duckdb artifact is created.

The project is currently in a "proof-of-concept/playing around" phase, but in my mind could help to solve the problem that existing semantic layers are often tightly-coupled to vendors like Databricks or Snowflake, increasing vendor lock-in and coupling to spark workloads that are often overkill for the size of the data in question.
