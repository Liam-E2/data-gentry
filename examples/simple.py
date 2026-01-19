import os
from tempfile import NamedTemporaryFile

os.environ["DATAGENT_VECSIZE"] = "1024"
os.environ["DATAGENT_DB_PATH"] = "./db.duckdb"

import boto3

from data_gentry.load import load_data, load_document
from data_gentry.connection import get_sqlalchemy_engine
from data_gentry.embeddings import BedrockEmbeddingSource
from data_gentry.chunking import ParagraphChunker
from data_gentry.retrieval import retrieve, RetrievalResult

# Load Data
engine = get_sqlalchemy_engine({"httpfs"})
load_data(engine, "https://data.providenceri.gov/api/views/4hhd-fzq6/rows.csv?accessType=DOWNLOAD", input_type="csv", table_name="checks")


# Load documentation
client = boto3.client("bedrock-runtime")
embedding_source = BedrockEmbeddingSource(client)

file = NamedTemporaryFile("w")
with open(file.name, "w") as f:
    f.write("""
This dataset contains information on unclaimed checks from the city of Providence, Rhode Island.
            
FUND_CASH_CODE: number
            
CHECK_DATE: timestamp
            
PAYEE_NAME: text
            
PAYEE_ADDRESS1: text
            
ADDRESS2: text

ADDRESS3: text
             
ADDRESS4: text
            
CITY: text
            
STATE: text
            
ZIP_CODE: text
            
PAYMENT_AMOUNT: number
""")

with open(file.name, "rb") as f:
    load_document(engine, embedding_source, ParagraphChunker(), f, table_name="checks")


# Example of hybrid retrieval
results : list[RetrievalResult] = retrieve(engine, "What fields contain address info?", embedding_source)
for r in results:
    print(r)

file.close()