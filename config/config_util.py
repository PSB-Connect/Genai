import os
from pathlib import Path

#----------------------MODEL DETAILS---------------------------------------------
ENCODER_MODEL_NAME = "microsoft/layoutlmv3-base"
DECODER_MODEL_NAME = "deepset/roberta-base-squad2"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

#---------------------- BASE FOLDER DETAILS------------------------------------------

BASE_FOLDER_PATH = Path(r"C:\Badri\Projects\Python\POC-2")


#----------------------FOLDER DETAILS-----------------------------------------------

SOURCE_FOLDER_PATH = BASE_FOLDER_PATH / "data/source"
PROCESSED_FOLDER_PATH = BASE_FOLDER_PATH / "data/Processed_Folder"  
METADATA_PATH = BASE_FOLDER_PATH / "data/metadata_json"  
INDEX_PATH = BASE_FOLDER_PATH / "data/faiss_index"
FOLDER_PATH = BASE_FOLDER_PATH / "data/pdf-files"
JSON_FILE_PATH = BASE_FOLDER_PATH / "data/json_files"
UPDATED_JSON_DATA_FILEPATH = BASE_FOLDER_PATH / "data/updated_json"

#----------------------VECTOR DB DETAILS------------------------------------------
VECTOR_DB_FIASS_FOLDER_PATH = BASE_FOLDER_PATH / "vector_db/fiass/"
VECTOR_DB_CHROMA_FOLDER_PATH = BASE_FOLDER_PATH / "vector_db/chroma/"