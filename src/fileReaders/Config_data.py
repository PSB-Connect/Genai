import os
from pathlib import Path

ENCODER_MODEL_NAME = "microsoft/layoutlmv3-base"
DECODER_MODEL_NAME = "deepset/roberta-base-squad2"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

BASE_FOLDER_PATH = Path(r"C:\Badri\Projects\Python\POC-2\data")
SOURCE_FOLDER_PATH = BASE_FOLDER_PATH / "source"  # Specify source folder name
TARGET_FOLDER_PATH = BASE_FOLDER_PATH / "target"  # Specify target folder name
METADATA_PATH = BASE_FOLDER_PATH / "metadata_json"  # Corrected case consistency
INDEX_PATH = BASE_FOLDER_PATH / "faiss_index"
FOLDER_PATH = BASE_FOLDER_PATH / "pdf-files"
