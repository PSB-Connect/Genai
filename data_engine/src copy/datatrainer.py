import os
import logging
import faiss

from ...common_utilities.file_manager import FileLoader
from .text_processor import TextProcessor
from .vector_store_manager import VectorStoreManager



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataTrainer:
    def __init__(self, chunk_size=300, chunk_overlap=50, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.file_loader = FileLoader()
        self.text_processor = TextProcessor(chunk_size, chunk_overlap)
        self.vector_store_manager = VectorStoreManager(embedding_model_name)
        self.index = None

    def load_and_process_data(self, input_path, vector_store_path="llama_vector_store"):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Path not found: {input_path}")
        all_documents = []
        for root, _, files in os.walk(input_path):
            for file in files:
                file_path = os.path.join(root, file)
                logger.info(f"Processing file: {file_path}")
                documents = self.file_loader.load_file(file_path)
                all_documents.extend(documents)

        texts = self.text_processor.split_documents(all_documents)
        if not texts:
            raise ValueError("No texts to add to the vector store.")

        if os.path.exists(vector_store_path):
            logger.info("Loading existing vector store...")
            self.vector_store_manager.load_vector_store(vector_store_path)
            logger.info("Adding new documents to the vector store...")
            self.vector_store_manager.add_documents_to_vector_store(texts)
        else:
            logger.info("Creating new vector store...")
            self.vector_store_manager.create_vector_store(texts)

        self.vector_store_manager.save_vector_store(vector_store_path)
        logger.info("Data processed and embeddings stored successfully!")

    def initialize_faiss_index(self, dimension, n_clusters):
        self.index = faiss.IndexIVFFlat(faiss.IndexFlatL2(dimension), dimension, n_clusters)
        logger.info(f"FAISS index initialized with dimension {dimension} and {n_clusters} clusters.")

    def train_faiss_index(self, data):
        self.index.train(data)
        logger.info("FAISS index trained.")

    def add_to_faiss_index(self, data):
        self.index.add(data)
        logger.info("Data added to FAISS index.")

    def search_faiss_index(self, query, k=5):
        D, I = self.index.search(query, k)
        logger.info(f"Search results: {I}")
        return I

    def query_vector_store(self, query, k=5):
        return self.vector_store_manager.query_vector_store(query, k)

    def query_vector_store_with_llama2(self, query, k=5):
        return self.vector_store_manager.query_vector_store_with_llama2(query, k)