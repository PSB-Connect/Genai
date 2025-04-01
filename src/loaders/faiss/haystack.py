import FAISSDocumentStore
from haystack.document_stores import FAISSDocumentStore  # Corrected import
from haystack.components.retrievers import EmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack import Pipeline
from haystack.utils import convert_files_to_document_objs

# Initialize FAISS Document Store
document_store = FAISSDocumentStore(index_path="faiss_index", similarity="cosine", embedding_dim=768)

# Load and Convert Documents
data_dir = "data/"  # Ensure this directory contains TXT, PDF, or DOCX files
documents = convert_files_to_document_objs(dir_path=data_dir)

# Define Embedding Model
embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")

# Initialize the Retriever
retriever = EmbeddingRetriever(document_store=document_store, embedder=embedder)  # Use 'embedder=' instead of 'embedding_model='

# Index Documents into FAISS
document_store.write_documents(documents)
document_store.update_embeddings(retriever)

print(f"Successfully uploaded {len(documents)} documents to FAISS!")

# Test Query
pipeline = Pipeline()
pipeline.add_component("retriever", retriever)

query = "What is artificial intelligence?"
results = pipeline.run({"retriever": {"query": query, "top_k": 5}})

# Print Retrieved Results
for doc in results["retriever"]["documents"]:
    print(f"Document: {doc.content[:300]}...\n")