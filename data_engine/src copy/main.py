import logging
from test.fiass_ollama.datatrainer import DataTrainer
import numpy as np
import os
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    base_path = r"E:\Badri_NewVM\devops\GenAIPOC\fileRepository"
    path= os.path.join(base_path,"processed_outputs")
    
    print(f'{path}')
    
    input_path = r"E:\Badri_NewVM\devops\GenAIPOC\fileRepository\processed_outputs"
    vector_store_path = r"E:\Badri_NewVM\devops\GenAIPOC\test\fiass_ollama\fiass_ollama_vectore_store"

    data_trainer = DataTrainer()
    data_trainer.load_and_process_data(input_path, vector_store_path)
    
    data_trainer.load_and_process_data(input_path, vector_store_path)

     # Initialize FAISS index
    dimension = 128  # Example dimension
    n_clusters = 100  # Example number of clusters
    data_trainer.initialize_faiss_index(dimension, n_clusters)

    # Sample data for training
    data = np.random.random((10000, dimension)).astype('float32')
    data_trainer.train_faiss_index(data)
    data_trainer.add_to_faiss_index(data)

        # Sample query
    query = np.random.random((1, dimension)).astype('float32')
    results = data_trainer.search_faiss_index(query, k=5)
    print(results)  

    while True:
        query = input("Enter your query (type 'bye' or 'exit' to quit): ")
        if query.lower() in ["bye", "exit"]:
            print("Goodbye!")
            break
        response = data_trainer.query_vector_store_with_llama2(query)
        print("Response:", response)

if __name__ == "__main__":
    main()
# python -m test.fiass_ollama.datatrainer_main