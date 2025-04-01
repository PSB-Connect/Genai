from langchain.schema import Document
import config.config_util as configUtil  
from common_utilities.file_manager import FileManager
from data_engine.src.vector_store_manager import VectorStoreManager  # Ensure this exists

class FAISSDataTrainer:
    def __init__(self, file_manager: FileManager, vector_manager: VectorStoreManager):
        self.file_manager = file_manager  # ✅ Dependency injection
        self.vector_manager = vector_manager  # ✅ Injected vector manager instance

    def train_and_store(self, folder_path, save_path="faiss_invoice_db"):
        # Extract invoices from FileManager
        all_invoice_data = self.file_manager.process_folder()


        print(f'FAISSDataTrainer documents = {all_invoice_data}')

        # # Convert extracted data into LangChain Document format
        # documents = [
        #     Document(page_content=data["content"], metadata={"file_name": data["file_name"]})
        #     for data in all_invoice_data
        # ]

        documents = [
            Document(
                page_content=f"Invoice Number: {data['invoice_number']}, Customer: {data['customer_name']}, Amount: {data['invoice_amount']} {data['currency']}",  
                metadata={"file_name": data["file_name"], "doc_id": data["doc_id"]}
            )
            for data in all_invoice_data
        ]

        

        # Store in FAISS
        if documents:
            self.vector_manager.create_vector_store(documents)  # ✅ Call vector manager
            self.vector_manager.save_vector_store(save_path)  # ✅ Save FAISS DB
            print(f"✅ Successfully stored {len(documents)} documents in FAISS.")
        else:
            print("⚠️ No valid invoice data found.")

        return all_invoice_data

if __name__ == "__main__":
    # Initialize dependencies
    file_manager = FileManager()  
    vector_manager = VectorStoreManager()

    # Create an instance of FAISSDataTrainer
    trainer = FAISSDataTrainer(file_manager, vector_manager)

    # Define folder path where invoices are stored
    folder_path = configUtil.UPDATED_JSON_DATA_FILEPATH   
    # Run training process
    trainer.train_and_store(folder_path)
