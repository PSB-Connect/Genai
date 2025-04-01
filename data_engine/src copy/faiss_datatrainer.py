from langchain.schema import Document
import config.config_util as  configUtil
from common_utilities.file_manager import FileManager
from data_engine.src.faiss_datatrainer import FAISSDataTrainer

# python -m data_engine.src.faiss_datatrainer

class FAISSDataTrainer:
    def __init__(self, file_manager, vector_manager):
        self.file_manager = file_manager
        self.vector_manager = vector_manager
        self.FAISSDataTrainer = FAISSDataTrainer()
        self.FileManager =FileManager()

    def train_and_store(self, folder_path, save_path="faiss_invoice_db"):
        # Extract invoices from FileManager
        all_invoice_data = self.FileManager.process_folder(folder_path)

        # Convert extracted data into LangChain Document format
        documents = [
            Document(page_content=data["content"], metadata={"file_name": data["file_name"]})
            for data in all_invoice_data
        ]

        # Store in FAISS
        if documents:
            self.FAISSDataTrainer.create_vector_store(documents)
            self.FAISSDataTrainer.save_vector_store(save_path)
            print(f"✅ Successfully stored {len(documents)} documents in FAISS.")
        else:
            print("⚠️ No valid invoice data found.")

        return all_invoice_data
