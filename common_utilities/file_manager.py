import os
import logging
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    CSVLoader,
)
import uuid
import config.config_util as configUtil
import config.config_app as configApp
from common_utilities.invoice_generator import InvoiceGenerator
from common_utilities.invoice_processor import InvoiceProcessor

# Configure logging
logging.basicConfig(level=logging.DEBUG, force=True)
logger = logging.getLogger(__name__)


class FileManager:
    def __init__(self):
        self.json_file_path = configUtil.JSON_FILE_PATH  # Ensure this path exists
        self.updated_json_file_path = configUtil.UPDATED_JSON_DATA_FILEPATH
        self.process_folder_path = configUtil.PROCESSED_FOLDER_PATH  # Verify if needed
        self.Invoice_Generator = InvoiceGenerator()
        self.Invoice_Processor=InvoiceProcessor(configUtil.UPDATED_JSON_DATA_FILEPATH)

    def _load_file(self, file_path):
        """Load a file based on its extension using appropriate loader."""
        try:
            logger.debug(f"Loading file: {file_path}")

            if file_path.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
            elif file_path.endswith(".pdf"):
                loader = UnstructuredPDFLoader(file_path)
            elif file_path.endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file_path.endswith(".xlsx"):
                loader = UnstructuredExcelLoader(file_path)
            elif file_path.endswith(".csv"):
                loader = CSVLoader(file_path)
            else:
                logger.warning(f"Skipping unsupported file type: {file_path}")
                return []

            documents = loader.load()
            logger.info(f"Loaded {len(documents)} documents from {file_path}")

            for doc in documents:
                filename_without_ext, _ = os.path.splitext(os.path.basename(file_path))
                doc.metadata["file_name"] = filename_without_ext  # Correctly removed extension
                doc.metadata["file_type"] = file_path.split(".")[-1]
            return documents
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return []

    def _load_and_process_data(self, input_path):
        """Load and process data from a given input directory."""
        if not os.path.exists(input_path):
            logger.error(f"Path not found: {input_path}")
            raise FileNotFoundError(f"Path not found: {input_path}")

        all_documents = []
        for root, _, files in os.walk(input_path):
            for file in files:
                file_path = os.path.join(root, file)
                logger.info(f"Processing file: {file_path}")
                documents = self._load_file(file_path)
                all_documents.extend(documents)

        return all_documents

    def manage_file_execution(self):
        """Main execution flow."""
        logger.info("Starting file execution...")

        # Step 1: Process folder
        all_processed_files = self.process_folder()

        # Step 2: Generate invoice using JSON data
        if all_processed_files:
            self.Invoice_Generator.generate_invoice(self.updated_json_file_path)
        else:
            logger.warning("No files were processed. Invoice generation skipped.")

    def create_metadata(self, invoice_data, file_path):
        """Generate metadata for each document in the invoice file."""
        metadata = {
            "file_name": os.path.splitext(os.path.basename(file_path))[0],  # Removed extension
            "file_type": file_path.split(".")[-1],  # Extracted file type
            "doc_id":str(uuid.uuid4()),  # Assign a unique ID
            "customer_name": invoice_data.get("customer_name", "N/A"),
            "customer_id": invoice_data.get("customer_id", "N/A"),
            "bill_account": invoice_data.get("bill_account", "N/A"),
            "bill_address": invoice_data.get("bill_address", "N/A"),
            "service_address": invoice_data.get("service_address", "N/A"),
            "invoice_number": invoice_data.get("invoice_number", "N/A"),
            "po_number": invoice_data.get("po_number", "N/A"),
            "date": invoice_data.get("date", "N/A"),
            "invoice_amount": invoice_data.get("invoice_amount", "N/A"),
            "currency": invoice_data.get("currency", "N/A"),
            "circuit_details": invoice_data.get("circuit_details", "N/A"),
        }
        return metadata

    def process_folder(self):
        """Process all JSON files in the configured folder."""
        all_invoice_data = []

        if not os.path.exists(self.json_file_path):
            logger.error(f"JSON file path does not exist: {self.json_file_path}")
            return []

        logger.info(f"Processing JSON files from {self.json_file_path}...")

        for filename in os.listdir(self.json_file_path):
            if filename.endswith(".json"):
                logger.info(f"Processing JSON file: {filename}")
                file_path = os.path.join(self.json_file_path, filename)
                # convert to small case
                self.Invoice_Processor.process_invoice(file_path)                
                self.Invoice_Processor.save_updated_invoice(file_path)

        
        for filename in os.listdir(self.updated_json_file_path):
                print(f'filename ={filename}')
                if filename.endswith(".json"):
                    logger.info(f"Processing JSON file: {filename}")
                    file_path = os.path.join(self.updated_json_file_path, filename)

                invoice_data = self.Invoice_Generator.generate_invoice(file_path)
                print(f'invoice_data = {invoice_data}')

                if invoice_data:
                    metadata = self.create_metadata(invoice_data, file_path)
                    logger.info(f"Generated Metadata: {metadata}")
                    
                    all_invoice_data.append(metadata)

                    if configApp.IS_CREATE_TEMPLATE_FILE:
                        filename_without_ext, _ = os.path.splitext(filename)
                        self.Invoice_Generator.create_files(invoice_data, filename_without_ext)
                else:
                    logger.warning(f"No invoice data generated for {file_path}")

        return all_invoice_data


# # If running as a script
# if __name__ == "__main__":
#     logger.info("Starting Invoice Processing...")

#     file_manager = FileManager()

#     all_invoice_data = file_manager.process_folder()
#     logger.info(f"All Invoice Metadata: {all_invoice_data}")

#     logger.info("Processing complete!")
