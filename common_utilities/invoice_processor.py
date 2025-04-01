import json
import os
import re

class InvoiceProcessor:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.updated_invoice = None
        os.makedirs(self.output_dir, exist_ok=True)

    def clean_text(self, text):
        """Removes newlines but preserves spaces, underscores, hyphens, and @ symbols."""
        return re.sub(r'\s*\n\s*', ' ', text).strip()  # Replace \n with a space and trim

    def convert_to_lowercase(self, data):
        """Recursively converts all dictionary keys and values to lowercase while preserving spaces."""
        if isinstance(data, dict):
            return {self.clean_text(key).lower(): self.convert_to_lowercase(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.convert_to_lowercase(item) for item in data]
        elif isinstance(data, str):
            return self.clean_text(data.lower())  # Convert to lowercase but keep spaces
        else:
            return data

    def read_invoice(self, file_path):
        """Reads the JSON invoice from the file."""
        print(f'Reading invoice: {file_path}')
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                invoice_data = json.load(file)
            return invoice_data
        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.")
            return None
        except json.JSONDecodeError:
            print("Error: Invalid JSON format.")
            return None

    def process_invoice(self, file_path):
        """Processes the invoice to clean and convert all keys and values to lowercase."""
        invoice_data = self.read_invoice(file_path)
        if invoice_data:
            self.updated_invoice = self.convert_to_lowercase(invoice_data)

    def save_updated_invoice(self, file_path):
        """Saves the updated invoice in the specified directory with 'updated_' prefix."""
        if self.updated_invoice is None:
            print("Error: No invoice data to save.")
            return
        
        base_name = os.path.basename(file_path)
        new_filename = os.path.join(self.output_dir, f"updated_{base_name}")

        try:
            with open(new_filename, 'w', encoding='utf-8') as file:
                json.dump(self.updated_invoice, file, indent=4, ensure_ascii=False)
            print(f"Updated invoice saved as: {new_filename}")
        except Exception as e:
            print(f"Error saving file: {e}")

# # Example usage:
# output_directory = "processed_invoices"
# processor = InvoiceProcessor(output_directory)

# input_invoice_path = "updated_invoice2.json"  # Your input file
# processor.process_invoice(input_invoice_path)
# processor.save_updated_invoice(input_invoice_path)
