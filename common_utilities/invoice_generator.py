import os
import json
import logging
import config.config_util as configutil
import config.config_Template as configtemplateInvoice
import config.config_app as configApp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InvoiceGenerator:
    def __init__(self):
        self.json_file_path = configutil.JSON_FILE_PATH  # Ensure JSON folder path is initialized
        self.output_folder = configutil.PROCESSED_FOLDER_PATH
        self.Invoice_Template = configtemplateInvoice.TEMPLATE  # Holds the template data 

        os.makedirs(self.output_folder, exist_ok=True)

    def _clean_key(self, key):
        """Normalize keys for consistency."""
        key = key.replace("\\n", "").replace("\n", "").replace("#", "").strip()
        special_keys = {"a-location": "a_location", "b-location": "b_location"}
        return special_keys.get(key, key.replace("-", "_").replace(" ", "_").lower())

    def _clean_value(self, value):
        """Sanitize and remove newline characters from values."""
        if isinstance(value, str):
            return value.replace("\n", " ").strip()
        return value if value else "N/A"

    def generate_invoice(self, json_file_path):
        """Extract invoice details from JSON file."""
        try:
            # import pdb; pdb.set_trace()

            with open(json_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            logger.info(f"Loaded JSON data from {json_file_path}")

            print(f' generate_invoice data ={data}')

        except (json.JSONDecodeError, FileNotFoundError, Exception) as e:
            logger.error(f"Error reading {json_file_path}: {e}")
            return None

        if not data:
            logger.warning(f"Empty or invalid JSON structure in {json_file_path}")
            return None

        try:
            # # Extract invoice metadata
            invoice_fields = {
                "customer_name": self._clean_value(data.get("customer name") or data.get("customer_name")),
                "customer_id": self._clean_value(data.get("customer id") or data.get("customer_id")),
                "invoice_number": self._clean_value(data.get("Invoice Number") or data.get("invoice_number")or data.get("invoice number")),
                "bill_account": self._clean_value(data.get("bill account") or data.get("bill_account")),
                "bill_address": self._clean_value(data.get("bill address") or data.get("bill_address")),
                "service_address": self._clean_value(data.get("service address") or data.get("service_address")),
                "po_number": self._clean_value(data.get("PO Number") or data.get("po_number")),
                "date": self._clean_value(data.get("Date") or data.get("date")),
                "invoice_amount": self._clean_value(data.get("invoice amount") or data.get("invoice_amount")),
                "currency": self._clean_value(data.get("currency") or data.get("currency")),
            }

            
            

            # Extract circuit details
            circuit_details = []
            for table in data.get("Tables", []):
                if "circuitid" not in table:
                    continue  # Skip summary rows

                circuit_info = {
                    "circuitid": self._clean_value(table.get("circuitid")),
                    "a_location": self._clean_value(table.get("a-location")),
                    "b_location": self._clean_value(table.get("b-location")),
                    "bandwidth": self._clean_value(table.get("bandwidth")),
                    "mrc": self._clean_value(table.get("mrc")),
                    "nonrec_chgs": self._clean_value(table.get("nonrec_chgs")),
                    "total_charges": self._clean_value(table.get("total_charges"))
                }

                # Format each circuit's details
                circuit_text = f"""
                Circuit ID: {circuit_info['circuitid']}
                A End City: {circuit_info['a_location']}
                B End City: {circuit_info['b_location']}
                Bandwidth: {circuit_info['bandwidth']}
                Monthly Recurring Charge: {circuit_info['mrc']}
                Non Recurring Charge: {circuit_info['nonrec_chgs']}
                Total Charges: {circuit_info['total_charges']}
                """
                circuit_details.append(circuit_text.strip())

            invoice_fields["circuit_details"] = "\n".join(circuit_details)        

        except KeyError as e:
            logger.error(f"Missing expected key in JSON file: {e}")
            return None

        return invoice_fields if any(invoice_fields.values()) else None  # Ensure at least one valid field

    def create_files(self, invoice_fields, filename):
        """Generate invoice text files from extracted data."""
        if not invoice_fields:
            logger.error("Invoice fields are empty. Skipping file creation.")
            return

        if configApp.IS_CREATE_TEMPLATE_FILE:
            try:
                filled_template = self.Invoice_Template.format(**invoice_fields)
            except KeyError as e:
                logger.error(f"Missing key {e} in invoice fields. Cannot generate file.")
                return          
            
            # Generate output filename
            output_filename = f"invoice_{filename}.txt"
            output_path = os.path.join(self.output_folder, output_filename)

            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(filled_template.strip())

                logger.info(f"Generated invoice: {output_filename}")

            except IOError as e:
                logger.error(f"Failed to write invoice file: {e}")

    def process_folder(self):
        """Process all JSON files in the configured folder."""
        if not os.path.exists(self.json_file_path):
            logger.error(f"JSON file path does not exist: {self.json_file_path}")
            return

        all_invoice_metadata = []
        json_files = [f for f in os.listdir(self.json_file_path) if f.endswith(".json")]

        if not json_files:
            logger.warning(f"No JSON files found in {self.json_file_path}")
            return

        for filename in json_files:
            file_path = os.path.join(self.json_file_path, filename)
            invoice_data = self.generate_invoice(file_path)

            if invoice_data:
                logger.info(f"Invoice data extracted for {filename}")
                all_invoice_metadata.append(invoice_data)

                if configApp.IS_CREATE_TEMPLATE_FILE:
                    self.create_files(invoice_data, filename)
            else:
                logger.warning(f"No invoice data generated for {file_path}")

        logger.info(f"All Invoice Metadata: {json.dumps(all_invoice_metadata, indent=2)}")
        logger.info("Processing complete!")

# If running as a script
# if __name__ == "__main__":
#     invoice_generator = InvoiceGenerator()
#     invoice_generator.process_folder()
