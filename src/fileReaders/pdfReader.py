import os
import fitz
import pdfplumber
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from nltk.tokenize import sent_tokenize
import nltk

class PDFExtractor:
    def __init__(self, folder_path, model_name="microsoft/layoutlmv3-base", embed_model="all-MiniLM-L6-v2", index_path="faiss_index", metadata_path="metadata_json"):
        self.folder_path = folder_path
        self.model_name = model_name
        self.index_path = index_path
        self.metadata_path = metadata_path

        # HF model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.nlp_pipeline = pipeline("token-classification", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="simple")

        # Embeddings
        self.embedder = SentenceTransformer(embed_model)
        dim = self.embedder.get_sentence_embedding_dimension()

        # Load or create FAISS index
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            self.index = faiss.read_index(index_path)
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            print("✅ FAISS index and metadata loaded.")
        else:
            self.index = faiss.IndexFlatL2(dim)
            self.metadata = []
            print("🆕 New FAISS index created.")

    def extract_text(self, file_path):
        text = ""
        try:
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text()
        except Exception as e:
            print(f"Text extraction error in {file_path}: {e}")
        return text

    def extract_tables(self, file_path):
        tables = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_tables = page.extract_tables()
                    tables.extend(page_tables)
        except Exception as e:
            print(f"Table extraction error in {file_path}: {e}")
        return tables

    def extract_images(self, file_path):
        images = []
        try:
            with fitz.open(file_path) as doc:
                for i, page in enumerate(doc):
                    img_list = page.get_images(full=True)
                    for img_index, img in enumerate(img_list):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        images.append({
                            "page": i + 1,
                            "image_index": img_index,
                            "image_bytes": base_image["image"],
                            "ext": base_image["ext"],
                        })
        except Exception as e:
            print(f"Image extraction error in {file_path}: {e}")
        return images

    def index_text(self, text, filename):
        if not text:
            return
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        embeddings = self.embedder.encode(chunks)
        self.index.add(np.array(embeddings).astype("float32"))
        for i, chunk in enumerate(chunks):
            self.metadata.append({"filename": filename, "chunk_index": i, "text": chunk})

    def extract_from_pdf(self, file_path):
        nltk.download('punkt')
        print(f"📄 Processing: {file_path}")
        text = self.extract_text(file_path)
        tables = self.extract_tables(file_path)
        images = self.extract_images(file_path)
        entities = []
        if text:
            sentences = sent_tokenize(text)
            for sent in sentences:
                if sent.strip():  # avoid empty strings
                    try:
                        results = self.nlp_pipeline(sent)
                        entities.extend(results)
                    except Exception as e:
                        print(f"⚠️ Skipped sentence due to error: {e}")

                self.index_text(text, os.path.basename(file_path))

                return {
                    "text": text,
                    "entities": entities,
                    "tables": tables,
                    "images": images,
                }

    def extract_from_all_pdfs(self):
        results = {}
        for filename in os.listdir(self.folder_path):
            if filename.lower().endswith(".pdf"):
                full_path = os.path.join(self.folder_path, filename)
                results[filename] = self.extract_from_pdf(full_path)
        return results

    def search(self, query, top_k=5):
        embedding = self.embedder.encode([query]).astype("float32")
        distances, indices = self.index.search(embedding, top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                results.append({
                    "score": float(distances[0][i]),
                    "metadata": self.metadata[idx]
                })
        return results

    def save_index(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)
        print("💾 FAISS index and metadata saved.")

# Example usage:
extractor = PDFExtractor(r"C:\Users\dvikr\Documents\GEN-AI-Workspace\POC-2\data\pdf-files")
extractor.extract_from_all_pdfs()
extractor.save_index()

# Later...
extractor = PDFExtractor(r"C:\Users\dvikr\Documents\GEN-AI-Workspace\POC-2\data\pdf-files")  # It auto-loads saved index
results = extractor.search("deadline or due date")
for res in results:
    print(res["metadata"]["filename"], res["metadata"]["text"])
