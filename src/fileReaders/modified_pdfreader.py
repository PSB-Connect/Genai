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
from tqdm import tqdm  # for progress bars

import src.fileReaders.Config_data as configutil

class PDFExtractor:
    def __init__(self):
        self.folder_path = configutil.FOLDER_PATH
        self.model_name = configutil.ENCODER_MODEL_NAME
        self.index_path = configutil.INDEX_PATH
        self.metadata_path = configutil.METADATA_PATH
        self.embed_model = configutil.EMBEDDING_MODELNAME

        # Initialize NLTK
        nltk.download('punkt', quiet=True)
        
        # Initialize models
        self._initialize_models(self.model_name, self.embed_model)
        
        # Load or create FAISS index
        self._initialize_index(self.index_path, self.metadata_path)

    def _initialize_models(self, model_name, embed_model):
        """Initialize the NLP and embedding models"""
        print("🔄 Initializing models...")
        # HF model for entity recognition
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.nlp_pipeline = pipeline(
            "token-classification", 
            model=self.model, 
            tokenizer=self.tokenizer, 
            aggregation_strategy="simple"
        )
        
        # Sentence transformer for embeddings
        self.embedder = SentenceTransformer(embed_model)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()

    def _initialize_index(self, index_path, metadata_path):
        """Initialize or load the FAISS index and metadata"""
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            self.index = faiss.read_index(index_path)
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            print(f"✅ Loaded existing index with {len(self.metadata)} entries.")
        else:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.metadata = []
            print("🆕 Created new FAISS index.")

    def extract_text(self, file_path):
        """Extract text from PDF using PyMuPDF (fitz)"""
        text = ""
        try:
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text()
        except Exception as e:
            print(f"\n⚠️ Text extraction error in {os.path.basename(file_path)}: {e}")
        return text.strip()

    def extract_tables(self, file_path):
        """Extract tables from PDF using pdfplumber"""
        tables = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_tables = page.extract_tables()
                    if page_tables:
                        tables.extend(page_tables)
        except Exception as e:
            print(f"\n⚠️ Table extraction error in {os.path.basename(file_path)}: {e}")
        return tables

    def extract_images(self, file_path):
        """Extract images from PDF using PyMuPDF (fitz)"""
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
            print(f"\n⚠️ Image extraction error in {os.path.basename(file_path)}: {e}")
        return images

    def _chunk_text(self, text, chunk_size=1000, overlap=200):
        """Split text into chunks with overlap to preserve context"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        
        for sent in sentences:
            if len(current_chunk) + len(sent) < chunk_size:
                current_chunk += " " + sent
            else:
                chunks.append(current_chunk.strip())
                current_chunk = current_chunk[-overlap:] + " " + sent if overlap > 0 else sent
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def index_text(self, text, filename, page_num=None):
        """Index text chunks with their embeddings"""
        if not text:
            return
        
        chunks = self._chunk_text(text)
        embeddings = self.embedder.encode(chunks, show_progress_bar=False)
        
        # Add to index and metadata
        self.index.add(np.array(embeddings).astype("float32"))
        for i, chunk in enumerate(chunks):
            self.metadata.append({
                "filename": filename,
                "chunk_index": i,
                "text": chunk,
                "page_num": page_num
            })

    def extract_from_pdf(self, file_path):
        """Process a single PDF file"""
        filename = os.path.basename(file_path)
        print(f"\n📄 Processing: {filename}")
        
        # Extract content
        text = self.extract_text(file_path)
        tables = self.extract_tables(file_path)
        images = self.extract_images(file_path)
        entities = []
        
        # Process entities if text exists
        if text:
            sentences = sent_tokenize(text)
            for sent in tqdm(sentences, desc="Extracting entities", leave=False):
                if sent.strip():
                    try:
                        results = self.nlp_pipeline(sent)
                        entities.extend(results)
                    except Exception as e:
                        print(f"\n⚠️ Entity extraction error: {e}")
            
            # Index the text
            self.index_text(text, filename)
        
        return {
            "filename": filename,
            "text": text,
            "entities": entities,
            "tables": tables,
            "images": images,
        }

    def extract_from_all_pdfs(self):
        """Process all PDFs in the folder"""
        results = {}
        pdf_files = [f for f in os.listdir(self.folder_path) if f.lower().endswith(".pdf")]
        
        if not pdf_files:
            print("❌ No PDF files found in the specified folder.")
            return results
        
        for filename in tqdm(pdf_files, desc="Processing PDFs"):
            full_path = os.path.join(self.folder_path, filename)
            results[filename] = self.extract_from_pdf(full_path)
        
        return results

    def search(self, query, top_k=5):
        """Search the indexed content"""
        if len(self.metadata) == 0:
            print("⚠️ No indexed content to search.")
            return []
            
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
        """Save the index and metadata to disk"""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)
        print(f"💾 Saved index with {len(self.metadata)} entries to {self.index_path}")

# Example usage
if __name__ == "__main__":
    extractor = PDFExtractor()
    
    # Process all PDFs
    extractor.extract_from_all_pdfs()
    extractor.save_index()
    
    # Search example
    print("\n🔍 Search results:")
    results = extractor.search("deadline or due date", top_k=3)
    for i, res in enumerate(results, 1):
        print(f"\nResult {i} (Score: {res['score']:.4f})")
        print(f"File: {res['metadata']['filename']}")
        print(f"Text: {res['metadata']['text'][:200]}...")