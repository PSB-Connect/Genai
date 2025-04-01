import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline as hf_pipeline
import warnings
import src.fileReaders.Config_data as configutil

class PDFQueryAgent:
    def __init__(self):
        self.qa_model = configutil.DECODER_MODEL_NAME
        self.model_name = configutil.ENCODER_MODEL_NAME
        self.index_path = configutil.INDEX_PATH
        self.metadata_path = configutil.METADATA_PATH
        self.embed_model = configutil.EMBEDDING_MODELNAME
       
        # HuggingFace embeddings wrapped for LangChain
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embed_model)

        # Load FAISS index with warning suppression
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.vectorstore = FAISS.load_local(
                self.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        
        # Load metadata
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # HuggingFace QA pipeline (acts as the LLM in LangChain)
        qa_pipeline = hf_pipeline(
            "question-answering", 
            model=self.qa_model, 
            tokenizer=self.qa_model
        )
        self.llm = HuggingFacePipeline(pipeline=qa_pipeline)

        # LangChain retriever-based QA chain
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 5}
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm, 
            retriever=self.retriever, 
            return_source_documents=True
        )

    def query(self, question):
        """Query the indexed documents with a question"""
        if not question.strip():
            return {"answer": "Please provide a valid question.", "sources": []}
            
        result = self.qa_chain.invoke({"query": question})
        
        # Enhance sources with full text from metadata
        sources = []
        for doc in result["source_documents"]:
            source_metadata = doc.metadata
            # Find matching metadata entry to get full text
            for meta in self.metadata:
                if (meta["filename"] == source_metadata.get("filename") and 
                    meta.get("chunk_index") == source_metadata.get("chunk_index")):
                    source_metadata["full_text"] = meta["text"]
                    break
            sources.append(source_metadata)
        
        return {
            "answer": result["result"],
            "sources": sources
        }

if __name__ == "__main__":
    try:
        agent = PDFQueryAgent()
        response = agent.query("What is the deadline?")
        
        print("Answer:", response["answer"])
        print("\nSources:")
        for i, source in enumerate(response["sources"], 1):
            print(f"{i}. File: {source.get('filename', 'Unknown')}")
            print(f"   Page: {source.get('page_num', 'N/A')}")
            print(f"   Text excerpt: {source.get('full_text', '')[:200]}...\n")
            
    except Exception as e:
        print(f"Error: {str(e)}")