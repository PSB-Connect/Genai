import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline as hf_pipeline


class PDFQueryAgent:
    def __init__(self, index_path=r"faiss_index", metadata_path=r"metadata_json", embed_model="all-MiniLM-L6-v2", qa_model="deepset/roberta-base-squad2"):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embed_model = embed_model

        # HuggingFace embeddings wrapped for LangChain
        self.embeddings = HuggingFaceEmbeddings(model_name=embed_model)

        # Load FAISS index
        self.vectorstore = FAISS.load_local(
            index_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # HuggingFace QA pipeline (acts as the LLM in LangChain)
        qa_pipeline = hf_pipeline("question-answering", model=qa_model, tokenizer=qa_model)
        self.llm = HuggingFacePipeline(pipeline=qa_pipeline)

        # LangChain retriever-based QA chain
        self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=self.retriever, return_source_documents=True)

    def query(self, question):
        result = self.qa_chain.invoke(question)
        return {
            "answer": result["result"],
            "sources": [doc.metadata for doc in result["source_documents"]]
        }
    

agent = PDFQueryAgent()
response = agent.query("What is the deadline?")
print("Answer:", response["answer"])
for source in response["sources"]:
    print("Source:", source)
