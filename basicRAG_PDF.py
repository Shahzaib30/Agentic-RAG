import os
import gc 
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

def run_rag():
    print("Loading PDFs....")
    loader = PyPDFDirectoryLoader("data/")
    docs = loader.load()

    #now chunking into small chunks to keep the context window small
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"Loaded {len(splits)} chunks from the PDFs.")

    # Embeddings 
    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    vector_store = FAISS.from_documents(splits, embeddings)

    del embeddings
    gc.collect()

    llm = ChatOllama(model="phi3:mini", temperature=0)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    template = """You are a Research Assistant. Use the following context to answer the question. 
    Ignore technical metadata like 'pdfTeX' or 'Document IDs'. Focus on the actual research findings.

    Context:
    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template=template)
    rag_chain = (
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")} | prompt | llm | StrOutputParser()

    )
    query = "What is the main topic of the PDFs? Summarize the key points."
    print("Asking the expert...")

    for chunk in rag_chain.stream({"question": query}):
        print(chunk, end = "", flush=True)
    print("\n" + "-`"*50 + "\n")

if __name__ == "__main__":
    load_dotenv()
    run_rag()


"""
output: 
The provided documents appear to be research findings related to a model called RAG (Retrieval-Augmented Generation). The primary focus seems to be on evaluating and comparing different aspects of this generative AI system, particularly in terms of its ability to handle various tasks such as fact verification.

Key points from the documents include:

1. Retrieval methods are used by RAG models for knowledge retrieval during generation processes. The model's performance is assessed based on how well it can complete titles using partial decoding, indicating that its parametric memory contains enough information to generate full sentences or paragraphs without relying heavily on specific documents (e.g., "The Sun Also Rises" and "A Farewell to Arms").

2. The researchers conducted experiments with the BART-only baseline model as well, which suggests that they are comparing its performance against models using retrieval methods like RAG. They also mention a knowledge refinement method designed for extracting critical information from retrieved documents while avoiding noisy or irrelevant content.

3. The researchers tested their approach on the FEVER (Fact Extraction and VERification) dataset, which involves tasks such as fact verification where models need to determine whether statements are correct based on evidence found in external knowledge sources like Wikipedia Dumps. They also mention using semi-structured data from PDFs for testing purposes.

4. The documents discuss the importance of noise robustness and negative rejection capabilities, which evaluate a model's ability to manage misinformation within retrieved documents while avoiding providing incorrect or irrelevant responses during generation tasks (e.g., Jeopardy). They also emphasize information integration as an essential aspect for handling complex questions that require synthesizing knowledge from multiple sources.

5. The researchers conducted experiments on various aspects of RAG models, including external knowledge required and model adaptation needed in the early stages versus later developments like Modular RAG with fine-tuning techniques. They also discuss challenges associated with semi-structured data (e.g., PDFs) due to text separation issues during retrieval processes or difficulties incorporating table information into semantic similarity searches.

In summary, these research findings focus on evaluating the performance of RAG models in various aspects such as fact verification, noise robustness, negative rejection, and handling complex scenarios involving multiple documents with semi-structured data sources like PDFs. The goal is to understand how well retrieval methods can support generative AI systems' capabilities while ensuring accurate responses during tasks that require knowledge integration from diverse external resources.

"""


