from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

my_data = [
    Document(page_content="Shahzaib is a software developer with expertise in Python, JavaScript, and machine learning.", metadata={"source": "personal_info.txt"}),
    Document(page_content="He has experience working on various projects, including web development, data analysis, and AI applications.", metadata={"source": "personal_info.txt"}),
    Document(page_content="Shahzaib has done his BS in Artificial Intelligence from NUML Islamabad and has worked on several projects during his academic career.", metadata={"source": "personal_info.txt"}),
]

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size =100, chunk_overlap=0  )
chunks = splitter.split_documents(my_data)

vectorstore_faiss = FAISS.from_documents(chunks, embeddings)

llm = ChatOllama(model="phi3:mini", temperature=0   )
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question based only on the following context:\n{context}"),
    ("human", "{question}")
])

def ask_my_expert(query):
    docs = vectorstore_faiss.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    chain = prompt | llm
    response = chain.invoke({"context": context, "question": query})
    return response.content

print("Asking my expert ....")
print(ask_my_expert("What is Shahzaib's expertise? Where did he do his BS? and what's his cgpa"))