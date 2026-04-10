from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

myPrivateInfo = """
My name is Shahzaib and I am a software developer with expertise in Python, JavaScript, and machine learning. I have experience working on various projects, including web development, data analysis, and AI applications. I am passionate about learning new technologies and applying them to solve real-world problems.
I have done my BS in Artificial Intelligence from NUML Islamabad and have worked on several projects during my academic career. I am currently looking for freelance opportunities where I can utilize my skills and contribute to interesting projects.
"""

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
chunks = text_splitter.split_text(myPrivateInfo)

vectorstore_faiss = FAISS.from_texts(chunks, embeddings)

llm = ChatOllama(model="phi3:mini", temperature=0)
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
print(ask_my_expert("What is my name, my skills and my University?"))

