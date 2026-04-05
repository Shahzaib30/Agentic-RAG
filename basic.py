import requests 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os

load_dotenv()

gemini_key = os.getenv("GOOGLE_GEMINI_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0.7, api_key=gemini_key, max_retries = 6)

response = llm.invoke("How much time does it take to learn complete langchain and langgraph along with postgresql and pgvector , chromadb will take for me i am a a beginner")

if isinstance(response.content, str):
    print(response.content[0]['text'])
else:
    print(response.content)