import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOllama(model="phi3:mini", temperature=0)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert AI tutor helping a student at NUML Islamabad."),
        ("human", "{input}"),
    ]
)

parser = StrOutputParser()

chain = prompt | llm | parser

print("Sending query to model ....")

try:
    response = chain.invoke({"input": "Explain langchain in simple words."})
    print("Response from model:", response)
except Exception as e:  
    print("Error occurred:", e)