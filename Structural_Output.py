from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from typing import List
from pydantic import BaseModel, Field

load_dotenv()

class FreelanceProject(BaseModel):
    client_name : str = Field(description="Name of the client")
    budget : int = Field(description="Budget for the project")
    skills : List[str] = Field(description="List of skills required for the project")


llm = ChatOllama(model="phi3:mini", temperature=0)

structural_output = llm.with_structured_output(FreelanceProject)
messy_text = """
Yo! I'm Shahzaib and i am looking for a developer to help with my web project. I can pay $500 for the whole 
thing. And i need someone who knows python and javascript. If you are interested, hit me up!
"""

print("Extracting Structured Data from Messy Text ....")
result = structural_output.invoke(messy_text)
print("Structured Output:", result)
print("Client Name:", result.client_name)
print("Budget:", result.budget)
print("Skills:", result.skills)