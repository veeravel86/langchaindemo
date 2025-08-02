import os
from langchain_ollama import ChatOllama

llm = ChatOllama(model="gemma:2b")
question = input("What's your question? ")
response = llm.invoke(question)
print(response.content)