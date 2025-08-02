import os
import streamlit as st
from langchain_openai import ChatOpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

st.title("Ask a question")
question = st.text_input("What's your question?")

if question:
    response = llm.invoke(question)
    st.write(response.content)
