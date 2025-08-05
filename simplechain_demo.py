from langchain.chains.summarize.map_reduce_prompt import prompt_template
from langchain.prompts import PromptTemplate
import os
import streamlit as st
from langchain_openai import ChatOpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

prompt_template=PromptTemplate(
    input_variables=["country", "no_of_paras", "language"],
    template="""You are an expert in traditional cuisines.
    You provide information about a specific dish from a specific country.
    Avoid giving information about fictional places or non-existent places.
    If the country is fictional or non-existent answer: I dont know.
    Answer the question : what is the traditional cuisine of {country}?
    Answer in {no_of_paras} paragraps in the language {language} 
    """
)

simplechain = prompt_template | llm

st.title("Cuisine info")
country = st.text_input("Enter the country:")
no_of_paras = st.number_input("Enter the number of paras:", min_value=1, max_value=5)
language = st.text_input ("enter the language")


if country:
    response = simplechain.invoke({
                                    "country": country,
                                     "no_of_paras": no_of_paras,
                                     "language": language})
    st.write(response.content)
