from langchain.chains.summarize.map_reduce_prompt import prompt_template
from langchain.prompts import PromptTemplate
import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

topic_prompt_template=PromptTemplate(
    input_variables=["topic"],
    template="""You are an experienced speech writer.
    you need to craft an impact tile for a speech on 
    the following topic : {topic}
    Answer exactly with one title.
    """
)

speech_prompt_template=PromptTemplate(
    input_variables=["title"],
    template="""you need to write a powerful speech of 350 words
                for the below mentioned title : {title}
                
                """
)

first_chain = topic_prompt_template | llm | StrOutputParser()
second_chain = speech_prompt_template | llm
final_chain = first_chain | second_chain

st.title("Speech Generator")
topic = st.text_input("Enter the topic:")


if topic:
    response = final_chain.invoke({
                                    "topic": topic})
    st.write(response.content)
