
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

OPENAI_API_KEY=st.secrets["OPENAI_API_KEY"]
#LANGCHAIN_TRACING_V2="True"
LANGCHAIN_API_KEY=st.secrets["LANGCHAIN_API_KEY"]
LANGCHAIN_PROJECT="Plan-and-execute"
print(LANGCHAIN_API_KEY)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user's request only based on the given context."),
    ("user", "Question: {question}\nContext: {context}")
])
model = ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()

chain = prompt | model | output_parser

question = "Can you summarize this morning's meetings?"
context = "During this morning's meeting, we solved all world conflict."
chain.invoke({"question": question, "context": context})