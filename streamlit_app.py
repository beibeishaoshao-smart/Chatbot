import os
import streamlit as st
from loaders import DataLoader
from indexer import Indexer
from retriever import QARetriever
from memory import MemoryManager
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

PDF_PATH = "/Users/nikigao/Desktop/chatbot/data/Flower_Employee_Handbook.pdf"
DOCX_PATH = "/Users/nikigao/Desktop/chatbot/data/Flower_Operations_Guide.docx"

@st.cache_resource
def init_pipelines():
    # Load documents and build index
    documents = DataLoader(PDF_PATH, DOCX_PATH).load()
    vector_store = Indexer().build(documents)

    # Initialize conversation memory
    memory = MemoryManager().get_memory()

    # Conversational QA chain (with context)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0.0),
        retriever=vector_store.as_retriever(),
        memory=memory
    )

    # Marketing planning chain: generates structured marketing strategy
    marketing_prompt = PromptTemplate(
        input_variables=["goal"],
        template=(
            "You are a marketing consultant. Help the user design a marketing plan.\n"
            "Based on the user goal, outline: target audience, channels, creative strategy, implementation steps, timeline, and KPIs.\n"
            "User goal: {goal}"
        )
    )
    marketing_chain = LLMChain(
        llm=ChatOpenAI(temperature=0.7),
        prompt=marketing_prompt,
        memory=memory
    )

    return {"qa": qa_chain, "marketing": marketing_chain}

st.title("🌸 Fresh Flower Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

pipelines = init_pipelines()

user_input = st.text_input("Enter your question or marketing request:")
if st.button("Send") and user_input:
    # Route to marketing or QA based on keyword
    if "marketing" in user_input.lower() or "campaign" in user_input.lower():
        response = pipelines["marketing"].run(goal=user_input)
    else:
        result = pipelines["qa"].predict(question=user_input)
        response = result.get("answer", result)

    st.session_state.history.append((user_input, response))

# Display chat history
for query, answer in st.session_state.history:
    st.markdown(f"**You:** {query}")
    st.markdown(f"**Bot:** {answer}")
