import os
import streamlit as st
from langchain_openai.chat_models import ChatOpenAI
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import yaml

st.set_page_config(page_title=None, page_icon=None, layout="centered", initial_sidebar_state="expanded", menu_items=None)
st.title("LLM-powered chatbot for NFDI4Earth")

# Ensure API key and conversation history persist
if "open_api_key" not in st.session_state:
    st.session_state["open_api_key"] = ""

if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []

if "doc_links" not in st.session_state:
    with open("doc_links.yaml", "r") as f:
        data = yaml.safe_load(f)
        st.session_state["doc_links"] = data["doc_links"]

# Sidebar: Enter API Key
with st.sidebar:
    st.text("Enter your API Key:")
    st.session_state["open_api_key"] = st.text_input("OpenAI API Key", type="password", value=st.session_state["open_api_key"])
    expanded = st.sidebar.expander("Document Links", expanded=False)

os.environ["OPENAI_API_KEY"] = st.session_state["open_api_key"]

def format_output(output):
    """
    Formats the RAG chain output with Answer: and Source: labels.

    Args:
        output: Dictionary containing answer and context information

    Returns:
        Formatted string with answer and source information
    """
    answer = output["answer"]
    source_link = output["context"][0].metadata["source"]
    return f"Answer: {answer}\nSource: {source_link}"


def generate_response(input_text, doc_links):
    llm = ChatOpenAI(model="gpt-4o-mini")
    loader = WebBaseLoader(doc_links)
    docs = loader.load()
    for doc in docs:
        if doc.page_content.startswith('---'):
            parts = doc.page_content.split('---', 2)  # Split into three parts
            doc.page_content = parts[2].strip() if len(parts) > 2 else doc.page_content

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    template = """You are a helpful and informative AI assistant. Use the following information to answer the question:

    {context}

    Question: {question}. 
    Only answer the question based on the {context}.

    """

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    rag_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)
    user_question = input_text
    output = rag_chain.invoke({"input": user_question})  
    formatted_output = format_output(output)    

    # Update conversation history in session state
    st.session_state["conversation_history"].append({"question": user_question, "answer": formatted_output})

    # Display conversation history
    for item in reversed(st.session_state["conversation_history"]):
        with st.container():
            st.write(f"You: {item['question']}")
            st.write(f"Chatbox {item['answer']}")


# Input and Form
with st.form("my_form"):
    text = st.text_area(
        "Enter your question below:",
        "What is NFDI4Earth?",
    )
    submitted = st.form_submit_button("Submit")

# Warnings and Response Generation
if not st.session_state["open_api_key"].startswith("sk-"):
    st.warning("Please enter your OpenAI API key!", icon="⚠")
if submitted and st.session_state["open_api_key"].startswith("sk-"):
    generate_response(text, st.session_state["doc_links"])
