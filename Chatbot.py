import os
import streamlit as st
from langchain_openai.chat_models import ChatOpenAI
import bs4
from langchain import hub
import chromadb
from langchain_chroma import Chroma
from chromadb import Client, Settings
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

st.set_page_config(page_title=None, page_icon=None, layout="centered", initial_sidebar_state="expanded", menu_items=None)
st.title("LLM-powered chatbot for NFDI4Earth (Test version)")

# Ensure conversation history persists
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []

if "doc_links" not in st.session_state:
    with open("doc_links.yaml", "r") as f:
        data = yaml.safe_load(f)
        st.session_state["doc_links"] = data["doc_links"]

# Read OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Sidebar: Display API Key status
with st.sidebar:
    st.text("API Key Status:")
    if openai_api_key:
        st.success("API Key is set.")
    else:
        st.error("API Key is missing. Please set the OPENAI_API_KEY environment variable.")

    expanded = st.sidebar.expander("Document Links", expanded=False)

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
    # Check if OpenAI API key is set
    if not openai_api_key:
        st.error("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
        return

    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4", openai_api_key=openai_api_key)  # Use "gpt-4" or your desired model

    # Load documents from the provided links
    loader = WebBaseLoader(doc_links)
    docs = loader.load()

    # Preprocess documents (remove metadata or unwanted content)
    for doc in docs:
        if doc.page_content.startswith('---'):
            parts = doc.page_content.split('---', 2)  # Split into three parts
            doc.page_content = parts[2].strip() if len(parts) > 2 else doc.page_content

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    splits = text_splitter.split_documents(docs)

    # Initialize ChromaDB client (new way)
    client = chromadb.PersistentClient(path="/app/chroma_data")  # Use PersistentClient for local storage

    # Create vectorstore
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(openai_api_key=openai_api_key),
        client=client
    )

    # Define the prompt template
    template = """You are a helpful and informative AI assistant. Use the following information to answer the question:

    {context}

    Question: {question}. 
    Only answer the question based on the {context}.
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # Pull the retrieval QA chat prompt from the hub
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # Create the document chain and retrieval chain
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    rag_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

    # Invoke the RAG chain with the user's question
    user_question = input_text
    output = rag_chain.invoke({"input": user_question})
    formatted_output = format_output(output)  # Assuming format_output is a helper function

    # Update conversation history in session state
    if "conversation_history" not in st.session_state:
        st.session_state["conversation_history"] = []
    st.session_state["conversation_history"].append({"question": user_question, "answer": formatted_output})

    # Display conversation history
    for item in reversed(st.session_state["conversation_history"]):
        with st.container():
            st.write(f"You: {item['question']}")
            st.write(f"Chatbot: {item['answer']}")


# Input and Form
with st.form("my_form"):
    text = st.text_area(
        "Enter your question below:",
        "What is NFDI4Earth?",
    )
    submitted = st.form_submit_button("Submit")

# Warnings and Response Generation
if not openai_api_key:
    st.warning("Please set the OPENAI_API_KEY environment variable!", icon="⚠")
if submitted and openai_api_key:
    generate_response(text, st.session_state["doc_links"])
