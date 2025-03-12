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

def format_output(output, similarity_score):
    """
    Formats the RAG chain output with Answer: and Source: labels based on the similarity score.

    Args:
        output: Dictionary containing answer and context information
        similarity_score: The similarity score of the retrieved document

    Returns:
        Formatted string with answer and source information (if similarity score is below 0.3)
    """
    answer = output["answer"]
    source_link = output["context"][0].metadata["source"]

    # If similarity score is less than 0.3, include the source
    if similarity_score < 0.3:
        return f"Answer: {answer}\nSource: {source_link}"
    else:
        # Otherwise, return only the answer
        return f"Answer: {answer}"


def generate_response(input_text, doc_links):
    if not openai_api_key:
        st.error("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
        return

    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)
    loader = WebBaseLoader(doc_links)
    docs = loader.load()
    for doc in docs:
        if doc.page_content.startswith('---'):
            parts = doc.page_content.split('---', 2)  # Split into three parts
            doc.page_content = parts[2].strip() if len(parts) > 2 else doc.page_content

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(openai_api_key=openai_api_key))
    retriever = vectorstore.as_retriever()
    docs_with_scores = vectorstore.similarity_search_with_score(input_text, k=3)
    similar_docs_with_scores = vectorstore.similarity_search_with_score(input_text)
    similarity_score = similar_docs_with_scores[0][1]  
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
    formatted_output = format_output(output,similarity_score)    
    
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
if not openai_api_key:
    st.warning("Please set the OPENAI_API_KEY environment variable!", icon="⚠")
if submitted and openai_api_key:
    generate_response(text, st.session_state["doc_links"])