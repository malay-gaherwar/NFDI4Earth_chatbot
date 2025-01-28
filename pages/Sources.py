import streamlit as st
import os
import yaml

if "doc_links" not in st.session_state:
  with open("doc_links.yaml", "r") as f:
    data = yaml.safe_load(f)
    st.session_state["doc_links"] = [doc["url"] for doc in data["doc_links"]] 
    


# Define a function to manage source links
def manage_source_links():
    """
    Manages source links, allowing addition, removal, and display.
    """
    source_links = st.session_state["doc_links"]
    
  
    # Remove a source link
    for i, link in enumerate(st.session_state["doc_links"]):
            st.write(f"{i+1}. {link}")
            if st.button(f"Remove Link {i+1}"):
                st.session_state["doc_links"].pop(i)
                st.rerun()

    

# Create Streamlit app layout
st.title("Source Links Management")

manage_source_links()