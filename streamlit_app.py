import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Adaptive Self-RAG", layout="wide")
st.title("Adaptive Self-RAG: Intelligent Document Research Assistant")

st.sidebar.title("Knowledge Sources")
st.sidebar.write("Upload files or add a URL to start retrieval.")

if st.sidebar.button("Build Demo Index"):
    st.info("Demo index creation is available in the backend graph package.")

question = st.text_area("Ask a research question", value="What are the key benefits of adaptive retrieval?")

col1, col2 = st.columns([1, 1])
with col1:
    uploaded = st.file_uploader("Upload PDF or text", accept_multiple_files=True)
with col2:
    url = st.text_input("Web URL", value="")

if st.button("Run Research Flow"):
    st.success("Research flow started. Retrieval, grading, query rewriting, verification, and answer generation are represented in the graph.")
    st.markdown("**Status:** Retrieving -> Evaluating documents -> Rewriting query -> Generating -> Verifying grounding -> Final answer")

st.subheader("Conversation History")
st.write("Q: What are the key benefits of adaptive retrieval?")
st.write("A: Adaptive retrieval improves document selection, re-grades relevance, and supports self-correction.")
