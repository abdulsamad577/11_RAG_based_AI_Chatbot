import streamlit as st
import tempfile
import os
import logging
from pathlib import Path
from datetime import datetime

# =======================
# Logging Configuration
# =======================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# =======================
# Streamlit Page Config
# =======================
st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =======================
# Custom CSS Styling
# =======================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .chat-user {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .chat-ai {
        background-color: #f1f8e9;
        border-left: 4px solid #4caf50;
    }
    .status-box {
        padding: 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
    .success {background-color: #d4edda; border-left: 5px solid #28a745;}
    .error {background-color: #f8d7da; border-left: 5px solid #dc3545;}
    .warning {background-color: #fff3cd; border-left: 5px solid #ffc107;}
</style>
""", unsafe_allow_html=True)

# =======================
# Session State Init
# =======================
def init_session():
    defaults = {
        "documents_processed": False,
        "vectorstore_ready": False,
        "chat_history": [],
        "current_file": None,
        "num_chunks": 0,
        "sample_chunk": ""
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session()

# =======================
# Main Header
# =======================
st.markdown("""
<div class="main-header">
    <h1>🤖 RAG AI Assistant</h1>
    <p>Upload, process, and chat with your documents — powered by Retrieval-Augmented Generation</p>
</div>
""", unsafe_allow_html=True)

# =======================
# Config
# =======================
INDEX_PATH = "faiss_index"
MODEL_NAME = "Qwen/Qwen3-0.6B"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# =======================
# Sidebar
# =======================
with st.sidebar:
    st.subheader("📁 Upload Document")
    uploaded_file = st.file_uploader(
        "Upload PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt"]
    )

    if uploaded_file:
        if st.session_state.current_file != uploaded_file.name:
            st.session_state.current_file = uploaded_file.name
            st.session_state.documents_processed = False
            st.session_state.vectorstore_ready = False
            st.session_state.chat_history = []

    st.markdown("---")
    with st.expander("⚙️ Settings", expanded=False):
        chunk_size = st.slider("Chunk Size", 200, 1000, 500, 50)
        chunk_overlap = st.slider("Chunk Overlap", 20, 200, 50, 10)
        retrieval_k = st.slider("Retrieval Results (k)", 1, 10, 3)

    st.markdown("---")
    st.subheader("🧠 Model Info")
    st.info(f"**LLM:** {MODEL_NAME}")
    st.info(f"**Embeddings:** {EMBEDDING_MODEL}")

# =======================
# Main Layout
# =======================
col1, col2 = st.columns([1, 2])

# ---- Left Panel: Document Processing ----
with col1:
    st.header("📄 Document Processing")
    if uploaded_file:
        st.write(f"**File Name:** {uploaded_file.name}")
        st.write(f"**File Size:** {uploaded_file.size/1024:.1f} KB")
        st.write(f"**File Type:** {uploaded_file.type}")

        if st.button("🚀 Process Document", use_container_width=True):
            if not st.session_state.documents_processed:
                with st.spinner("Processing document..."):
                    try:
                        from langchain_community.document_loaders import PyPDFLoader
                        from langchain.text_splitter import RecursiveCharacterTextSplitter
                        from langchain_community.vectorstores import FAISS
                        from langchain_community.embeddings import HuggingFaceEmbeddings
                        from langchain.schema import Document

                        # Handle PDF or TXT (DOCX can be added similarly)
                        if uploaded_file.type == "application/pdf":
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                                tmp.write(uploaded_file.read())
                                path = tmp.name
                            loader = PyPDFLoader(path)
                            documents = loader.load()
                            os.unlink(path)
                        else:
                            content = uploaded_file.read().decode("utf-8")
                            documents = [Document(page_content=content, metadata={"source": uploaded_file.name})]

                        # Split & Embed
                        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                        docs = splitter.split_documents(documents)
                        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
                        vectorstore = FAISS.from_documents(docs, embeddings)
                        vectorstore.save_local(INDEX_PATH)

                        # Update State
                        st.session_state.documents_processed = True
                        st.session_state.vectorstore_ready = True
                        st.session_state.num_chunks = len(docs)
                        st.session_state.sample_chunk = docs[0].page_content[:200] + "..." if docs else ""

                        st.success("✅ Document processed successfully!")
                        logger.info("Document processed successfully")

                    except Exception as e:
                        st.error(f"Error: {e}")
                        logger.error(f"Processing error: {e}")
            else:
                st.info("ℹ️ Document already processed")

        if st.session_state.documents_processed:
            st.metric("Total Chunks", st.session_state.num_chunks)
            if st.session_state.sample_chunk:
                with st.expander("📄 Sample Preview"):
                    st.markdown(st.session_state.sample_chunk)

    else:
        st.warning("⚠️ Please upload a document to start.")

# ---- Right Panel: Chat Interface ----
with col2:
    st.header("💬 Chat with Document")
    if st.session_state.vectorstore_ready:
        for q, a in st.session_state.chat_history:
            st.markdown(f'<div class="chat-message chat-user">👤 <b>You:</b> {q}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-message chat-ai">🤖 <b>AI:</b> {a}</div>', unsafe_allow_html=True)

        query = st.text_input("Ask a question...", placeholder="e.g. Summarize the document")

        c1, c2 = st.columns([3, 1])
        with c1:
            if st.button("💭 Ask", use_container_width=True) and query:
                from langchain.prompts import PromptTemplate
                from langchain.chains import RetrievalQA
                from langchain_community.vectorstores import FAISS
                from langchain_community.embeddings import HuggingFaceEmbeddings
                from transformers import pipeline
                from langchain.llms import HuggingFacePipeline

                try:
                    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
                    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
                    retriever = vectorstore.as_retriever(search_kwargs={"k": retrieval_k})

                    local_llm = pipeline("text-generation", model=MODEL_NAME)
                    llm = HuggingFacePipeline(pipeline=local_llm)

                    template = """Answer the question using the context below.
                    If not in context, say you don’t know.

                    Context: {context}
                    Question: {question}
                    Answer:"""

                    prompt = PromptTemplate(input_variables=["context", "question"], template=template)
                    qa = RetrievalQA.from_chain_type(
                        llm=llm,
                        retriever=retriever,
                        chain_type="stuff",
                        chain_type_kwargs={"prompt": prompt}
                    )

                    result = qa.run(query)
                    st.session_state.chat_history.append((query, result))
                    st.rerun()

                except Exception as e:
                    st.error(f"Error generating response: {e}")
                    logger.error(f"Chat error: {e}")

        with c2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
    else:
        st.info("ℹ️ Process a document first to start chatting.")
        st.subheader("💡 Example Questions")
        st.write("- What’s the main topic?")
        st.write("- Summarize the key points")
        st.write("- Any recommendations mentioned?")

# =======================
# Footer
from datetime import datetime

current_year = datetime.now().year

st.markdown(
    f"""
<div style="
    text-align: center;
    color: #e3e3e3;
    font-size: 0.9rem;
    padding: 6px 0;
    background-color: black;
    border-radius: 6px;
    ">
    ⚡ Built with Streamlit, LangChain, and Hugging Face | © {current_year} RAG AI Assistant
</div>
""",
    unsafe_allow_html=True
)
