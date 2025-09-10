import streamlit as st
st.title("RAG-based AI Chatbot")
st.write("This is a simple RAG-based AI Chatbot application.")

st.sidebar.title("Upload Document")
uploaded_file = st.sidebar.file_uploader("Choose a file",type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    # Process the uploaded file
    st.write("File uploaded successfully!")

# Setting
index_path="faiss_index"
model_name="Qwen/Qwen3-0.6B"


# Step 2 & 3: Load Single PDF and Split into Chunks

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import tempfile

documents = None
docs = None

if uploaded_file is not None and uploaded_file.type == "application/pdf":
    st.write("Processing the uploaded PDF file...")
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    # PDF load karo
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"Total pages loaded: {len(documents)}")

    # Text ko chunks me todna
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,    # har chunk ~300 words
        chunk_overlap=50   # overlap taake context na tootay
    )

    docs = text_splitter.split_documents(documents)

    st.write(f"Total chunks created: {len(docs)}")
    st.write("First Chunk example:\n", docs[0])
elif uploaded_file is not None:
    st.warning("Only PDF files are supported for processing at this time.")
# High-level helper with pipeline

# Step 4: Create Embeddings using SentenceTransformers (Offline)

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Offline model (lightweight and free)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Apne chunks ko embeddings me convert karo aur FAISS me store karo
vectorstore = FAISS.from_documents(docs, embeddings)

# Save embeddings locally
vectorstore.save_local("faiss_index")

st.write("✅ Offline embeddings created and saved successfully!")

# Step 5: Store and Test Search in FAISS

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings  # ya OpenAIEmbeddings agar online use kar rahe ho

# Same embeddings model jo tumne create karne ke waqt use kiya tha
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# FAISS index ko load karo (jo tumne step 4 me save kiya tha)
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Test query
# input_query = st.text_input("Enter your query for the uploaded document:")
# query = input_query if input_query else "What is the main topic of the document?"

# Top 3 similar chunks retrieve karo
# results = vectorstore.similarity_search(query, k=3)

# st.write("🔍 Query:", query)
# st.write("Top 3 relevant chunks:")
# for i, res in enumerate(results, 1):
    # st.write(f"{i}. {res.page_content[:200]}...")  # sirf pehle 200 characters show karo


# Step 6: Retrieval + Generation Pipeline

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint   # Offline ke liye tum HF model bhi use kar sakte ho

# 1. Embeddings (same model jo tumne pehle use kiya tha)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. FAISS se load karo (jisme tumne step 5 me save kiya tha)
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# 3. Retriever banao
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. Prompt template
template = """
Answer the question using only the provided context.
If you don’t know, say "I don’t know."

Context: {context}
Question: {question}
Answer:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

# 5. LLM (Offline HuggingFace model)
from transformers import pipeline
local_llm = pipeline("text-generation", model="Qwen/Qwen3-0.6B")  # tum koi bhi HF model use kar sakte ho

# Wrapper (LangChain friendly)
from langchain.llms import HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=local_llm)

# 6. RetrievalQA Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",   # context ko "stuff" karke LLM ko bhejna
    chain_type_kwargs={"prompt": prompt}
)

# 7. Test Query
query= st.text_input("Enter your question about the document:")
# query = "Explain the role of middleware in client / server communcation. Aslo differentiate between 2 -tier and 3 tier client/server architecture."
result = qa.run(query)

# st.write("🔍 Question:", query)
st.write("🤖 Answer:", result)


# # Load the Qwen 0.6B model

# from transformers import pipeline

# pipe = pipeline(
#     "text-generation",
#     model="Qwen/Qwen3-0.6B",   # or local path e.g. "./models/Qwen3-0.6B"
#     trust_remote_code=True,
#     device_map="cpu"           # force CPU if no GPU
# )


# st.title("Welcome to Qwen 0.6B Chatbot")
# input=st.text_input("Ask me anything")

# if st.button("Submit"):
#     response = pipe(input, max_new_tokens=50, do_sample=True, temperature=0.7)
#     st.write(response[0]['generated_text'])





