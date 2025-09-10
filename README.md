# 📘 Project Outlines: RAG-based AI Chatbot

## 1. Project Setup

**Goal:** Create the environment and install all tools.

**Instructions:**
- Install Python 3.10+ (Anaconda or standard).
- Install required libraries:
    ```bash
    pip install langchain openai faiss-cpu streamlit pypdf
    ```
- Get OpenAI API key from [platform.openai.com](https://platform.openai.com).
- Save API key in environment:
    - Linux/macOS:
        ```bash
        export OPENAI_API_KEY="your_api_key"
        ```
    - Windows:
        ```cmd
        set OPENAI_API_KEY="your_api_key"
        ```

---

## 2. Data Ingestion

**Goal:** Load documents into the chatbot system.

**Instructions:**
- Place your document (e.g., `data.pdf`) in a `docs/` folder.
- Use LangChain’s `PyPDFLoader` to read text:
    - Extract all text from PDF.
    - Split into small parts (chunks).

---

## 3. Text Splitting (Chunking)

**Goal:** Break large documents into smaller, searchable pieces.

**Instructions:**
- Use `RecursiveCharacterTextSplitter` from LangChain.
- Recommended settings:
    - Chunk size: 500 tokens (~300 words)
    - Overlap: 50 tokens
- Each chunk should keep metadata: page number, file name.

---

## 4. Create Embeddings

**Goal:** Turn text chunks into vectors (numbers) that can be searched.

**Instructions:**
- Use OpenAI embeddings:
    - `text-embedding-3-small` (cheap, fast)
    - `text-embedding-3-large` (better accuracy, higher cost)
- Convert all chunks → embeddings.
- Save embeddings + metadata.

---

## 5. Store in Vector Database

**Goal:** Save embeddings so we can quickly retrieve relevant chunks.

**Options:**
- **Beginner:** FAISS (local, free, simple)
- **Production:** Pinecone (cloud, scalable)

**Instructions (FAISS example):**
- Store all embeddings into FAISS index.
- Test search: enter a query → get top-3 similar chunks.

---

## 6. Retrieval + Generation Pipeline

**Goal:** Connect retriever with LLM to form RAG.

**Instructions:**
- When user asks a question:
    1. Search vector DB → get top chunks.
    2. Add these chunks to a prompt template.
    3. Send to LLM (OpenAI GPT model).
    4. Get answer back.

**Prompt example:**
```
Answer the question using only the provided context.
If you don’t know, say “I don’t know.”
Context: {retrieved_chunks}
Question: {user_query}
Answer:
```

---

## 7. Build Streamlit Chat UI

**Goal:** Make a simple web app for chatting.

**Instructions:**
- Use `st.chat_input` for user queries.
- Show responses with `st.chat_message`.
- Display sources (page/file) under each answer.
- Add a sidebar:
    - Upload new documents.
    - Re-index them automatically.

---

## 8. Improve Accuracy (Intermediate Level)

**Goal:** Make answers better and reduce hallucination.

**Instructions:**
- Tune retrieval: test `top_k = 3, 5, 8`.
- Experiment with chunk size (300–800 tokens).
- Add conversation memory so chatbot remembers history.
- Try hybrid search (keyword + embeddings).

---

## 9. Deployment

**Goal:** Share your chatbot.

**Instructions:**
- **Local run:**
    ```bash
    streamlit run app.py
    ```
- **Free hosting:**
    - Streamlit Cloud
    - Hugging Face Spaces
- **Professional hosting:**
    - Dockerize app
    - Deploy on Render, AWS, or GCP

---

## 10. Extra Features (Advanced)

Once basic version works, add these:
- Handle multiple file formats (PDF, DOCX, TXT).
- Add “I don’t know” filter if answer not found.
- Summarize long documents before answering.
- Track metrics: queries per day, average response time.

---

## ✅ Final Checklist (Beginner → MVP)

- Environment ready (LangChain + FAISS + Streamlit)
- Load a PDF file and extract text
- Split into chunks with metadata
- Create embeddings with OpenAI
- Store + search with FAISS
- Retrieval + LLM pipeline built
- Streamlit chat UI working
- Sources displayed in answers

