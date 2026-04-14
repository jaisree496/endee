import streamlit as st
from utils.embeddings import get_embedding
from utils.endee_db import EndeeVectorDB
from utils.llm import GroqLLM  # NEW IMPORT

st.title("Multi-Document RAG Assistant (Endee + Groq)")
st.markdown("**Powered by Endee Vector DB + Groq Llama3**")

# Persist DB in session
if "db" not in st.session_state:
    st.session_state.db = EndeeVectorDB()
    st.session_state.llm = GroqLLM()  # NEW: LLM instance

db = st.session_state.db
llm = st.session_state.llm

# Sidebar for instructions
with st.sidebar:
    st.info("**Instructions**")
    st.markdown("""
    1. Upload .txt files
    2. Wait for processing 
    3. Ask questions about your documents
    4. Get AI answers with sources!
    """)

# Upload files
uploaded_files = st.file_uploader(
    "Upload multiple text files",
    type=["txt"],
    accept_multiple_files=True
)

# Process files
if uploaded_files and "loaded" not in st.session_state:
    progress_bar = st.progress(0)
    total_files = len(uploaded_files)
    
    for i, file in enumerate(uploaded_files):
        content = file.read().decode("utf-8")
        chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]
        
        for chunk in chunks:
            emb = get_embedding(chunk)
            db.add(chunk, emb, file.name)
        
        progress_bar.progress((i + 1) / total_files)
    
    st.session_state.loaded = True
    st.success(f"{total_files} files processed! ({len(db.texts)} chunks indexed)")

# Debug info
col1, col2 = st.columns(2)
with col1:
    st.metric("Indexed Chunks", len(db.texts))
with col2:
    st.metric("Vector DB", "Endee")

# Query interface
st.subheader("Ask Questions About Your Documents")
query = st.text_input("Enter your question:", key="query")

col1, col2 = st.columns([3, 1])
with col1:
    k_docs = st.slider("Retrieve top K docs", 1, 5, 3)
with col2:
    if st.button("Search & Answer", type="primary"):

        if query and db.texts:
            with st.spinner("Searching Endee DB + Generating answer..."):
                # 1. Get embeddings & retrieve
                query_emb = get_embedding(query)
                results = db.search(query_emb, k=k_docs)
                
                # 2. Prepare context
                context = "\n\n".join([f"[{r['source']}] {r['text']}" for r in results])
                
                # 3. Generate answer with Groq
                answer = llm.generate(query, context)
                
                # Display results
                st.subheader("AI Answer")
                st.markdown(f"**{answer}**")
                
                st.subheader("Retrieved Sources")
                for i, res in enumerate(results, 1):
                    with st.expander(f"Source {i}: {res['source']}"):
                        st.write(res["text"])
                        
        else:
            st.warning("Please upload files first and enter a question!")

# Footer
st.markdown("---")
st.markdown("*Built for **Endee SDE/ML Internship** using Endee Vector Database *")