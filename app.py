#!/usr/bin/env python3
import os
import tempfile
import shutil
from typing import List, Tuple

import streamlit as st

from rag_utils import (
    save_uploaded_pdfs,
    ingest_paths,
    split_documents,
    get_or_create_faiss,
    make_hybrid_from_faiss,
    make_qa_chain,
    pretty_sources_from_resp,
)

st.set_page_config(page_title="PDF Q&A (Hybrid RAG)", page_icon="📄", layout="wide")

# ---------------------- Sidebar Controls ----------------------
with st.sidebar:
    st.title("📄 PDF Q&A — Hybrid RAG")
    st.caption("Upload PDFs → Split → Embed → Vector Store → Hybrid Search → Answer")

    api_key = st.text_input("OpenAI API Key", type="password", help="Will be used only for this session.")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.markdown("---")
    st.subheader("Chunking")
    chunk_size = st.slider("Chunk size", min_value=300, max_value=2000, value=1200, step=50)
    chunk_overlap = st.slider("Chunk overlap", min_value=0, max_value=600, value=200, step=25)

    st.subheader("Retrieval")
    top_k = st.slider("Top-K (per retriever)", min_value=2, max_value=12, value=6, step=1)
    alpha = st.slider("Dense weight α (FAISS)", min_value=0.0, max_value=1.0, value=0.6, step=0.05,
                      help="0.0 = BM25 only, 1.0 = Dense only")
    
    st.subheader("LLM")
    model_name = st.selectbox("Chat model", ["gpt-4o-mini", "gpt-4o", "gpt-4o-mini-ttl"], index=0)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

    st.subheader("Embeddings")
    embed_model = st.selectbox("Embedding model", ["text-embedding-3-small", "text-embedding-3-large"], index=0)

    st.markdown("---")
    index_root = st.text_input("Index directory", value="indexes", help="Folder where FAISS indexes are stored.")
    build_btn = st.button("🔧 Build / Load Index", type="primary")

# ---------------------- Main Layout ----------------------
st.header("Interactive PDF Q&A (Hybrid RAG)")
st.write("Upload one or more PDFs, then ask questions grounded in those documents.")

uploaded_files = st.file_uploader("Upload PDF file(s)", type=["pdf"], accept_multiple_files=True)

# Session state containers
if "last_index_dir" not in st.session_state:
    st.session_state["last_index_dir"] = None
if "vectordb" not in st.session_state:
    st.session_state["vectordb"] = None

# ---------------------- Build / Load Index ----------------------
if build_btn:
    if not api_key:
        st.error("Please enter your OpenAI API Key in the sidebar.")
    elif not uploaded_files:
        st.error("Please upload at least one PDF.")
    else:
        with st.status("Processing documents…", expanded=True) as status:
            try:
                # 1) Save uploaded PDFs to a temp folder (and compute fingerprint)

                tmp_dir = tempfile.mkdtemp(prefix="pdf_qna_")
                saved_paths, corpus_sha1 = save_uploaded_pdfs(uploaded_files, tmp_dir)
                st.write(f"Saved {len(saved_paths)} files. Corpus SHA1: `{corpus_sha1[:10]}…`")
                
                # 2) Ingest
                st.write("Loading PDFs…")
                docs = ingest_paths(saved_paths)

                # 3) Split
                st.write(f"Splitting documents (chunk_size={chunk_size}, overlap={chunk_overlap})…")
                chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

                # 4) Build/Load FAISS
                index_dir = os.path.join(index_root, corpus_sha1[:12])
                st.write(f"Index directory: `{index_dir}`")
                vectordb, created = get_or_create_faiss(
                    chunks,
                    index_dir=index_dir,
                    embed_model=embed_model,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                st.session_state["vectordb"] = vectordb
                st.session_state["last_index_dir"] = index_dir
                if created:
                    st.success("Built and saved FAISS index.")
                else:
                    st.info("Loaded FAISS index from disk.")

                status.update(label="Index ready ✔", state="complete")
            except Exception as e:
                st.exception(e)
            finally:
                # cleanup temp upload dir
                try:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                except Exception:
                    pass

# ---------------------- Ask Questions ----------------------
st.markdown("### Ask a question")
question = st.text_input("Your question", placeholder="e.g., What is the paper’s DOI?")

colA, colB, colC = st.columns([1,1,1])
with colA:
    run_btn = st.button("🔎 Retrieve & Answer", type="primary")
with colB:
    show_ctx = st.checkbox("Show retrieved chunks", value=False)
with colC:
    show_tokens = st.checkbox("Verbose debug", value=False)

if run_btn:
    if not api_key:
        st.error("Please enter your OpenAI API Key in the sidebar.")
    elif not question.strip():
        st.error("Please enter a question.")
    elif st.session_state.get("vectordb", None) is None:
        st.error("Please build or load an index first.")
    else:
        try:
            # Hybrid retriever from FAISS + BM25
            retriever = make_hybrid_from_faiss(st.session_state["vectordb"], k=top_k, alpha=alpha)

            # Build QA chain
            chain = make_qa_chain(retriever, model=model_name, temperature=temperature)

            # Run chain
            with st.spinner("Thinking…"):
                resp = chain.invoke({"input": question})

            answer = resp.get("answer") or resp.get("result") or ""
            st.markdown("## ✅ Answer")
            st.write(answer)

            # Pretty sources
            sources = pretty_sources_from_resp(resp)
            if sources:
                st.markdown("### 📚 Sources")
                for s in sources:
                    with st.expander(f"{s['file']} — page {s['page']}"):
                        st.write(s["snippet"])

            # Raw debug
            if show_ctx:
                st.markdown("---")
                st.subheader("Retrieved Chunks")
                ctx_docs = resp.get("context", [])
                for i, d in enumerate(ctx_docs):
                    with st.expander(f"Chunk {i+1} — Page {d.metadata.get('page', '?') + 1 if isinstance(d.metadata.get('page'), int) else '?'} — {d.metadata.get('source', '')}"):
                        st.write(d.page_content)

            if show_tokens:
                st.markdown("---")
                st.subheader("Raw Response Object")
                st.json(resp)

        except Exception as e:
            st.exception(e)

# Footer
st.markdown("---")
st.caption("Built with LangChain, FAISS, BM25 hybrid retrieval, and Streamlit.")