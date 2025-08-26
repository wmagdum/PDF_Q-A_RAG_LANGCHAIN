from __future__ import annotations
import os
import json
import hashlib
import tempfile
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


# ---------------------- File Handling ----------------------
def save_uploaded_pdfs(uploaded_files, target_dir: str) -> Tuple[List[str], str]:
    """
    Saves uploaded Streamlit files to target_dir. Returns (paths, corpus_sha1).
    corpus_sha1 is a fingerprint of the raw file bytes (order-insensitive).
    """
    paths = []
    sha = hashlib.sha1()
    for uf in uploaded_files:
        # uf is a streamlit.UploadedFile; we must read its bytes
        data = uf.read()
        sha.update(hashlib.sha1(data).digest())
        filename = uf.name
        out_path = os.path.join(target_dir, filename)
        with open(out_path, "wb") as f:
            f.write(data)
        paths.append(out_path)
    corpus_sha1 = sha.hexdigest()
    return paths, corpus_sha1


# ---------------------- Ingest & Split ----------------------
def ingest_paths(paths: List[str]) -> List[Document]:
    """Ingest a list of PDF paths into LangChain Documents."""
    all_docs: List[Document] = []
    for p in paths:
        loader = PyPDFLoader(p)
        docs = loader.load()
        # fill metadata
        for d in docs:
            d.metadata = {**(d.metadata or {}), "source": p, "filename": os.path.basename(p)}
        all_docs.extend(docs)
    return all_docs


def split_documents(docs: List[Document], chunk_size: int = 1200, chunk_overlap: int = 200) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


# ---------------------- Vector Store Persistence ----------------------
def _hash_corpus(chunks: List[Document]) -> str:
    h = hashlib.sha1()
    for d in chunks:
        h.update((d.page_content or "").encode("utf-8"))
    return h.hexdigest()


def get_or_create_faiss(
    chunks: List[Document],
    index_dir: str = "faiss_idx",
    embed_model: str = "text-embedding-3-small",
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
):
    """
    Loads FAISS from index_dir if present (and matches metadata); otherwise builds and saves.
    Returns: (vectordb, created_new: bool)
    """
    os.makedirs(index_dir, exist_ok=True)
    meta_path = os.path.join(index_dir, "meta.json")
    embeddings = OpenAIEmbeddings(model=embed_model)

    if os.path.exists(os.path.join(index_dir, "index.faiss")) and os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("embed_model") == embed_model and \
           meta.get("chunk_size") == chunk_size and \
           meta.get("chunk_overlap") == chunk_overlap:
            vectordb = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
            return vectordb, False

    # build new
    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local(index_dir)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "embed_model": embed_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "corpus_sha1": _hash_corpus(chunks)
        }, f, ensure_ascii=False, indent=2)
    return vectordb, True


# ---------------------- Hybrid Retriever ----------------------
def make_hybrid_from_faiss(vectordb, k: int = 6, alpha: float = 0.6):
    """
    Build a hybrid retriever combining BM25 (keywords) and FAISS (dense vectors).
    alpha is weight for dense; (1 - alpha) for BM25.
    """
    # get docs from FAISS docstore
    docs = list(vectordb.docstore._dict.values())
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = k
    dense = vectordb.as_retriever(search_kwargs={"k": k})
    return EnsembleRetriever(retrievers=[bm25, dense], weights=[1.0 - alpha, alpha])


# ---------------------- QA Chain ----------------------
SYSTEM_PROMPT = """You are a precise, citation-first assistant.
Answer the user's question using ONLY the provided context.
If the answer is not in the context, say you don't know.
Keep answers concise and include key values (names, numbers, dates) when present.
"""

def make_qa_chain(retriever, model: str = "gpt-4o-mini", temperature: float = 0.0):
    llm = ChatOpenAI(model=model, temperature=temperature)
    prompt = ChatPromptTemplate.from_template(
        SYSTEM_PROMPT + "\n\nQuestion: {input}\n\nContext:\n{context}"
    )
    doc_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, doc_chain)


def pretty_sources_from_resp(resp) -> List[dict]:
    """Return list of {file, page, snippet} from a chain response."""
    out = []
    for d in resp.get("context", []):
        meta = d.metadata or {}
        page = meta.get("page")
        out.append({
            "file": meta.get("filename") or meta.get("source"),
            "page": (page + 1) if isinstance(page, int) else page,
            "snippet": d.page_content[:600] + ("…" if len(d.page_content) > 600 else ""),
        })
    return out
