import os
import shutil
import tempfile
import zipfile
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from ingest import build_index, load_index
from query import get_query_engine, ask

st.set_page_config(page_title="Obsidian Vault RAG Assistant", page_icon="📓")
st.title("📓 Obsidian Vault RAG Assistant")

PERSIST_DIR = os.environ.get("PERSIST_DIR", "storage")
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "obsidian_rag_upload")


def save_uploaded_files(files) -> str:
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    for f in files:
        if f.name.lower().endswith(".zip"):
            zip_path = os.path.join(UPLOAD_DIR, f.name)
            with open(zip_path, "wb") as out:
                out.write(f.getbuffer())
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(UPLOAD_DIR)
            os.remove(zip_path)
        else:
            with open(os.path.join(UPLOAD_DIR, f.name), "wb") as out:
                out.write(f.getbuffer())

    return UPLOAD_DIR


with st.sidebar:
    st.subheader("Your Vault")
    uploaded_files = st.file_uploader(
        "Upload your notes: a .zip of your vault, or individual files (.md, .txt, .pdf, .docx, and more)",
        type=None,
        accept_multiple_files=True,
    )
    st.caption("A .zip of your vault folder is easiest. Images and other binary files are automatically skipped.")

    if st.button("Build / Rebuild index"):
        if not uploaded_files:
            st.warning("Upload at least one .md file or a .zip first.")
        else:
            with st.spinner("Indexing your notes..."):
                vault_path = save_uploaded_files(uploaded_files)
                st.session_state["index"] = build_index(vault_path, PERSIST_DIR)
                st.session_state["messages"] = []
            st.success("Index rebuilt from your uploaded notes")

if "index" not in st.session_state:
    if os.path.exists(PERSIST_DIR):
        with st.spinner("Loading previous index..."):
            st.session_state["index"] = load_index(PERSIST_DIR)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "index" not in st.session_state:
    st.info("Upload your Obsidian notes in the sidebar and click 'Build / Rebuild index' to get started.")
else:
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("Ask something about your notes..."):
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                engine = get_query_engine(st.session_state["index"])
                answer, sources = ask(engine, question)
                full_answer = answer
                if sources:
                    full_answer += "\n\n**Sources:** " + ", ".join(sources)
                st.markdown(full_answer)

        st.session_state["messages"].append({"role": "assistant", "content": full_answer})
