"""Browser interface for the local GitHub RAG learning project."""

import json
import os
import shutil
from pathlib import Path

import streamlit as st
from gitingest import ingest

from chunk_repo import chunk_repository
from rag import DATA_DIR, answer, build_index, load_index, save_index, search

# Use the portable Git installation available in this workspace when needed.
portable_git = Path(__file__).parent.parent / "Projects" / "git" / "cmd"
if not shutil.which("git") and (portable_git / "git.exe").exists():
    os.environ["PATH"] = str(portable_git) + os.pathsep + os.environ.get("PATH", "")

st.set_page_config(page_title="GitHub Code Chat", page_icon="💬")
st.title("GitHub Code Chat")
st.caption("Ask questions about a repository. Answers use retrieved code and show its sources.")

if "index" not in st.session_state:
    st.session_state.index = None
    try:
        st.session_state.index = load_index()
    except (FileNotFoundError, ValueError, KeyError):
        pass
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Repository")
    url = st.text_input("Public GitHub URL", value="https://github.com/pallets/itsdangerous")
    if st.button("Load and index repository", type="primary"):
        if not url.startswith("https://github.com/"):
            st.error("Enter a public repository URL starting with https://github.com/.")
        else:
            try:
                with st.status("Reading repository…", expanded=True) as status:
                    summary, tree, content = ingest(url)
                    chunks = chunk_repository(content)
                    st.write("Building the search index…")
                    bar = st.progress(0.0)
                    index = build_index(chunks, lambda done, total: bar.progress(done / total))
                    # Save the new repository only once embedding succeeds.
                    DATA_DIR.mkdir(exist_ok=True)
                    for name, text in (("summary.txt", summary), ("tree.txt", tree), ("content.txt", content)):
                        (DATA_DIR / name).write_text(text, encoding="utf-8")
                    (DATA_DIR / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
                    save_index(index)
                    st.session_state.index = index
                    st.session_state.messages = []
                    status.update(label="Repository is ready", state="complete", expanded=False)
            except Exception as exc:
                st.error(str(exc))
    if st.session_state.index:
        st.success(f"Ready: {len(st.session_state.index['chunks'])} searchable pieces")
        if (DATA_DIR / "summary.txt").exists():
            st.text((DATA_DIR / "summary.txt").read_text(encoding="utf-8"))
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()


def show_sources(sources):
    with st.expander("Source code used for this answer"):
        for number, source in enumerate(sources, 1):
            st.markdown(f"**[{number}] `{source['file_path']}` · lines "
                        f"{source['start_line']}–{source['end_line']}")
            st.code(source["text"], language="python" if source["file_path"].endswith(".py") else None)


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            show_sources(message["sources"])

question = st.chat_input("Ask about the code…", disabled=st.session_state.index is None)
if question:
    with st.chat_message("user"):
        st.markdown(question)
    try:
        with st.chat_message("assistant"):
            with st.spinner("Finding code and writing an answer…"):
                sources = search(st.session_state.index, question)
                history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                response = answer(question, sources, history)
            st.markdown(response)
            show_sources(sources)
        st.session_state.messages.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": response, "sources": sources},
        ])
    except Exception as exc:
        st.error(str(exc))
