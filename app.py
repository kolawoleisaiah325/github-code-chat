"""GitHub Code Chat: inspect sources offline, or ask a local Ollama model."""
import json
import time

import streamlit as st

from rag import (CHAT_MODEL, EMBED_MODEL, available_models, build_index,
                 context_sources, generate_answer, search, source_url)
from repository import ingest_repository
from sample import SAMPLE_QUESTIONS, sample_index

st.set_page_config(page_title="GitHub Code Chat", page_icon="💬", layout="wide")
st.markdown("""<style>
.block-container{max-width:1120px;padding-top:2.5rem;padding-bottom:3rem}
h1{letter-spacing:-.04em!important}h3{letter-spacing:-.02em!important}
.hero-kicker{font-size:.75rem;letter-spacing:.16em;color:#f9a17c;font-weight:650;margin-bottom:1rem}
.hero-copy{color:#b4bcc8;font-size:1.05rem;max-width:660px;line-height:1.8}
[data-testid=stSidebar]{border-right:1px solid #30353e}
[data-testid=stMetric]{background:#171d27;border:1px solid #303b49;border-radius:10px;padding:1rem}
[data-testid=stMetricLabel]{color:#b4bcc8}[data-testid=stChatMessage]{border:1px solid #303b49;border-radius:10px}
.stButton button{min-height:42px}code{font-size:.85em}
@media(max-width:640px){.block-container{padding-top:1.3rem;padding-inline:1rem}}
</style>""", unsafe_allow_html=True)

if "index" not in st.session_state:
    st.session_state.index = sample_index()
    st.session_state.messages = []
    st.session_state.mode = "Source search"
    st.session_state.summary = "Bundled fictional shop: cart, pricing, inventory, shipping and order rules."


def show_sources(sources, repository):
    with st.expander(f"Source code used · {len(sources)} excerpts", expanded=True):
        for number, source in enumerate(sources, 1):
            st.markdown(f"**[{number}] `{source['file_path']}`** · lines {source['start_line']}–{source['end_line']}")
            link = source_url(source, repository)
            if link:
                st.link_button(f"Open source [{number}] on GitHub ↗", link)
            st.code(source["text"], language="python" if source["file_path"].endswith(".py") else None)
        st.caption("Source links for imported repositories use the indexed commit. Bundled example links use main. Ranking scores are not answer-confidence probabilities.")


with st.sidebar:
    st.markdown("### 💬 GitHub Code Chat")
    st.caption("LOCAL INFERENCE · INSPECTABLE EVIDENCE")
    st.divider()
    st.markdown("#### Start exploring")
    if st.button("Load bundled example", use_container_width=True):
        st.session_state.index = sample_index()
        st.session_state.messages = []
        st.session_state.mode = "Source search"
        st.session_state.summary = "Bundled fictional shop: cart, pricing, inventory, shipping and order rules."
        st.rerun()
    st.radio("Answer mode", ["Source search", "Local AI"], key="mode",
             help="Source search works without models and returns excerpts, not generated answers.")
    use_ai = st.session_state.mode == "Local AI"
    models_ready = False
    if use_ai:
        try:
            installed = {name.removesuffix(":latest") for name in available_models()}
            models_ready = {EMBED_MODEL, CHAT_MODEL}.issubset(installed)
            if models_ready:
                st.success("Ollama models available")
            else:
                st.warning("Required Ollama models are missing.")
        except RuntimeError as exc:
            st.warning(str(exc))
        if not models_ready:
            st.code("ollama pull nomic-embed-text\nollama pull llama3.2", language="bash")
        if not st.session_state.index.get("vectors"):
            if st.button("Prepare local AI", disabled=not models_ready, type="primary", use_container_width=True):
                try:
                    with st.spinner("Embedding the current code snapshot…"):
                        old = st.session_state.index
                        candidate = build_index(old["chunks"], repository=old["repository"])
                    st.session_state.index = candidate
                    st.session_state.messages = []
                    st.rerun()
                except (RuntimeError, ValueError) as exc:
                    st.error(str(exc))
    st.divider()
    with st.form("repository-form"):
        url = st.text_input("Public repository URL", placeholder="https://github.com/owner/repository")
        load = st.form_submit_button("Load repository", use_container_width=True)
    if load:
        try:
            with st.status("Reading a pinned repository snapshot…", expanded=True) as state:
                summary, tree, content, chunks, metadata = ingest_repository(url)
                use_embeddings = use_ai and models_ready
                if use_embeddings:
                    st.write("Building embeddings…")
                bar = st.progress(0.0)
                candidate = build_index(chunks, lambda done, total: bar.progress(done / total),
                                        use_embeddings=use_embeddings, repository=metadata)
                st.session_state.index = candidate
                st.session_state.messages = []
                st.session_state.summary = summary
                state.update(label="Repository ready", state="complete", expanded=False)
                st.rerun()
        except (ValueError, RuntimeError, OSError) as exc:
            st.error(str(exc))
        except Exception:
            st.error("Repository ingestion failed. Check the URL, install Git, or try a smaller public repository. Your previous snapshot is retained.")
    st.caption("Repository code is read as text. It is never executed. This app keeps each browser session's index in memory.")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

index = st.session_state.index
repository = index["repository"]
st.markdown('<div class="hero-kicker">DEVELOPER TOOLS / RETRIEVAL-AUGMENTED GENERATION</div>', unsafe_allow_html=True)
st.title("Ask the code.")
st.markdown("### Check the evidence.")
st.markdown('<p class="hero-copy">Explore how a repository works through its actual source. Find relevant excerpts, inspect their line references, and optionally ask a local model to explain them.</p>', unsafe_allow_html=True)
left, middle, right = st.columns(3)
left.metric("ACTIVE SNAPSHOT", repository.get("name", "Repository"))
middle.metric("SEARCHABLE PIECES", len(index["chunks"]))
right.metric("ANSWER MODE", st.session_state.mode)
if repository.get("commit"):
    st.caption(f"Pinned commit: {repository['commit']} · Indexed: {index['indexed_at'][:19]} UTC")
else:
    st.caption("Bundled fictional shop · Works immediately without GitHub access or model downloads")
with st.expander("About this snapshot"):
    st.text(st.session_state.summary)
    st.caption("Answers may misunderstand code. Citation-format checks do not verify that a statement is true. Follow-up retrieval uses the current question; include a function or feature name.")
options = ["Keyword"] if not index.get("vectors") else ["Hybrid", "Keyword", "Vector"]
with st.expander("Retrieval settings"):
    retrieval = st.selectbox("Search method", options)
    top_k = st.slider("Maximum source excerpts", 1, 8, 4)
    st.caption("Hybrid combines BM25 keyword ranking and embedding similarity using reciprocal rank fusion. Overlapping excerpts are reduced before generation.")
method = {"Keyword": "lexical", "Hybrid": "hybrid", "Vector": "vector"}[retrieval]
ready = not use_ai or (models_ready and bool(index.get("vectors")))
if not ready:
    st.info("Choose Source search to explore now, or prepare local AI in the sidebar.")
quick_question = None
if not st.session_state.messages and repository.get("kind") == "bundled example":
    st.markdown("#### Try a question")
    columns = st.columns(2)
    for number, question in enumerate(SAMPLE_QUESTIONS):
        if columns[number % 2].button(question, key=f"sample-{number}", disabled=not ready, use_container_width=True):
            quick_question = question

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption(message["details"])
            show_sources(message["sources"], message["repository"])

chat_question = st.chat_input("Ask about a function, rule, or behavior…", disabled=not ready, max_chars=2000)
question = chat_question or quick_question
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            started = time.perf_counter()
            with st.spinner("Finding supporting code…"):
                sources = context_sources(search(index, question, limit=top_k, method=method))
                retrieval_seconds = round(time.perf_counter() - started, 3)
                if use_ai:
                    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                    result = generate_answer(question, sources, history)
                    response, sources = result["text"], result["sources"]
                    details = f"Local AI · {retrieval} retrieval {retrieval_seconds}s · Generation {result['seconds']}s · Status: {result['status']}"
                else:
                    response = ("Here are the closest source excerpts. This is a source search, not a generated explanation."
                                if sources else "No matching source evidence was found. Try naming a function or feature.")
                    details = f"Source search · BM25 keyword retrieval · {retrieval_seconds}s · No LLM used"
            st.markdown(response)
            st.caption(details)
            if sources:
                show_sources(sources, repository)
            st.session_state.messages.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": response, "sources": sources, "details": details, "repository": dict(repository)},
            ])
        except (RuntimeError, ValueError) as exc:
            st.error(str(exc))

if st.session_state.messages:
    transcript = json.dumps({"repository": repository, "messages": st.session_state.messages}, indent=2)
    st.download_button("Download conversation and sources", transcript, "code-chat-conversation.json", "application/json")
st.divider()
st.caption("Built by Isaiah Kolawole · Local AI engineering prototype · Source evidence remains available for review")
