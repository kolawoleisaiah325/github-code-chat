# GitHub Code Chat

A learning project for asking questions about GitHub repositories using local AI.

## Open the app

Keep Ollama running, then from this project folder run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Open http://localhost:8501. The app loads the saved sample index on startup. Enter a public GitHub repository URL and click **Load and index repository** to replace it. Ask a specific question, then expand **Source code used for this answer** to inspect the evidence. Clearing the conversation keeps the repository ready.

Required Ollama models:

```powershell
ollama pull llama3.2
ollama pull nomic-embed-text
```

The first model writes answers; the second produces embeddings. Inference runs locally. Fetching a public repository and downloading models require internet access.

This implementation calls Ollama directly and stores the index in a JSON file so the retrieval math is visible. The reference tutorial uses LlamaIndex to coordinate these operations.

## Lesson 1: ingestion

Ingestion means bringing the repository's files into our app as text. It is the first part of retrieval-augmented generation (RAG).

`read_repo.py` calls GitIngest, which returns a summary, a folder tree, and file contents. The script saves these in `data/` so you can inspect exactly what the app has read. It does not execute the repository's code.

## Environment

Python 3.12 is used with a project-specific `.venv`. Install the lesson's dependencies with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

To read a public repository (Git must be available on PATH):

```powershell
.\.venv\Scripts\python.exe read_repo.py https://github.com/octocat/Hello-World
```

You can also supply a local folder. The example repository is intentionally tiny so the extracted text is easy to inspect.

## Lesson 2: chunking

Run this after extracting a repository:

```powershell
.\.venv\Scripts\python.exe chunk_repo.py
```

`chunk_repo.py` splits each extracted file into pieces of up to 40 lines, with 8 lines shared between adjacent pieces. This overlap preserves some context at the boundaries. Chunks never cross a file boundary. The output in `data/chunks.json` includes each chunk's text, filename, and line numbers.

For example, a 90-line file becomes chunks covering lines 1-40, 33-72, and 65-90. Later, retrieval will choose chunks relevant to a question and give their text to the language model.

This is a simple learning baseline. Line counts do not limit token counts, and a function may span multiple chunks. Later improvements can split code at function boundaries and enforce token limits. Filenames are recovered from GitIngest's text delimiters; a more robust loader would preserve them directly as metadata during ingestion.

## Lesson 3: embeddings and retrieval

`rag.py` sends code sections to Ollama's `nomic-embed-text` model. Each section becomes a vector (a list of numbers). It stores the vectors with their text and source references in `data/index.json`.

When you ask a question, the same model embeds that question. **Cosine similarity** compares its vector to each stored vector. The ten most similar sections become the evidence given to Llama. A similarity score describes closeness in this search; it is not the probability that an answer is correct.

The model uses `search_document:` for repository text and `search_query:` for questions. Indexing skips dependency lockfiles and further divides long chunks into pieces of up to 1,600 characters. Oversized model inputs return an error rather than silently losing text. Existing indexes are rejected if their source text has changed.

Build or query the index from the terminal:

```powershell
.\.venv\Scripts\python.exe search_repo.py --build
.\.venv\Scripts\python.exe search_repo.py "How does URLSafeSerializer encode its payload?"
.\.venv\Scripts\python.exe search_repo.py "How does URLSafeSerializer encode its payload?" --answer
```

## Lesson 4: retrieval-augmented generation

`answer()` in `rag.py` puts the question and retrieved code into a prompt. Llama generates an explanation with numbered references. The browser shows the actual retrieved text alongside each answer so you can check it. Inspecting sources is part of using the app: a model can still misunderstand code or cite an excerpt incorrectly.

The app retains a few recent turns for conversational continuity. Retrieval uses the current question, so explicit follow-up questions work better than phrases like "what about that?". The index describes the repository snapshot at ingestion time and must be rebuilt to include new commits. Its filename and line metadata come from GitIngest's digest. The current app is intended for one local user and stores one active repository index.

Try these questions on the included ItsDangerous sample:

- What is the purpose of URLSafeSerializer?
- How does the serializer compress data before encoding it?
- How does TimestampSigner check whether a signature has expired?

## Pipeline

Repository → GitIngest → chunks → embeddings → similarity search → selected source excerpts → Llama → answer and source references.

Ollama API documentation: https://docs.ollama.com/api/embed

Embedding model instructions: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5

Ollama runs the language model. Python coordinates ingestion, retrieval, and the interface. Model downloads are managed separately by Ollama.

Reference tutorial: https://github.com/patchy631/ai-engineering-hub/tree/main/github-rag
