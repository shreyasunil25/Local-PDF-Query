# PDF QnA — Local RAG Web App

A local Retrieval-Augmented Generation (RAG) web application that lets users upload a PDF and ask natural language questions about its contents.

This project runs entirely on your machine using Ollama, so there are no OpenAI calls and no API costs.

## Features

- Upload a PDF through a clean Flask web interface
- Split the PDF into chunks for retrieval
- Generate embeddings locally using `nomic-embed-text`
- Store embeddings in ChromaDB
- Ask questions in a polished query interface
- Generate answers locally using `llama3.2`
- No paid APIs, fully local workflow

## Tech Stack

- Python
- Flask
- LangChain
- LangChain Community
- LangChain Ollama
- LangChain Chroma
- ChromaDB
- Ollama
- Vanilla HTML, CSS, JavaScript
- uv

## Project Structure

```text
pdf-qna/
├── app.py
├── templates/
│   └── index.html
├── uploads/
├── chroma_db/
├── .venv/
├── pyproject.toml
├── uv.lock
├── .gitignore
└── README.md
```

## How It Works

1. The user uploads a PDF.
2. The backend saves the file locally.
3. `PyPDFLoader` reads the PDF.
4. `RecursiveCharacterTextSplitter` breaks the content into chunks.
5. `OllamaEmbeddings` creates embeddings using `nomic-embed-text`.
6. ChromaDB stores the vectors locally.
7. On each question, the retriever fetches the most relevant chunks.
8. The prompt and retrieved context are sent to `llama3.2` through Ollama.
9. The app returns a grounded answer from the PDF.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Create the environment and install dependencies

```bash
uv venv
uv add langchain langchain-community langchain-ollama langchain-core langchain-chroma chromadb pypdf flask langchain-text-splitters
```

### 3. Install and start Ollama

Install Ollama from [ollama.com](https://ollama.com).

Then run:

```bash
ollama serve
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 4. Run the application

```bash
uv run python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## API Routes

### `GET /`

Serves the frontend.

### `POST /upload`

Accepts a PDF file, loads it, chunks it, embeds it, and stores vectors in ChromaDB.

### `POST /ask`

Accepts JSON in this format:

```json
{
  "question": "What is the main topic of this PDF?"
}
```

Returns:

```json
{
  "answer": "..."
}
```

## Notes

- Ollama must be running before you upload or ask questions.
- The first question may be slower because the model may still be loading into RAM.
- `uploads/` and `chroma_db/` are local runtime folders and are typically excluded from Git.
- This project uses modern LCEL composition instead of older legacy LangChain chain patterns.

## Future Improvements

- Show source chunks used for each answer
- Add multi-PDF support
- Add document reset/delete option
- Add loading states and progress feedback
- Add chat history persistence


You can add an MIT License if you want this to be open source.
