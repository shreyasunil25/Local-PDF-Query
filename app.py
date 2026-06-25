import os
import shutil
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
CHROMA_DIR = BASE_DIR / "chroma_db"

UPLOAD_FOLDER.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

vector_store = None
retriever = None
chain = None
current_pdf = None

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain_from_pdf(pdf_path: str):
    global vector_store, retriever, chain, current_pdf

    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(exist_ok=True)

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    prompt = PromptTemplate.from_template(
        """You are a precise PDF question-answering assistant.
Use only the context below to answer the question.
If the answer is not present in the context, say you could not find it in the uploaded PDF.

Context:
{context}

Question:
{question}

Answer:"""
    )

    llm = OllamaLLM(model=LLM_MODEL)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    current_pdf = os.path.basename(pdf_path)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_pdf():
    global current_pdf

    if "file" not in request.files:
        return jsonify({"error": "No file part found in request."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are allowed."}), 400

    filename = secure_filename(file.filename)
    save_path = UPLOAD_FOLDER / filename
    file.save(save_path)

    try:
        build_chain_from_pdf(str(save_path))
        return jsonify({
            "message": f"'{filename}' uploaded and indexed successfully. You may now ask questions about the document."
        })
    except Exception as e:
        return jsonify({"error": f"Failed to process PDF: {str(e)}"}), 500


@app.route("/ask", methods=["POST"])
def ask_question():
    global chain, current_pdf

    if chain is None:
      return jsonify({"error": "No PDF has been indexed yet. Please upload a PDF first."}), 400

    data = request.get_json(silent=True)

    if not data or "question" not in data:
        return jsonify({"error": "Request must include a 'question' field."}), 400

    question = data["question"].strip()

    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    try:
        answer = chain.invoke(question)
        return jsonify({
            "answer": answer,
            "document": current_pdf
        })
    except Exception as e:
        return jsonify({"error": f"Failed to generate answer: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)