from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

PDF_PATH = "document.pdf"
CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "nomic-embed-text"

print("Loading PDF...")
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()
print(f"Loaded {len(docs)} pages")

print("Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks")

print("Embedding and storing in ChromaDB...")
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)
print("Done. ChromaDB saved to ./chroma_db")