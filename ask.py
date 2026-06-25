from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"

embeddings = OllamaEmbeddings(model=EMBED_MODEL)
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

prompt = PromptTemplate.from_template("""Use the following context to answer the question.
If the answer is not in the context, say "I don't know based on the document."

Context:
{context}

Question: {question}

Answer:""")

llm = OllamaLLM(model=LLM_MODEL)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("PDF QnA ready. Type 'exit' to quit.\n")
while True:
    question = input("You: ").strip()
    if question.lower() in ("exit", "quit"):
        break
    if not question:
        continue
    answer = chain.invoke(question)
    print(f"\nAnswer: {answer}\n")