from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# must match what was used in ingest.py
VECTORSTORE_DIR = str(Path(__file__).parent.parent / "vectorstore")
COLLECTION_NAME = "sec_regulations"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
TOP_K = 5

# telling the model to only use the provided context and not make stuff up
SYSTEM_PROMPT = """You are a precise legal assistant for SEC regulatory documents.

Answer the user's question using ONLY the context provided below.
Do not use any outside knowledge.

If the answer is not present in the context, respond with exactly:
"I couldn't find information about that in the SEC documents provided."

Always end your answer by citing your sources in this format:
Sources: [filename, page X], [filename, page Y]

Context:
{context}"""


def format_context(docs: list) -> str:
    sections = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        filename = Path(source).name
        sections.append(f"[{filename}, page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)


def format_sources(docs: list) -> list[dict]:
    seen = set()
    sources = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        filename = Path(source).name
        key = (filename, page)
        if key not in seen:
            seen.add(key)
            sources.append({"file": filename, "page": page})
    return sources


def build_rag_chain():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # open the existing vectorstore, not creating a new one
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTORSTORE_DIR,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    # LCEL chain: retrieve context -> fill prompt -> call LLM -> parse output
    chain = (
        {"context": retriever | format_context, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return retriever, chain


class RAGQuery:
    # initializing once so we don't reload the model on every message
    def __init__(self):
        print("setting up rag chain...")
        self.retriever, self.chain = build_rag_chain()
        print("ready")

    def ask(self, question: str) -> dict:
        # retrieve separately so we can pull the source metadata
        docs = self.retriever.invoke(question)
        sources = format_sources(docs)
        answer = self.chain.invoke(question)
        return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    rag = RAGQuery()
    question = "What are the registration requirements under the Securities Act of 1933?"
    print(f"\nQuestion: {question}\n")
    result = rag.ask(question)
    print(f"Answer:\n{result['answer']}\n")
    print("Sources:")
    for s in result["sources"]:
        print(f"  - {s['file']}, page {s['page']}")
