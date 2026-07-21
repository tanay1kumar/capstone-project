from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PDF_DIR = Path(__file__).parent.parent / "data" / "sec_regulations"
VECTORSTORE_DIR = str(Path(__file__).parent.parent / "vectorstore")
COLLECTION_NAME = "sec_regulations"

# chunk size / overlap - tweaked these values after some testing
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# using a local embedding model so we don't burn API credits on ingestion
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


def load_pdfs(pdf_dir: Path) -> list:
    documents = []
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")

    for pdf_path in pdf_files:
        print(f"loading {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        documents.extend(pages)
        print(f"  {len(pages)} pages")

    print(f"total pages: {len(documents)}")
    return documents


def chunk_documents(documents: list) -> list:
    # tries to split on paragraph breaks first, then newlines, then sentences
    # overlap means adjacent chunks share some text so context isn't cut off
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    print(f"split into {len(chunks)} chunks")
    return chunks


def build_vectorstore(chunks: list) -> Chroma:
    print("loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("embedding chunks, this takes a bit...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=VECTORSTORE_DIR,
    )

    print(f"done. {len(chunks)} chunks saved to vectorstore/")
    return vectorstore


def main():
    print("starting ingestion...\n")
    documents = load_pdfs(PDF_DIR)
    chunks = chunk_documents(documents)
    build_vectorstore(chunks)
    print("\nall done, run the chatbot now")


if __name__ == "__main__":
    main()
