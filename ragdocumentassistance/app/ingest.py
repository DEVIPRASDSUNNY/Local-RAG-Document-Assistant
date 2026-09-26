from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


DATA_DIR = Path("documents")
VECTOR_DIR = "vectorstore"


def load_documents():
    documents = []

    for pdf_file in DATA_DIR.glob("*.pdf"):
        print(f"Loading: {pdf_file.name}")

        loader = PyPDFLoader(str(pdf_file))
        documents.extend(loader.load())

    return documents


def create_vector_store():

    documents = load_documents()

    if not documents:
        raise ValueError(
            "No PDF documents found in the documents/ folder."
        )

    print(f"\nLoaded {len(documents)} pages.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    print("\nCreating embeddings...")
    print("The embedding model may download the first time.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Creating Chroma vector store...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DIR
    )

    print("\nVector store created successfully!")


if __name__ == "__main__":
    create_vector_store()