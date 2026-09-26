from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


VECTOR_DIR = "vectorstore"


# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Load Chroma vector database
vector_store = Chroma(
    persist_directory=VECTOR_DIR,
    embedding_function=embeddings
)


# Retrieve more chunks so the model has better context
retriever = vector_store.as_retriever(
    search_kwargs={"k": 6}
)


# Local Llama model
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


# Grounded RAG prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a document question-answering assistant.

Your task is to answer questions using ONLY the provided document context.

Rules:
1. Use ONLY the provided document context.
2. Do not use outside knowledge.
3. Do not invent or infer missing information.
4. If the requested value is not explicitly available in the context,
   say:
   "The provided documents do not contain enough information to answer this question."
5. For questions asking for a TOTAL, do NOT calculate a total from a
   partial table.
6. Only calculate a total when ALL required values are clearly present
   in the provided context.
7. If a table is incomplete or appears truncated, state that the
   available context is incomplete.
8. If the document explicitly contains a "Total" value, use that value
   instead of calculating it yourself.
9. When numbers are involved, clearly identify the numbers used in
   your reasoning.
Context:
{context}
"""
        ),
        ("human", "{question}")
    ]
)


def retrieve_documents(question: str):
    """Retrieve relevant document chunks."""
    return retriever.invoke(question)


def generate_answer(question: str, documents):
    """Generate an answer from retrieved document chunks."""

    context_parts = []

    for document in documents:
        page = document.metadata.get("page", "Unknown")

        context_parts.append(
            f"[Page {page + 1}]\n{document.page_content}"
        )

    context = "\n\n".join(context_parts)

    messages = prompt.invoke(
        {
            "context": context,
            "question": question
        }
    )

    response = llm.invoke(messages)

    return response.content


def display_sources(documents):
    """Display the pages used for the answer."""

    print("\nSources:")
    print("--------")

    seen_pages = set()

    for document in documents:
        page = document.metadata.get("page", "Unknown")
        source = document.metadata.get("source", "Unknown")

        key = (source, page)

        if key not in seen_pages:
            seen_pages.add(key)

            print(
                f"Page {page + 1} | {source}"
            )