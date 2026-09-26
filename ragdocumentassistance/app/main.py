from fastapi import FastAPI
from pydantic import BaseModel

from app.graph import rag_graph


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Local RAG Document Assistant",
    description=(
        "A local RAG API using "
        "LangGraph, Chroma, Ollama and Llama 3.2"
    ),
    version="1.0.0"
)


# --------------------------------------------------
# Request model
# --------------------------------------------------

class QuestionRequest(BaseModel):
    question: str


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "Local RAG Document Assistant is running"
    }


# --------------------------------------------------
# Ask question
# --------------------------------------------------

@app.post("/ask")
def ask_question(request: QuestionRequest):

    result = rag_graph.invoke(
        {
            "question": request.question,
            "documents": [],
            "answer": "",
            "relevance": "",
            "attempts": 0
        }
    )

    # ----------------------------------------------
    # Prepare sources
    # ----------------------------------------------

    sources = []

    seen = set()

    for document in result["documents"]:

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        page = document.metadata.get(
            "page",
            "Unknown"
        )

        key = (source, page)

        if key not in seen:

            seen.add(key)

            sources.append(
                {
                    "page": page + 1,
                    "source": source
                }
            )

    # ----------------------------------------------
    # Return API response
    # ----------------------------------------------

    return {
    "question": request.question,
    "answer": result["answer"],
    "sources": sources
    }