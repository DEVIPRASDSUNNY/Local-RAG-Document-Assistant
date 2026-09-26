from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from app.rag import (
    retrieve_documents,
    generate_answer,
)

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# RAG State
# --------------------------------------------------

class RAGState(TypedDict):
    question: str
    documents: list
    answer: str
    relevance: str
    attempts: int


# --------------------------------------------------
# Local Llama model
# --------------------------------------------------

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


# --------------------------------------------------
# 1. Retrieve documents
# --------------------------------------------------

def retrieve_node(state: RAGState):

    documents = retrieve_documents(
        state["question"]
    )

    return {
        "documents": documents,
        "attempts": state.get("attempts", 0) + 1
    }


# --------------------------------------------------
# 2. Check whether retrieved documents are relevant
# --------------------------------------------------

relevance_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a relevance checker.

Determine whether the provided documents contain
information that can help answer the user's question.

Respond with ONLY one word:

YES

or

NO

Do not provide explanations.
"""
        ),
        (
            "human",
            """Question:
{question}

Documents:
{documents}
"""
        )
    ]
)


def relevance_node(state: RAGState):

    context = "\n\n".join(
        document.page_content
        for document in state["documents"]
    )

    messages = relevance_prompt.invoke(
        {
            "question": state["question"],
            "documents": context
        }
    )

    response = llm.invoke(messages)

    result = response.content.strip().upper()

    if result == "YES":
        relevance = "yes"
    else:
        relevance = "no"

    return {
        "relevance": relevance
    }


# --------------------------------------------------
# 3. Rewrite the question
# --------------------------------------------------

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You rewrite questions for document retrieval.

Rewrite the user's question so that it is clearer
and more specific for searching the provided document.

Return ONLY the rewritten question.
Do not answer the question.
"""
        ),
        (
            "human",
            "Original question:\n{question}"
        )
    ]
)


def rewrite_node(state: RAGState):

    messages = rewrite_prompt.invoke(
        {
            "question": state["question"]
        }
    )

    response = llm.invoke(messages)

    new_question = response.content.strip()

    return {
        "question": new_question
    }


# --------------------------------------------------
# 4. Generate final answer
# --------------------------------------------------

def generate_node(state: RAGState):

    answer = generate_answer(
        state["question"],
        state["documents"]
    )

    return {
        "answer": answer
    }


# --------------------------------------------------
# 5. Handle questions not found in documents
# --------------------------------------------------

def not_found_node(state: RAGState):

    return {
        "answer": (
            "I could not find the answer in the provided documents."
        ),
        "documents": []
    }


# --------------------------------------------------
# 6. Decide what happens after relevance check
# --------------------------------------------------

def relevance_router(state: RAGState):

    # Relevant documents → generate answer
    if state["relevance"] == "yes":
        return "generate"

    # If we already tried twice, stop
    if state["attempts"] >= 2:
        return "not_found"

    # Otherwise rewrite the question
    return "rewrite"


# --------------------------------------------------
# Build LangGraph
# --------------------------------------------------

graph_builder = StateGraph(RAGState)


# Add nodes

graph_builder.add_node(
    "retrieve",
    retrieve_node
)

graph_builder.add_node(
    "check_relevance",
    relevance_node
)

graph_builder.add_node(
    "rewrite",
    rewrite_node
)

graph_builder.add_node(
    "generate",
    generate_node
)

graph_builder.add_node(
    "not_found",
    not_found_node
)


# --------------------------------------------------
# Define graph flow
# --------------------------------------------------

# START → Retrieve

graph_builder.add_edge(
    START,
    "retrieve"
)


# Retrieve → Relevance Check

graph_builder.add_edge(
    "retrieve",
    "check_relevance"
)


# Relevance Check → Conditional Routing

graph_builder.add_conditional_edges(
    "check_relevance",
    relevance_router,
    {
        "generate": "generate",
        "rewrite": "rewrite",
        "not_found": "not_found"
    }
)


# Rewrite → Retrieve again

graph_builder.add_edge(
    "rewrite",
    "retrieve"
)


# Generate → END

graph_builder.add_edge(
    "generate",
    END
)


# Not Found → END

graph_builder.add_edge(
    "not_found",
    END
)


# --------------------------------------------------
# Compile graph
# --------------------------------------------------

rag_graph = graph_builder.compile()