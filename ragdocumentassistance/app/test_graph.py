from app.graph import rag_graph


question = input("Ask a question: ")


result = rag_graph.invoke(
    {
        "question": question,
        "documents": [],
        "answer": "",
        "relevance": "",
        "attempts": 0
    }
)


print("\nAnswer:")
print("-------")
print(result["answer"])


print("\nFinal Question Used:")
print("--------------------")
print(result["question"])


print("\nRetrieved Sources:")
print("------------------")


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

        print(
            f"Page {page + 1} | {source}"
        )