from app.rag import (
    retrieve_documents,
    generate_answer,
    display_sources
)


question = input("Ask a question: ")

documents = retrieve_documents(question)

answer = generate_answer(
    question,
    documents
)

print("\nAnswer:")
print("-------")
print(answer)

display_sources(documents)