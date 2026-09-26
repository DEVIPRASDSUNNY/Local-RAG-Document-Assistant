from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

response = llm.invoke(
    "Explain RAG in one simple sentence."
)

print(response.content)