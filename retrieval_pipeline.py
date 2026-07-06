from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
# Inicializando DataBase
persistent_directory = "db/chroma_db"

# Inicializando embbending model    1 IA
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)


pergunta = "What was NVIDIA's first graphics accelerator called?"


# === FORMA 1 (sem valor de similiaridade base)
#retriever = db.as_retriever(search_kwargs={"k": 5}) # 5 significa retornar os top 5 chunks de maior similiaridade com o prompt do usuario

# === FORMA 2 (com valor de similaridade base)
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.6  # So retorna valores com similaridade do coseno >= 0.6
    }
)

relevant_docs = retriever.invoke(pergunta)

print(relevant_docs)

if relevant_docs:

    print(f"\nPergunta do usuario: {pergunta}")
    # Exibe resultados
    print("--- Contexto ---\n")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Documento {i}:\n{doc.page_content[:100]}...\n")

    # Combina a pergunta e os documentos para fornecer um prompt melhor para a ia
    combined_input = f"""Based on the following documents, please answer this question: {pergunta}

    Documents:
    {chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

    Please provide a clear, helpful answer using only the information from these documents.
    """

else:
    # Criar modelo de IA de fallback 2 IA
    search_tool= TavilySearchResults(max_results=3)

    web_search = search_tool.invoke(pergunta)

    print(f"\nPergunta do usuario: {pergunta}")
    # Exibe resultados
    print("--- Contexto ---\n")

    print("--- Contexto ---\n")
    for i, item in enumerate(web_search, 1):
        print(f"=== FONTE WEB {i} ===")
        print(f"Fonte: {item['url']}")
        print(f"Conteúdo: {item['content'][:100]}...\n")

    # Combina a pergunta e os documentos para fornecer um prompt melhor para a ia
    combined_input = f"""Based on the following documents, please answer this question: {pergunta}

    Documents:
    {chr(10).join([f"- {item['content']}" for item in web_search])}

    Please provide a clear, helpful answer using only the information from these documents.
    """

# Criar o modelo de IA llama3   3 IA
model = OllamaLLM(model="llama3")

# Define o comportamento da IA
messages = [
    SystemMessage(content="You are a helpful assistant."), # Define o que a IA e
    HumanMessage(content=combined_input), # Define o que a IA vai responder
]

# Invoca o modelo de IA junto aos documentos para dar a respostta
result = model.invoke(messages)

# Resposta da LLM
print("\n--- RESPOSTA ---")
print(result)

# Outras perguntas: 

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"
