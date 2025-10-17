from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

if __name__ == "__main__":
    # docs = load_all_documents("data")
    # docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    #store.build_from_documents(docs)
    store.load()
    #print(store.query("What is attention mechanism?", top_k=3))
    rag_search = RAGSearch()
    query = "hi"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
