from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_PATH = "chroma_db"

def test_database():
    print("1. Loading embedding model and the database")
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model
    )

    print("2. Testing the RAG pipeline's performance with an example query")

    queries = [
        "How do residual blocks solve the vanishing gradient problem?",
        "How does NeRF represent a 3D scene using a neural network and volume rendering?",
        "How does U-Net use skip connections between encoder and decoder for segmentation?"
    ]

    for query in queries:
        print(f"Query: {query}\n")
        results = db.similarity_search_with_score(query, k=3)
        for i, (content, score) in enumerate(results):
            print(f"--- Result {i+1} ---")
            print(f"Distance Score (lower the better): {score:.4f}")
            print(f"Source file: {content.metadata.get('source', 'Unknown')}")
            print(f"Page Number: {content.metadata.get('page', 'Unknown')}")
            print(f"Text Chunk: {content.page_content}\n")
        print("=" * 60 + "\n")

if __name__ == '__main__':
    test_database()
