import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Mentioning the paths
DATA_PATH = "Baseline_Papers"
CHROMA_PATH = "chroma_db"

def build_rag():
    print("1. Loading the PDFs:")
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages in total.")

    # Check for PDF reading
    # for page in documents:
    #     print(page)

    print("2. Chunking Text:")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1600,
        chunk_overlap = 300,
        length_function = len,
        is_separator_regex=False
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    print("3. Embedding and saving using Chroma:")
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # Saving the vector database
    db = Chroma.from_documents(
        chunks,
        embedding_model,
        persist_directory=CHROMA_PATH
    )

    print(f'Database saved in {CHROMA_PATH} directory')


if __name__ == '__main__':
    build_rag()