from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from generation import get_llm

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def build_vector_store(chunks):
    return FAISS.from_documents(chunks, get_embeddings())

def get_retriever(vector_store):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

def get_reranking_retriever(vector_store):
    # 1. Fetch the top 20 candidates using normal similarity search
    base_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 20},
    )

    # 2. The Cross-Encoder reads all 20 and scores them against the question
    encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    
    # 3. Keep only the top 4 best chunks
    reranker = CrossEncoderReranker(model=encoder, top_n=4)

    # 4. Combine them
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever,
    )
def get_multiquery_reranking_retriever(vector_store):
    # 1. Base Vector Search (fetch 20 chunks)
    base_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 20},
    )
    
    # 2. Multi-Query: Generate 3 versions of the question
    multiquery_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=get_llm(),
    )
    
    # 3. Reranker: Pick the absolute best 4 chunks
    encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    reranker = CrossEncoderReranker(model=encoder, top_n=4)
    
    # 4. Combine everything
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=multiquery_retriever,
    )