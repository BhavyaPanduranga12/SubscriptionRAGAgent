
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

from app.config import OPENAI_EMBEDDING_MODEL
from app.knowledge_base import get_policy_documents


# Create the embedding model using our configured OpenAI model.
embeddings = OpenAIEmbeddings(
    model=OPENAI_EMBEDDING_MODEL
)

# Create an in-memory vector store.
vector_store = InMemoryVectorStore(embedding=embeddings)


def build_vector_store():
    """Convert policy documents into embeddings and index them."""
    documents = get_policy_documents()

    vector_store.add_documents(documents)

    return vector_store


def retrieve_policies(question: str, k: int = 3):
    """Retrieve the most relevant policies for a question."""
    results = vector_store.similarity_search(
        question,
        k=k,
    )

    return results

def retrieve_policies_with_scores(question: str, k: int = 3):
    """Retrieve policy documents along with similarity scores."""
    return vector_store.similarity_search_with_score(
        question,
        k=k,
    )

