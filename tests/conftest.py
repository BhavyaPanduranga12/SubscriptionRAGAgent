import pytest

from deepeval.metrics import ContextualRelevancyMetric
from app.retriever import build_vector_store
from framework.metrics import create_contextual_precision_metric, create_contextual_recall_metric, create_contextual_relevancy_metric

@pytest.fixture(scope="session", autouse=True)
def initialize_vector_store():
    #Initialize the vector store once per pytest session.
    build_vector_store()

@pytest.fixture()
def contextual_relevancy_metric():
    #Create a Contextual Relevancy metric for a test.
    return create_contextual_relevancy_metric()

@pytest.fixture()
def contextual_precision_metric():
    return create_contextual_precision_metric()

@pytest.fixture()
def contextual_recall_metric():
    return create_contextual_recall_metric()