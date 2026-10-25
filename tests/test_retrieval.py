
import pytest

from app.retriever import build_vector_store, retrieve_policies, retrieve_policies_with_scores


@pytest.fixture(scope="module", autouse=True)
def setup_vector_store():
    """Build the policy index before running retrieval tests."""
    build_vector_store()


@pytest.mark.parametrize(
    "question, expected_policy_id",
    [
        (
            "What happens if I cancel my subscription?",
            "CANCEL-001",
        ),
        (
            "Can I get a refund for my payment?",
            "REFUND-001",
        ),
        (
            "How much does the Premium plan cost?",
            "PLAN-002",
        ),
        (
            "What happens if my renewal payment fails?",
            "PAY-001",
        ),
        (
            "Can I upgrade from Basic to Premium?",
            "UPG-001",
        ),
        (
            "I want to stop my subscription. What happens next?",
            "CANCEL-001",
        ),
        (
            "Will I get my money back if I request a refund?",
            "REFUND-001",
        ),
        (
            "My payment didn't go through when renewing. What should I do?",
            "PAY-001",
        ),
    ],
)

def test_expected_policy_is_retrieved(question, expected_policy_id):
    results = retrieve_policies_with_scores(question, k=3)

    retrieved_policy_ids = []

    for rank, (document, score) in enumerate(results, start=1):
        policy_id = document.metadata["policy_id"]
        retrieved_policy_ids.append(policy_id)

        print(
            f"\nRank: {rank} | "
            f"Policy: {policy_id} | "
            f"Score: {score:.4f}"
        )

    assert expected_policy_id in retrieved_policy_ids, (
        f"Expected policy {expected_policy_id} was not retrieved.\n"
        f"Question: {question}\n"
        f"Retrieved policies: {retrieved_policy_ids}"
    )

    expected_rank = retrieved_policy_ids.index(expected_policy_id) + 1

    assert expected_rank == 1, (
        f"Expected {expected_policy_id} at Rank 1, "
        f"but found it at Rank {expected_rank}"
)


# platform win32 -- Python 3.13.2, pytest-9.1.1, pluggy-1.7.0 -- C:\Users\Bhavya Panduranga\Documents\SubscriptionRAGAgent\.venv\Scripts\python.exe
# cachedir: .pytest_cache
# rootdir: C:\Users\Bhavya Panduranga\Documents\SubscriptionRAGAgent
# plugins: anyio-4.15.1, deepeval-4.3.1, langsmith-0.14.12, asyncio-1.4.0, repeat-0.9.4, rerunfailures-16.7, xdist-3.8.0
# asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
# collected 5 items                                                                                                            

# tests/test_retrieval.py::test_expected_policy_is_retrieved[What happens if I cancel my subscription?-CANCEL-001] 
# Rank: 1 | Policy: CANCEL-001 | Score: 0.6445

# Rank: 2 | Policy: REFUND-001 | Score: 0.4618

# Rank: 3 | Policy: PAY-001 | Score: 0.4489
# PASSED
# tests/test_retrieval.py::test_expected_policy_is_retrieved[Can I get a refund for my payment?-REFUND-001] 
# Rank: 1 | Policy: REFUND-001 | Score: 0.5119

# Rank: 2 | Policy: CANCEL-001 | Score: 0.3873

# Rank: 3 | Policy: PAY-001 | Score: 0.3743
# PASSED
# tests/test_retrieval.py::test_expected_policy_is_retrieved[How much does the Premium plan cost?-PLAN-002] 
# Rank: 1 | Policy: PLAN-002 | Score: 0.6877

# Rank: 2 | Policy: PLAN-001 | Score: 0.6029

# Rank: 3 | Policy: UPG-001 | Score: 0.5614
# PASSED
# tests/test_retrieval.py::test_expected_policy_is_retrieved[What happens if my renewal payment fails?-PAY-001] 
# Rank: 1 | Policy: PAY-001 | Score: 0.6904

# Rank: 2 | Policy: BILL-001 | Score: 0.4513

# Rank: 3 | Policy: CANCEL-001 | Score: 0.4039
# PASSED
# tests/test_retrieval.py::test_expected_policy_is_retrieved[Can I upgrade from Basic to Premium?-UPG-001] 
# Rank: 1 | Policy: UPG-001 | Score: 0.7560

# Rank: 2 | Policy: DOWN-001 | Score: 0.6746

# Rank: 3 | Policy: PLAN-002 | Score: 0.5624
# PASSEDRunning teardown with pytest sessionfinish...
