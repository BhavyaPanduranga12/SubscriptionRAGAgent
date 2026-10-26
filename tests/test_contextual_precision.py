# Contextual Precision Metric
# Evaluates whether relevant retrieved information is ranked higher
# than irrelevant information in the retrieval context.
# Contextual Precision - Did the retriever rank relevant information effectively?
# It uses an LLM as a judge to evaluate the relationship between:
#  1. input — the question asked by the user.
#  2. expected_output — the reference answer describing the information needed.
#  3. retrieval_context — the information retrieved from the knowledge base,
#     where the order of relevant and irrelevant documents matters.

import os , sys
import pytest
from framework.test_data import CONTEXTUAL_PRECISION_TEST_CASES
from app.rag_agent import rag_support_agent_with_context
from deepeval.test_case import LLMTestCase

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

@pytest.mark.parametrize("case", CONTEXTUAL_PRECISION_TEST_CASES)
def test_contextual_recall_agent_response(case, contextual_precision_metric ):
    response = rag_support_agent_with_context(case["question"])

    test_case = LLMTestCase(
        input = case["question"],
        actual_output = response["answer"],
        expected_output = case["expected_output"],
        retrieval_context = response["retrieval_context"] 
    )

    metric = contextual_precision_metric
    metric.measure(test_case)

    print(
            f"Question: {case['question']}\n"
            f"Metric score: {metric.score}\n"
            f"Metric reason: {metric.reason}\n"
            f"Agent Answer: {response['answer']}\n" 
            f"Retrived context: {response['retrieval_context']}"
        )
    
    for i, context in enumerate(response["retrieval_context"], start=1):
        print(f"\nDocument {i}:")
        print(context)
    
    assert metric.is_successful() , (
            f"Question: {case['question']}\n"
            f"Score: {metric.score}\n"
            f"Reason: {metric.reason}\n"
            f"Answer: {response['answer']}"
        )

# platform win32 -- Python 3.13.2, pytest-9.1.1, pluggy-1.7.0 -- C:\Users\Bhavya Panduranga\Documents\SubscriptionRAGAgent\.venv\Scripts\python.exe
# cachedir: .pytest_cache
# rootdir: C:\Users\Bhavya Panduranga\Documents\SubscriptionRAGAgent
# plugins: anyio-4.15.1, deepeval-4.3.1, langsmith-0.14.12, asyncio-1.4.0, repeat-0.9.4, rerunfailures-16.7, xdist-3.8.0
# asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
# collected 3 items                                                                                                               

# tests/test_contextual_precision.py::test_contextual_recall_agent_response[case0] [Confident AI Trace Log]  No Confident AI API key found. Skipping trace posting.
# Question: Can I downgrade from Premium to Basic immediately?
# Metric score: 1.0
# Metric reason: The score is 1.00 because the first node in retrieval contexts, which is relevant, is ranked highest and clearly explains the downgrade timing with the quote, 'The downgrade takes effect at the next billing period unless the account explicitly confirms otherwise.' The subsequent nodes, ranked lower, are irrelevant as they discuss upgrading and plan details unrelated to downgrading, ensuring perfect contextual precision.
# Agent Answer: Downgrades from Premium to Basic take effect at the next billing period unless you explicitly confirm you want it immediately. You can request the downgrade and specify if you want it to happen right away.
# Retrived context: ['Downgrade policy: Customers can request a downgrade from Premium to Basic. The downgrade takes effect at the next billing period unless the account explicitly confirms otherwise.', 'Upgrade policy: Customers can upgrade from Basic to Premium at any time. Any prorated charge depends on the remaining billing period and the applicable subscription calculation.', 'Basic subscription plan: The Basic plan costs $10 per month. It includes access to standard features and one user account.']

# Document 1:
# Downgrade policy: Customers can request a downgrade from Premium to Basic. The downgrade takes effect at the next billing period unless the account explicitly confirms otherwise.

# Document 2:
# Upgrade policy: Customers can upgrade from Basic to Premium at any time. Any prorated charge depends on the remaining billing period and the applicable subscription calculation.

# Document 3:
# Basic subscription plan: The Basic plan costs $10 per month. It includes access to standard features and one user account.
# PASSED

# tests/test_contextual_precision.py::test_contextual_recall_agent_response[case1] [Confident AI Trace Log]  No Confident AI API key found. Skipping trace posting.
# Question: Can I downgrade from Premium to Basic today if my account terms allow an immediate change?
# Metric score: 1.0
# Metric reason: The score is 1.00 because the first node in retrieval contexts is highly relevant, directly addressing the downgrade policy with the statement 'The downgrade takes effect at the next billing period unless the account explicitly confirms otherwise.' The subsequent nodes, ranked lower, discuss unrelated topics such as upgrading and cancellation policies, which justifies their lower ranking and results in perfect contextual precision.
# Agent Answer: Yes, you can downgrade from Premium to Basic today if your account terms explicitly allow an immediate change. Otherwise, the downgrade takes effect at the next billing period.
# Retrived context: ['Downgrade policy: Customers can request a downgrade from Premium to Basic. The downgrade takes effect at the next billing period unless the account explicitly confirms otherwise.', 'Upgrade policy: Customers can upgrade from Basic to Premium at any time. Any prorated charge depends on the remaining billing period and the applicable subscription calculation.', 'Cancellation policy: Customers can cancel their subscription at any time. Access remains active until the end of the paid billing period, unless the account terms specify otherwise.']

# Document 1:
# Downgrade policy: Customers can request a downgrade from Premium to Basic. The downgrade takes effect at the next billing period unless the account explicitly confirms otherwise.

# Document 2:
# Upgrade policy: Customers can upgrade from Basic to Premium at any time. Any prorated charge depends on the remaining billing period and the applicable subscription calculation.

# Document 3:
# Cancellation policy: Customers can cancel their subscription at any time. Access remains active until the end of the paid billing period, unless the account terms specify otherwise.
# PASSED

# tests/test_contextual_precision.py::test_contextual_recall_agent_response[case2] [Confident AI Trace Log]  No Confident AI API key found. Skipping trace posting.
# Question: Can I downgrade from Premium to Basic and keep unlimited users and priority support?
# Metric score: 0.8333333333333333
# Metric reason: The score is 0.83 because the first node in the retrieval contexts correctly ranks a relevant context about downgrading from Premium to Basic at the top, which is crucial for answering the question. However, the second node, which is irrelevant as it discusses upgrading rather than downgrading, is ranked higher than the third relevant node that details Premium plan features important for understanding feature retention. This misplacement of an irrelevant node before a relevant one lowers the score from perfect.
# Agent Answer: Downgrading from Premium to Basic is possible, but the Basic plan does not include unlimited users or priority support. Those features are part of the Premium plan.
# Retrived context: ['Downgrade policy: Customers can request a downgrade from Premium to Basic. The downgrade takes effect at the next billing period unless the account explicitly confirms otherwise.', 'Upgrade policy: Customers can upgrade from Basic to Premium at any time. Any prorated charge depends on the remaining billing period and the applicable subscription calculation.', 'Premium subscription plan: The Premium plan costs $25 per month. It includes all Basic features, advanced reporting, and up to five user accounts.']

# Document 1:
# Downgrade policy: Customers can request a downgrade from Premium to Basic. The downgrade takes effect at the next billing period unless the account explicitly confirms otherwise.

# Document 2:
# Upgrade policy: Customers can upgrade from Basic to Premium at any time. Any prorated charge depends on the remaining billing period and the applicable subscription calculation.

# Document 3:
# Premium subscription plan: The Premium plan costs $25 per month. It includes all Basic features, advanced reporting, and up to five user accounts.
# PASSEDRunning teardown with pytest sessionfinish...