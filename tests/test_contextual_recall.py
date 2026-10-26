# Contextual Recall Metric
# Evaluates whether the retrieved context contains the information needed
# to support the expected output.
# Contextual Recall - Did the retriever retrieve the necessary information?
# It uses an LLM as a judge to evaluate the relationship between:
#  1. input — the question asked by the user.
#  2. expected_output — the reference answer containing the required information.
#  3. retrieval_context — the information your retriever found in the knowledge base.

import os , sys
import pytest
from framework.test_data import CONTEXTUAL_RECALL_TEST_CASES
from app.rag_agent import rag_support_agent_with_context
from deepeval.test_case import LLMTestCase

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

@pytest.mark.parametrize("case", CONTEXTUAL_RECALL_TEST_CASES)
def test_contextual_recall_agent_response(case, contextual_recall_metric ):
    response = rag_support_agent_with_context(case["question"])

    test_case = LLMTestCase(
        input = case["question"],
        actual_output = response["answer"],
        expected_output = case["expected_output"],
        retrieval_context = response["retrieval_context"] 
    )

    metric = contextual_recall_metric
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

# tests/test_contextual_recall.py::test_contextual_recall_agent_response[case0] [Confident AI Trace Log]  No Confident AI API key found. Skipping trace posting.
# Question: What should I do if my subscription renewal payment fails?
# Metric score: 1.0
# Metric reason: The score is 1.00 because the expected output perfectly aligns with the information in node 1 of the retrieval context, accurately reflecting the advice to check the payment method and contact support, as well as the absence of a specified retry schedule in the policy. Great job capturing all relevant details!
# Agent Answer: If your subscription renewal payment fails, please check your payment method for any issues. If the problem continues, contact support for further assistance.
# Retrived context: ['Payment failure policy: If a subscription renewal payment fails, the customer should check their payment method and contact support if the issue continues. Do not assume a specific retry schedule unless it is documented.', 'Billing policy: Monthly subscriptions renew automatically on the same calendar day each month where possible. The customer is charged using the saved payment method.', 'Refund policy: Subscription payments are not automatically refundable. Refund requests must be reviewed under the applicable refund terms. Do not promise approval before review.']

# Document 1:
# Payment failure policy: If a subscription renewal payment fails, the customer should check their payment method and contact support if the issue continues. Do not assume a specific retry schedule unless it is documented.

# Document 2:
# Billing policy: Monthly subscriptions renew automatically on the same calendar day each month where possible. The customer is charged using the saved payment method.

# Document 3:
# Refund policy: Subscription payments are not automatically refundable. Refund requests must be reviewed under the applicable refund terms. Do not promise approval before review.
# PASSED

# tests/test_contextual_recall.py::test_contextual_recall_agent_response[case1] [Confident AI Trace Log]  No Confident AI API key found. Skipping trace posting.
# Question: If my renewal payment fails, when will the system retry charging my card?
# Metric score: 1.0
# Metric reason: The score is 1.00 because the expected output accurately reflects the information from node 1 in the retrieval context, clearly stating the advice on checking payment methods and contacting support without assuming retry schedules, with no unsupported content present.
# Agent Answer: The policy does not specify a retry schedule for failed renewal payments. If your payment fails, please check your payment method and contact support if the issue continues.
# Retrived context: ['Payment failure policy: If a subscription renewal payment fails, the customer should check their payment method and contact support if the issue continues. Do not assume a specific retry schedule unless it is documented.', 'Billing policy: Monthly subscriptions renew automatically on the same calendar day each month where possible. The customer is charged using the saved payment method.', 'Refund policy: Subscription payments are not automatically refundable. Refund requests must be reviewed under the applicable refund terms. Do not promise approval before review.']

# Document 1:
# Payment failure policy: If a subscription renewal payment fails, the customer should check their payment method and contact support if the issue continues. Do not assume a specific retry schedule unless it is documented.

# Document 2:
# Billing policy: Monthly subscriptions renew automatically on the same calendar day each month where possible. The customer is charged using the saved payment method.

# Document 3:
# Refund policy: Subscription payments are not automatically refundable. Refund requests must be reviewed under the applicable refund terms. Do not promise approval before review.
# PASSED

# tests/test_contextual_recall.py::test_contextual_recall_agent_response[case2] [Confident AI Trace Log]  No Confident AI API key found. Skipping trace posting.
# Question: Will the system retry my failed renewal payment exactly three times, every 24 hours?
# Metric score: 1.0
# Metric reason: The score is 1.00 because the first sentence in the expected output directly reflects the information from node 1 in the retrieval context, which clearly states not to assume a specific retry schedule unless documented. There are no unsupported sentences, making the summary fully aligned with the retrieval context. Great job capturing the key point accurately!
# Agent Answer: The policy does not specify a retry schedule for failed renewal payments. If your payment fails, please check your payment method and contact support if the issue continues.
# Retrived context: ['Payment failure policy: If a subscription renewal payment fails, the customer should check their payment method and contact support if the issue continues. Do not assume a specific retry schedule unless it is documented.', 'Billing policy: Monthly subscriptions renew automatically on the same calendar day each month where possible. The customer is charged using the saved payment method.', 'Refund policy: Subscription payments are not automatically refundable. Refund requests must be reviewed under the applicable refund terms. Do not promise approval before review.']

# Document 1:
# Payment failure policy: If a subscription renewal payment fails, the customer should check their payment method and contact support if the issue continues. Do not assume a specific retry schedule unless it is documented.

# Document 2:
# Billing policy: Monthly subscriptions renew automatically on the same calendar day each month where possible. The customer is charged using the saved payment method.

# Document 3:
# Refund policy: Subscription payments are not automatically refundable. Refund requests must be reviewed under the applicable refund terms. Do not promise approval before review.
# PASSEDRunning teardown with pytest sessionfinish...