
from deepeval import assert_test
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
)
from deepeval.test_case import LLMTestCase

from app.rag_agent import (
    build_vector_store,
    rag_support_agent_with_context,
)


def test_cancellation_answer_quality():
    """Check the quality of an answer about cancellation."""

    # Arrange
    build_vector_store()

    question = "What happens if I cancel my subscription?"

    # Act
    result = rag_support_agent_with_context(question)

    # Create a DeepEval test case
    test_case = LLMTestCase(
        input=question,
        actual_output=result["answer"],
        retrieval_context=result["retrieval_context"],
    )

    # Evaluate the answer
    faithfulness = FaithfulnessMetric(
        threshold=0.7,
        model="gpt-4.1-mini",
    )

    answer_relevancy = AnswerRelevancyMetric(
        threshold=0.7,
        model="gpt-4.1-mini",
    )

    assert_test(
        test_case,
        [faithfulness, answer_relevancy],
    )


def test_agent_does_not_invent_support_hours():
    """Check whether the agent acknowledges missing policy information."""

    # Arrange
    build_vector_store()

    question = (
        "What time does the subscription service's "
        "support team close?"
    )

    # Act
    result = rag_support_agent_with_context(question)

    answer = result["answer"].lower()

    # Check that the agent acknowledges missing information.
    expected_phrases = [
        "not specified",
        "don't have information",
        "do not have information",
        "not available",
        "doesn't specify",
        "does not specify",
        "cannot determine",
        "can't determine",
        "unable to determine",
        "don't know",
        "do not know",
        "does not include information",
    ]

    assert any(
        phrase in answer
        for phrase in expected_phrases
    ), (
        "Agent did not acknowledge missing information. "
        f"Answer: {result['answer']}"
    )

    # Evaluate whether the answer is grounded in retrieved context.
    test_case = LLMTestCase(
        input=question,
        actual_output=result["answer"],
        retrieval_context=result["retrieval_context"],
    )

    faithfulness = FaithfulnessMetric(
        threshold=0.7,
        model="gpt-4.1-mini",
    )

    assert_test(
        test_case,
        [faithfulness],
    )
