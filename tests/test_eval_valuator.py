
from deepeval.test_case import LLMTestCase

from deepeval.metrics import FaithfulnessMetric

def test_faithfulness_detects_false_refund_guarantee():
    test_case = LLMTestCase(
        input="Can I get a refund for my subscription?",
        actual_output=(
            "Every subscription payment is guaranteed "
            "to receive a full refund."
        ),
        retrieval_context=[
            "Subscription payments are not automatically refundable. "
            "Refund requests are reviewed under the applicable refund terms."
        ],
    )

    metric = FaithfulnessMetric(
        threshold=0.7,
        model="gpt-4.1-mini",
        include_reason=True,
    )

    metric.measure(test_case)

    print("\nFaithfulness score:", metric.score)
    print("Evaluation reason:", metric.reason)

    assert metric.score < 0.7, (
        "Expected DeepEval to flag the unsupported refund guarantee."
    )

if __name__ == "__main__":
    test_faithfulness_detects_false_refund_guarantee()

# Faithfulness score: 0.0
# Evaluation reason: The score is 0.00 because the actual output incorrectly 
# claims that every subscription payment is guaranteed a full refund, 
# which directly contradicts the context stating that subscription payments are not
# automatically refundable.
