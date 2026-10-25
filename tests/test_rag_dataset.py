import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
from deepeval.contextvars import get_current_golden
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.tracing import observe, update_current_trace

from app.rag_agent import (
    build_vector_store,
    rag_support_agent_with_context,
)


@observe(name="rag_support_agent")
def traced_rag_agent(user_input: str) -> str:
    golden = get_current_golden()

    if golden and golden.expected_output:
        update_current_trace(
            expected_output=golden.expected_output
        )

    result = rag_support_agent_with_context(user_input)
    return result["answer"]


dataset = EvaluationDataset(
    goldens=[
        Golden(
            input="How much does the Premium plan cost?",
            expected_output=(
                "The Premium plan costs $25 per month and includes "
                "Basic features, advanced reporting, and up to five users."
            ),
        ),
        Golden(
            input="What happens if I cancel my subscription?",
            expected_output=(
                "You can cancel at any time. Access remains until the "
                "end of the paid period unless the account terms specify otherwise."
            ),
        ),
        Golden(
            input="Can I get a refund for my subscription payment?",
            expected_output=(
                "Subscription payments are not automatically refundable. "
                "Refund requests are reviewed under the applicable refund terms."
            ),
        ),
    ]
)


metrics = [
    ContextualPrecisionMetric(
        threshold=0.7,
        model="gpt-4.1-mini",
        include_reason=True,
    ),
    ContextualRecallMetric(
        threshold=0.7,
        model="gpt-4.1-mini",
        include_reason=True,
    ),
    AnswerRelevancyMetric(
        threshold=0.7,
        model="gpt-4.1-mini",
        include_reason=True,
    ),
    FaithfulnessMetric(
        threshold=0.7,
        model="gpt-4.1-mini",
        include_reason=True,
    ),
]


if __name__ == "__main__":
    build_vector_store()

    for golden in dataset.evals_iterator(metrics=metrics):
        traced_rag_agent(golden.input)
