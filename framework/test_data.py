CONTEXTUAL_PRECISION_TEST_CASES = [
    {
        "type": "true",
        "question": "Can I downgrade from Premium to Basic immediately?",
        "expected_output": (
            "The downgrade takes effect in the next billing period, "
            "unless the account explicitly confirms otherwise."
        ),
    },
    {
        "type": "partial",
        "question": (
            "Can I downgrade from Premium to Basic today "
            "if my account terms allow an immediate change?"
        ),
        "expected_output": (
            "The standard policy says the downgrade takes effect "
            "in the next billing period, unless the account explicitly "
            "confirms otherwise."
        ),
    },
    {
        "type": "negative",
        "question": (
            "Can I downgrade from Premium to Basic and keep "
            "unlimited users and priority support?"
        ),
        "expected_output": (
            "The knowledge base does not specify whether these "
            "features remain available after a downgrade."
        ),
    },
]


CONTEXTUAL_RECALL_TEST_CASES = [
    {
        "type": "true",
        "question": "What should I do if my subscription renewal payment fails?",
        "expected_output": (
            "Check the payment method and contact support if the "
            "issue continues. The policy does not specify a retry schedule."
        ),
    },
    {
        "type": "partial",
        "question": (
            "If my renewal payment fails, when will the system "
            "retry charging my card?"
        ),
        "expected_output": (
            "The policy advises checking the payment method and "
            "contacting support if the issue continues, but it does "
            "not specify when the system retries the payment."
        ),
    },
    {
        "type": "negative",
        "question": (
            "Will the system retry my failed renewal payment "
            "exactly three times, every 24 hours?"
        ),
        "expected_output": (
            "The knowledge base does not specify the number or "
            "frequency of payment retries."
        ),
    },
]