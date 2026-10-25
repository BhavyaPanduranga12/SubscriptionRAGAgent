
from langchain_core.documents import Document


POLICY_DOCS = [
    Document(
        page_content=(
            "Basic subscription plan: The Basic plan costs $10 per month. "
            "It includes access to standard features and one user account."
        ),
        metadata={"category": "plans", "policy_id": "PLAN-001"},
    ),
    Document(
        page_content=(
            "Premium subscription plan: The Premium plan costs $25 per month. "
            "It includes all Basic features, advanced reporting, and up to "
            "five user accounts."
        ),
        metadata={"category": "plans", "policy_id": "PLAN-002"},
    ),
    Document(
        page_content=(
            "Billing policy: Monthly subscriptions renew automatically "
            "on the same calendar day each month where possible. The customer "
            "is charged using the saved payment method."
        ),
        metadata={"category": "billing", "policy_id": "BILL-001"},
    ),
    Document(
        page_content=(
            "Upgrade policy: Customers can upgrade from Basic to Premium "
            "at any time. Any prorated charge depends on the remaining "
            "billing period and the applicable subscription calculation."
        ),
        metadata={"category": "upgrades", "policy_id": "UPG-001"},
    ),
    Document(
        page_content=(
            "Downgrade policy: Customers can request a downgrade from "
            "Premium to Basic. The downgrade takes effect at the next "
            "billing period unless the account explicitly confirms otherwise."
        ),
        metadata={"category": "downgrades", "policy_id": "DOWN-001"},
    ),
    Document(
        page_content=(
            "Cancellation policy: Customers can cancel their subscription "
            "at any time. Access remains active until the end of the paid "
            "billing period, unless the account terms specify otherwise."
        ),
        metadata={"category": "cancellations", "policy_id": "CANCEL-001"},
    ),
    Document(
        page_content=(
            "Refund policy: Subscription payments are not automatically "
            "refundable. Refund requests must be reviewed under the applicable "
            "refund terms. Do not promise approval before review."
        ),
        metadata={"category": "refunds", "policy_id": "REFUND-001"},
    ),
    Document(
        page_content=(
            "Payment failure policy: If a subscription renewal payment fails, "
            "the customer should check their payment method and contact support "
            "if the issue continues. Do not assume a specific retry schedule "
            "unless it is documented."
        ),
        metadata={"category": "payments", "policy_id": "PAY-001"},
    ),
    Document(
        page_content=(
            "Account privacy policy: Customer-specific subscription details, "
            "payment information, and personal data must not be disclosed "
            "without appropriate authentication and authorization."
        ),
        metadata={"category": "security", "policy_id": "SEC-001"},
    ),
]


def get_policy_documents() -> list[Document]:
    """Return the subscription policy documents."""
    return POLICY_DOCS
