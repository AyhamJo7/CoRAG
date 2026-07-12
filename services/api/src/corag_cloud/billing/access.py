"""Access gates enforced at the processing boundaries (upload, ask).

Failures map to HTTP 402 with a machine-readable ``code`` so the frontend
can render the right upgrade prompt.
"""

from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException

from corag_cloud.billing.plans import limits_for

ACTIVE_STATUSES = {"active", "trialing"}


def _payment_required(code: str, message: str) -> HTTPException:
    return HTTPException(status_code=402, detail={"code": code, "message": message})


def require_active_access(tenant: dict[str, Any] | Any) -> None:
    """The tenant must be on a valid trial or an active subscription."""
    plan = tenant["plan"]
    status = tenant["subscription_status"]
    if status in ACTIVE_STATUSES:
        return
    if plan == "trial":
        trial_ends_at = tenant["trial_ends_at"]
        if trial_ends_at is not None and trial_ends_at > datetime.now(UTC):
            return
        raise _payment_required(
            "trial_expired", "Your trial has ended. Choose a plan to continue."
        )
    raise _payment_required(
        "subscription_inactive",
        "Your subscription is not active. Update billing to continue.",
    )


def require_document_capacity(tenant: dict[str, Any] | Any, add_bytes: int) -> None:
    """Reject uploads that would exceed the plan's document/storage caps."""
    limits = limits_for(tenant["plan"])
    if tenant["docs_count"] >= limits.max_documents:
        raise _payment_required(
            "document_limit",
            f"Your plan allows {limits.max_documents} documents. Upgrade to add more.",
        )
    if tenant["storage_bytes_used"] + add_bytes > limits.max_storage_bytes:
        raise _payment_required(
            "storage_limit",
            "This upload exceeds your plan's storage. Upgrade to add more.",
        )


def require_question_quota(tenant: dict[str, Any] | Any) -> None:
    """Reject questions once the monthly quota is spent."""
    limits = limits_for(tenant["plan"])
    if tenant["questions_used"] >= limits.questions_per_month:
        raise _payment_required(
            "question_quota",
            f"You've used all {limits.questions_per_month} questions in this "
            "period. Upgrade for more.",
        )
