"""Plan catalog: quotas per subscription tier.

Prices/Stripe price IDs live in env on the frontend (checkout); this module
is the backend source of truth for what each plan allows. Keep in sync with
frontend/src/config/plans.ts.
"""

from dataclasses import dataclass

MB = 1024 * 1024
GB = 1024 * MB


@dataclass(frozen=True)
class PlanLimits:
    questions_per_month: int
    max_documents: int
    max_storage_bytes: int


PLAN_LIMITS: dict[str, PlanLimits] = {
    "trial": PlanLimits(
        questions_per_month=25, max_documents=10, max_storage_bytes=20 * MB
    ),
    "starter": PlanLimits(
        questions_per_month=200, max_documents=50, max_storage_bytes=200 * MB
    ),
    "pro": PlanLimits(
        questions_per_month=1_000, max_documents=250, max_storage_bytes=1 * GB
    ),
    "team": PlanLimits(
        questions_per_month=5_000, max_documents=1_000, max_storage_bytes=5 * GB
    ),
}

PAID_PLANS = frozenset({"starter", "pro", "team"})

# Per-file ceiling regardless of plan (extraction happens in-request).
MAX_UPLOAD_BYTES = 10 * MB


def limits_for(plan: str) -> PlanLimits:
    """Limits for a plan; unknown plans fall back to trial limits."""
    return PLAN_LIMITS.get(plan, PLAN_LIMITS["trial"])


def is_valid_plan(plan: str) -> bool:
    return plan in PLAN_LIMITS
