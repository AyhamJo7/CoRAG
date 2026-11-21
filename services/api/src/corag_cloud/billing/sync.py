"""Subscription state sync from Stripe webhook events.

Rules (ported from the sibling products):
- idempotency: the event id is inserted into stripe_event first; a conflict
  means the event was already applied and the call is a no-op
- entitlements are SET from the event, never incremented
- questions_used resets only when the billing period advances
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from corag_cloud.db.pool import tenant_connection

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SyncOutcome:
    duplicate: bool
    updated: bool


async def apply_stripe_event(
    *,
    event_id: str,
    event_type: str,
    tenant_id: UUID,
    plan: str | None = None,
    stripe_customer_id: str | None = None,
    stripe_subscription_id: str | None = None,
    subscription_status: str | None = None,
    period_end: datetime | None = None,
    cancel: bool = False,
) -> SyncOutcome:
    """Apply one verified Stripe event to the tenant's billing state."""
    async with tenant_connection(tenant_id) as conn:
        claimed = await conn.fetchrow(
            "INSERT INTO stripe_event (id, type) VALUES ($1, $2) "
            "ON CONFLICT (id) DO NOTHING RETURNING id",
            event_id,
            event_type,
        )
        if claimed is None:
            logger.info("Stripe event %s already processed", event_id)
            return SyncOutcome(duplicate=True, updated=False)

        tenant = await conn.fetchrow(
            "SELECT current_period_end FROM tenant WHERE id = $1", tenant_id
        )
        if tenant is None:
            # Unknown/foreign tenant id in event metadata; the event stays
            # recorded so Stripe does not retry forever.
            logger.warning(
                "Stripe event %s references unknown tenant %s", event_id, tenant_id
            )
            return SyncOutcome(duplicate=False, updated=False)

        if cancel:
            await conn.execute(
                "UPDATE tenant SET subscription_status = 'canceled' WHERE id = $1",
                tenant_id,
            )
            return SyncOutcome(duplicate=False, updated=True)

        period_advanced = (
            period_end is not None and period_end != tenant["current_period_end"]
        )
        await conn.execute(
            """
            UPDATE tenant SET
                plan = COALESCE($2, plan),
                stripe_customer_id = COALESCE($3, stripe_customer_id),
                stripe_subscription_id = COALESCE($4, stripe_subscription_id),
                subscription_status = COALESCE($5, subscription_status),
                current_period_end = COALESCE($6, current_period_end),
                questions_used = CASE WHEN $7 THEN 0 ELSE questions_used END
            WHERE id = $1
            """,
            tenant_id,
            plan,
            stripe_customer_id,
            stripe_subscription_id,
            subscription_status,
            period_end,
            period_advanced,
        )
        return SyncOutcome(duplicate=False, updated=True)
