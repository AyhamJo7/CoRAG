import { NextResponse } from "next/server";

import {
  backendBilling,
  planForPrice,
  verifyStripeWebhook,
} from "@/server/billing";

// Public endpoint — Stripe calls it. Authenticity comes from the signature,
// not a session. tenant_id + plan are read from the (signed) event metadata,
// so no customer→tenant lookup is needed. Idempotency lives in the backend:
// the event id is recorded before state is applied.

export const runtime = "nodejs";

function str(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function metaObject(value: unknown): Record<string, unknown> {
  return value && typeof value === "object"
    ? (value as Record<string, unknown>)
    : {};
}

async function sync(payload: Record<string, unknown>): Promise<void> {
  const res = await backendBilling("/internal/billing/sync", "POST", payload);
  if (!res.ok) throw new Error(`sync ${res.status}`);
}

export async function POST(req: Request): Promise<NextResponse> {
  const secret = process.env.STRIPE_WEBHOOK_SECRET;
  if (!secret) {
    return NextResponse.json(
      { error: "webhook not configured" },
      { status: 503 },
    );
  }

  const payload = await req.text();
  const event = verifyStripeWebhook(
    payload,
    req.headers.get("stripe-signature"),
    secret,
  );
  if (!event) {
    return NextResponse.json({ error: "invalid signature" }, { status: 400 });
  }

  try {
    if (event.type === "checkout.session.completed") {
      const session = event.data.object;
      const meta = metaObject(session["metadata"]);
      const tenantId =
        str(meta["tenant_id"]) ?? str(session["client_reference_id"]);
      const plan = str(meta["plan"]);
      if (tenantId && plan) {
        await sync({
          event_id: event.id,
          event_type: event.type,
          tenant_id: tenantId,
          plan,
          stripe_customer_id: str(session["customer"]) ?? null,
          stripe_subscription_id: str(session["subscription"]) ?? null,
          subscription_status: "active",
        });
      }
    } else if (event.type === "customer.subscription.updated") {
      // Renewal or plan change. Map the *current* price → plan (metadata can
      // go stale after a portal plan change); the backend resets usage only
      // when the period advances.
      const subscription = event.data.object;
      const tenantId = str(metaObject(subscription["metadata"])["tenant_id"]);
      const items = subscription["items"];
      const dataArr =
        items && typeof items === "object"
          ? (items as { data?: unknown }).data
          : undefined;
      const first = Array.isArray(dataArr)
        ? (dataArr[0] as Record<string, unknown> | undefined)
        : undefined;
      const price =
        first && typeof first["price"] === "object"
          ? (first["price"] as Record<string, unknown>)
          : undefined;
      const plan = planForPrice(str(price?.["id"]));
      const periodEndRaw = subscription["current_period_end"];
      const periodEnd =
        typeof periodEndRaw === "number"
          ? new Date(periodEndRaw * 1000).toISOString()
          : null;
      if (tenantId && plan) {
        await sync({
          event_id: event.id,
          event_type: event.type,
          tenant_id: tenantId,
          plan,
          stripe_subscription_id: str(subscription["id"]) ?? null,
          subscription_status: str(subscription["status"]) ?? "active",
          period_end: periodEnd,
        });
      }
    } else if (event.type === "customer.subscription.deleted") {
      const subscription = event.data.object;
      const tenantId = str(metaObject(subscription["metadata"])["tenant_id"]);
      if (tenantId) {
        await sync({
          event_id: event.id,
          event_type: event.type,
          tenant_id: tenantId,
          cancel: true,
        });
      }
    }
  } catch {
    // Non-2xx lets Stripe retry with backoff.
    return NextResponse.json({ error: "processing failed" }, { status: 500 });
  }

  return NextResponse.json({ received: true });
}
