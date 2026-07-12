// Server-only Stripe helpers for the billing routes. Uses the Stripe REST API
// via fetch (no SDK dependency) and verifies webhooks with Node crypto. Never
// runs on the client. All tenant state changes are applied through the
// backend's internal billing endpoints, never a direct DB write.

import { createHmac, timingSafeEqual } from "node:crypto";

import { auth } from "@/auth";

const STRIPE_API = "https://api.stripe.com/v1";
const WEBHOOK_TOLERANCE_SECONDS = 60 * 5;

export type PaidPlan = "starter" | "pro" | "team";

const PLAN_PRICE_ENV: Record<PaidPlan, string> = {
  starter: "STRIPE_PRICE_STARTER",
  pro: "STRIPE_PRICE_PRO",
  team: "STRIPE_PRICE_TEAM",
};

export function isStripeConfigured(): boolean {
  return Boolean(process.env.STRIPE_SECRET_KEY);
}

export function priceForPlan(plan: PaidPlan): string | undefined {
  const value = process.env[PLAN_PRICE_ENV[plan]];
  return value && value.length > 0 ? value : undefined;
}

export function planForPrice(priceId: string | undefined): PaidPlan | null {
  if (!priceId) return null;
  for (const plan of Object.keys(PLAN_PRICE_ENV) as PaidPlan[]) {
    if (priceForPlan(plan) === priceId) return plan;
  }
  return null;
}

export function isPaidPlan(value: string): value is PaidPlan {
  return value === "starter" || value === "pro" || value === "team";
}

/** Resolve the authenticated user's active, membership-checked tenant, or null. */
export async function activeTenantId(req: Request): Promise<string | null> {
  const session = await auth();
  const tenants = session?.user?.tenants ?? [];
  if (tenants.length === 0) return null;
  const requested = req.headers.get("x-tenant-id");
  if (!requested) return tenants[0].id;
  return tenants.some((t) => t.id === requested) ? requested : null;
}

/** POST form-encoded to the Stripe REST API with the secret key. */
export async function stripeApi<T = Record<string, unknown>>(
  path: string,
  params: Record<string, string>,
): Promise<T> {
  const key = process.env.STRIPE_SECRET_KEY;
  if (!key) throw new Error("STRIPE_SECRET_KEY not set");
  const res = await fetch(`${STRIPE_API}${path}`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${key}`,
      "content-type": "application/x-www-form-urlencoded",
    },
    body: new URLSearchParams(params).toString(),
  });
  const json = (await res.json()) as T & { error?: { message?: string } };
  if (!res.ok) throw new Error(json.error?.message ?? `stripe ${res.status}`);
  return json;
}

/** Call a backend internal billing endpoint (internal token; server-only). */
export async function backendBilling(
  path: string,
  method: "GET" | "POST",
  body?: unknown,
): Promise<Response> {
  const base = process.env.BACKEND_URL?.replace(/\/$/, "");
  const token = process.env.INTERNAL_SERVICE_TOKEN;
  if (!base || !token) throw new Error("backend not configured");
  return fetch(`${base}${path}`, {
    method,
    headers: {
      "x-internal-token": token,
      ...(body !== undefined ? { "content-type": "application/json" } : {}),
    },
    body: body !== undefined ? JSON.stringify(body) : undefined,
    cache: "no-store",
  });
}

export interface StripeEvent {
  id: string;
  type: string;
  data: { object: Record<string, unknown> };
}

/** Verify a Stripe webhook signature (t=…,v1=…) and return the parsed event,
 * or null if the signature is missing/invalid/stale. */
export function verifyStripeWebhook(
  payload: string,
  sigHeader: string | null,
  secret: string,
): StripeEvent | null {
  if (!sigHeader) return null;
  const parts = Object.fromEntries(
    sigHeader.split(",").map((kv) => {
      const [k, v] = kv.split("=");
      return [k, v] as const;
    }),
  );
  const timestamp = parts["t"];
  const signature = parts["v1"];
  if (!timestamp || !signature) return null;

  const age = Math.abs(Math.floor(Date.now() / 1000) - Number(timestamp));
  if (!Number.isFinite(age) || age > WEBHOOK_TOLERANCE_SECONDS) return null;

  const expected = createHmac("sha256", secret)
    .update(`${timestamp}.${payload}`)
    .digest("hex");
  const a = Buffer.from(expected);
  const b = Buffer.from(signature);
  if (a.length !== b.length || !timingSafeEqual(a, b)) return null;

  try {
    return JSON.parse(payload) as StripeEvent;
  } catch {
    return null;
  }
}
