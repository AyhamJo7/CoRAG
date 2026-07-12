import { NextResponse } from "next/server";
import { z } from "zod";

import {
  activeTenantId,
  isStripeConfigured,
  priceForPlan,
  stripeApi,
} from "@/server/billing";

const bodySchema = z.object({ plan: z.enum(["starter", "pro", "team"]) });

export async function POST(req: Request): Promise<NextResponse> {
  if (!isStripeConfigured()) {
    return NextResponse.json(
      { error: "Billing is not available right now." },
      { status: 503 },
    );
  }

  const tenantId = await activeTenantId(req);
  if (!tenantId) {
    return NextResponse.json({ error: "Not signed in." }, { status: 401 });
  }

  let plan: z.infer<typeof bodySchema>["plan"];
  try {
    plan = bodySchema.parse(await req.json()).plan;
  } catch {
    return NextResponse.json({ error: "Invalid plan." }, { status: 422 });
  }

  const price = priceForPlan(plan);
  if (!price) {
    return NextResponse.json(
      { error: "Plan is not configured." },
      { status: 500 },
    );
  }

  const host = req.headers.get("host") ?? "corag.iquantum.co";
  const base = `https://${host}`;

  try {
    // Stripe creates the customer at checkout; the webhook links it and
    // applies the plan. tenant_id + plan travel in the metadata so the
    // webhook needs no lookup.
    const session = await stripeApi<{ id: string; url: string }>(
      "/checkout/sessions",
      {
        mode: "subscription",
        "line_items[0][price]": price,
        "line_items[0][quantity]": "1",
        client_reference_id: tenantId,
        "metadata[tenant_id]": tenantId,
        "metadata[plan]": plan,
        "subscription_data[metadata][tenant_id]": tenantId,
        "subscription_data[metadata][plan]": plan,
        allow_promotion_codes: "true",
        success_url: `${base}/billing?checkout=success`,
        cancel_url: `${base}/billing?checkout=cancelled`,
      },
    );
    return NextResponse.json({ url: session.url });
  } catch {
    return NextResponse.json(
      { error: "Checkout could not be started." },
      { status: 502 },
    );
  }
}
