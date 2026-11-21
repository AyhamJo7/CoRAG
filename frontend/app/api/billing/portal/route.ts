import { NextResponse } from "next/server";

import {
  activeTenantId,
  backendBilling,
  isStripeConfigured,
  stripeApi,
} from "@/server/billing";

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

  try {
    const res = await backendBilling(
      `/internal/billing/state?tenant_id=${tenantId}`,
      "GET",
    );
    if (!res.ok) throw new Error("state");
    const state = (await res.json()) as { stripe_customer_id: string | null };
    if (!state.stripe_customer_id) {
      return NextResponse.json(
        { error: "No active subscription." },
        { status: 400 },
      );
    }

    const host = req.headers.get("host") ?? "corag.iquantum.co";
    const session = await stripeApi<{ url: string }>(
      "/billing_portal/sessions",
      {
        customer: state.stripe_customer_id,
        return_url: `https://${host}/billing`,
      },
    );
    return NextResponse.json({ url: session.url });
  } catch {
    return NextResponse.json(
      { error: "Billing portal could not be opened." },
      { status: 502 },
    );
  }
}
