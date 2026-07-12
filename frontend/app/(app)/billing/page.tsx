"use client";

import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";

import { PLANS } from "@/config/plans";

interface Usage {
  plan: string;
  subscription_status: string | null;
  trial_ends_at: string | null;
}

function BillingContent() {
  const params = useSearchParams();
  const checkout = params.get("checkout");
  const [usage, setUsage] = useState<Usage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState<string | null>(null);

  useEffect(() => {
    void (async () => {
      const res = await fetch("/api/core/usage", { cache: "no-store" });
      if (res.ok) setUsage((await res.json()) as Usage);
    })();
  }, []);

  async function startCheckout(plan: string) {
    setError(null);
    setPending(plan);
    const res = await fetch("/api/billing/checkout", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ plan }),
    });
    setPending(null);
    const body = (await res.json().catch(() => null)) as {
      url?: string;
      error?: string;
    } | null;
    if (!res.ok || !body?.url) {
      setError(body?.error ?? "Checkout could not be started.");
      return;
    }
    window.location.href = body.url;
  }

  async function openPortal() {
    setError(null);
    setPending("portal");
    const res = await fetch("/api/billing/portal", { method: "POST" });
    setPending(null);
    const body = (await res.json().catch(() => null)) as {
      url?: string;
      error?: string;
    } | null;
    if (!res.ok || !body?.url) {
      setError(body?.error ?? "Billing portal could not be opened.");
      return;
    }
    window.location.href = body.url;
  }

  return (
    <main className="mx-auto min-h-screen w-full max-w-4xl px-6 py-16">
      <header className="flex items-center justify-between">
        <h1 className="text-3xl font-semibold">Billing</h1>
        <a href="/dashboard" className="text-sm text-[var(--muted)] hover:text-white">
          ← Dashboard
        </a>
      </header>

      {checkout === "success" ? (
        <p className="mt-6 rounded border border-[var(--accent)]/40 bg-[var(--accent)]/10 p-4 text-sm">
          Payment received — your plan is being activated.
        </p>
      ) : null}
      {checkout === "cancelled" ? (
        <p className="mt-6 rounded border border-white/15 bg-white/5 p-4 text-sm text-[var(--muted)]">
          Checkout cancelled.
        </p>
      ) : null}
      {error ? <p className="mt-6 text-sm text-red-400">{error}</p> : null}

      {usage ? (
        <p className="mt-6 text-sm text-[var(--muted)]">
          Current plan: <span className="text-white">{usage.plan}</span>
          {usage.subscription_status
            ? ` (${usage.subscription_status})`
            : usage.trial_ends_at
              ? ` — trial ends ${new Date(usage.trial_ends_at).toLocaleDateString()}`
              : ""}
          {usage.subscription_status ? (
            <>
              {" · "}
              <button
                type="button"
                onClick={() => void openPortal()}
                disabled={pending === "portal"}
                className="text-[var(--accent)] underline disabled:opacity-60"
              >
                Manage subscription
              </button>
            </>
          ) : null}
        </p>
      ) : null}

      <section className="mt-10 grid gap-4 md:grid-cols-3">
        {PLANS.filter((p) => p.id !== "trial").map((plan) => (
          <div
            key={plan.id}
            className={`rounded-lg border p-6 ${
              usage?.plan === plan.id
                ? "border-[var(--accent)]"
                : "border-white/10"
            }`}
          >
            <h2 className="font-medium">{plan.name}</h2>
            <p className="mt-1 text-2xl font-semibold">
              €{plan.priceEur}
              <span className="text-sm font-normal text-[var(--muted)]">/mo</span>
            </p>
            <ul className="mt-4 space-y-1 text-sm text-[var(--muted)]">
              <li>{plan.questions.toLocaleString()} questions / month</li>
              <li>{plan.documents.toLocaleString()} documents</li>
              <li>{plan.storage} storage</li>
            </ul>
            <button
              type="button"
              onClick={() => void startCheckout(plan.id)}
              disabled={pending === plan.id || usage?.plan === plan.id}
              className="mt-6 w-full rounded bg-[var(--accent)] px-3 py-2 text-sm font-medium text-black disabled:opacity-60"
            >
              {usage?.plan === plan.id
                ? "Current plan"
                : pending === plan.id
                  ? "Starting…"
                  : `Choose ${plan.name}`}
            </button>
          </div>
        ))}
      </section>

      <p className="mt-8 text-xs text-[var(--muted)]">
        Payments are processed by Stripe. Cancel anytime from the billing
        portal.
      </p>
    </main>
  );
}

export default function BillingPage() {
  return (
    <Suspense>
      <BillingContent />
    </Suspense>
  );
}
