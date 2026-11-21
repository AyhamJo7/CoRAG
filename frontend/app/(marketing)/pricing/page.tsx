import type { Metadata } from "next";

import { PLANS } from "@/config/plans";

export const metadata: Metadata = {
  title: "Pricing — CoRAG",
  description:
    "Plans for multi-hop question answering over your own documents.",
};

export default function PricingPage() {
  return (
    <main className="mx-auto min-h-screen w-full max-w-5xl px-6 py-20">
      <h1 className="text-center text-4xl font-semibold">Pricing</h1>
      <p className="mx-auto mt-4 max-w-xl text-center text-[var(--muted)]">
        Start with a free 14-day trial. Upgrade when your questions outgrow it.
      </p>

      <section className="mt-12 grid gap-4 md:grid-cols-4">
        {PLANS.map((plan) => (
          <div key={plan.id} className="rounded-lg border border-white/10 p-6">
            <h2 className="font-medium">{plan.name}</h2>
            <p className="mt-1 text-2xl font-semibold">
              {plan.priceEur === null ? "Free" : `€${plan.priceEur}`}
              {plan.priceEur !== null ? (
                <span className="text-sm font-normal text-[var(--muted)]">
                  /mo
                </span>
              ) : null}
            </p>
            <p className="mt-2 text-sm text-[var(--muted)]">{plan.blurb}</p>
            <ul className="mt-4 space-y-1 text-sm text-[var(--muted)]">
              <li>{plan.questions.toLocaleString()} questions / month</li>
              <li>{plan.documents.toLocaleString()} documents</li>
              <li>{plan.storage} storage</li>
            </ul>
            <a
              href="/sign-up"
              className="mt-6 block rounded bg-[var(--accent)] px-3 py-2 text-center text-sm font-medium text-black"
            >
              {plan.priceEur === null ? "Start free" : "Start trial"}
            </a>
          </div>
        ))}
      </section>
    </main>
  );
}
