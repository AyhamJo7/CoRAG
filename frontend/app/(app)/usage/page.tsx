"use client";

import { useEffect, useState } from "react";

interface Usage {
  plan: string;
  subscription_status: string | null;
  trial_ends_at: string | null;
  questions_used: number;
  questions_limit: number;
  docs_count: number;
  docs_limit: number;
  storage_bytes_used: number;
  storage_bytes_limit: number;
}

function Bar({ used, limit }: { used: number; limit: number }) {
  const pct = Math.min(100, Math.round((used / Math.max(limit, 1)) * 100));
  return (
    <div className="mt-2 h-2 w-full rounded bg-white/10">
      <div
        className={`h-2 rounded ${pct >= 100 ? "bg-red-400" : "bg-[var(--accent)]"}`}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

function formatBytes(n: number): string {
  if (n >= 1024 * 1024 * 1024) return `${(n / 1024 ** 3).toFixed(1)} GB`;
  if (n >= 1024 * 1024) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024).toFixed(0)} KB`;
}

export default function UsagePage() {
  const [usage, setUsage] = useState<Usage | null>(null);

  useEffect(() => {
    void (async () => {
      const res = await fetch("/api/core/usage", { cache: "no-store" });
      if (res.ok) setUsage((await res.json()) as Usage);
    })();
  }, []);

  return (
    <main className="mx-auto min-h-screen w-full max-w-3xl px-6 py-16">
      <header className="flex items-center justify-between">
        <h1 className="text-3xl font-semibold">Usage</h1>
        <a href="/dashboard" className="text-sm text-[var(--muted)] hover:text-white">
          ← Dashboard
        </a>
      </header>

      {usage ? (
        <section className="mt-10 space-y-8">
          <p className="text-sm text-[var(--muted)]">
            Plan: <span className="text-white">{usage.plan}</span>
            {usage.plan === "trial" && usage.trial_ends_at
              ? ` · trial ends ${new Date(usage.trial_ends_at).toLocaleDateString()}`
              : ""}
          </p>
          <div>
            <div className="flex justify-between text-sm">
              <span>Questions this period</span>
              <span className="text-[var(--muted)]">
                {usage.questions_used} / {usage.questions_limit}
              </span>
            </div>
            <Bar used={usage.questions_used} limit={usage.questions_limit} />
          </div>
          <div>
            <div className="flex justify-between text-sm">
              <span>Documents</span>
              <span className="text-[var(--muted)]">
                {usage.docs_count} / {usage.docs_limit}
              </span>
            </div>
            <Bar used={usage.docs_count} limit={usage.docs_limit} />
          </div>
          <div>
            <div className="flex justify-between text-sm">
              <span>Storage</span>
              <span className="text-[var(--muted)]">
                {formatBytes(usage.storage_bytes_used)} /{" "}
                {formatBytes(usage.storage_bytes_limit)}
              </span>
            </div>
            <Bar
              used={usage.storage_bytes_used}
              limit={usage.storage_bytes_limit}
            />
          </div>
        </section>
      ) : (
        <p className="mt-10 text-sm text-[var(--muted)]">Loading…</p>
      )}
    </main>
  );
}
