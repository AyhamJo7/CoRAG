"use client";

import { type FormEvent, useCallback, useEffect, useState } from "react";

interface ApiKey {
  id: string;
  name: string;
  key_prefix: string;
  created_at: string;
  last_used_at: string | null;
  revoked_at: string | null;
}

export default function ApiKeysPage() {
  const [keys, setKeys] = useState<ApiKey[]>([]);
  const [freshKey, setFreshKey] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  const refresh = useCallback(async () => {
    const res = await fetch("/api/core/api-keys", { cache: "no-store" });
    if (res.ok) setKeys((await res.json()) as ApiKey[]);
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  async function onCreate(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    setError(null);
    setPending(true);
    const res = await fetch("/api/core/api-keys", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ name: String(form.get("name") ?? "") }),
    });
    setPending(false);
    if (!res.ok) {
      const body = (await res.json().catch(() => null)) as {
        detail?: string;
      } | null;
      setError(
        typeof body?.detail === "string"
          ? body.detail
          : "Key could not be created.",
      );
      return;
    }
    const created = (await res.json()) as ApiKey & { key: string };
    setFreshKey(created.key);
    (event.target as HTMLFormElement).reset?.();
    await refresh();
  }

  async function onRevoke(id: string) {
    await fetch(`/api/core/api-keys/${id}`, { method: "DELETE" });
    await refresh();
  }

  return (
    <main className="mx-auto min-h-screen w-full max-w-3xl px-6 py-16">
      <header className="flex items-center justify-between">
        <h1 className="text-3xl font-semibold">API keys</h1>
        <a href="/dashboard" className="text-sm text-[var(--muted)] hover:text-white">
          ← Dashboard
        </a>
      </header>

      <p className="mt-4 text-sm text-[var(--muted)]">
        Use a key as{" "}
        <code className="rounded bg-white/10 px-1">
          Authorization: Bearer corag_live_…
        </code>{" "}
        against{" "}
        <code className="rounded bg-white/10 px-1">POST /v1/ask</code> (SSE
        response, shares your plan quota).
      </p>

      <form onSubmit={onCreate} className="mt-8 flex items-center gap-3">
        <input
          name="name"
          placeholder="Key name (e.g. production)"
          maxLength={100}
          className="flex-1 rounded border border-white/15 bg-white/5 px-3 py-2 text-sm outline-none focus:border-[var(--accent)]"
        />
        <button
          type="submit"
          disabled={pending}
          className="rounded bg-[var(--accent)] px-4 py-2 text-sm font-medium text-black disabled:opacity-60"
        >
          {pending ? "Creating…" : "Create key"}
        </button>
      </form>
      {error ? <p className="mt-3 text-sm text-red-400">{error}</p> : null}

      {freshKey ? (
        <div className="mt-6 rounded border border-[var(--accent)]/40 bg-[var(--accent)]/10 p-4">
          <p className="text-sm">
            Copy this key now — it will not be shown again.
          </p>
          <code className="mt-2 block break-all rounded bg-black/40 p-3 text-sm">
            {freshKey}
          </code>
        </div>
      ) : null}

      <ul className="mt-8 divide-y divide-white/10">
        {keys.map((key) => (
          <li key={key.id} className="flex items-center justify-between py-4">
            <div>
              <p className="font-medium">
                {key.name || "Unnamed key"}{" "}
                <code className="text-sm text-[var(--muted)]">
                  {key.key_prefix}…
                </code>
              </p>
              <p className="mt-0.5 text-sm text-[var(--muted)]">
                created {new Date(key.created_at).toLocaleDateString()}
                {key.last_used_at
                  ? ` · last used ${new Date(key.last_used_at).toLocaleString()}`
                  : " · never used"}
              </p>
            </div>
            {key.revoked_at ? (
              <span className="text-sm text-red-400">revoked</span>
            ) : (
              <button
                type="button"
                onClick={() => void onRevoke(key.id)}
                className="text-sm text-[var(--muted)] hover:text-red-400"
              >
                Revoke
              </button>
            )}
          </li>
        ))}
        {keys.length === 0 ? (
          <li className="py-8 text-sm text-[var(--muted)]">No keys yet.</li>
        ) : null}
      </ul>
    </main>
  );
}
