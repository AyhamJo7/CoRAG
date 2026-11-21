"use client";

import { type FormEvent, useState } from "react";

interface Citation {
  id: string;
  title: string;
  url: string;
  chunk_id: string;
}

interface AskStats {
  num_steps: number;
  num_chunks: number;
  latency_ms: number;
}

type Phase = "idle" | "asking" | "done" | "error";

function parseSseEvents(buffer: string): {
  events: { name: string; data: string }[];
  rest: string;
} {
  const events: { name: string; data: string }[] = [];
  const parts = buffer.split("\n\n");
  const rest = parts.pop() ?? "";
  for (const part of parts) {
    let name = "message";
    let data = "";
    for (const line of part.split("\n")) {
      if (line.startsWith("event: ")) name = line.slice(7).trim();
      if (line.startsWith("data: ")) data += line.slice(6);
    }
    if (data) events.push({ name, data });
  }
  return { events, rest };
}

export default function AskPage() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [answer, setAnswer] = useState("");
  const [citations, setCitations] = useState<Citation[]>([]);
  const [stats, setStats] = useState<AskStats | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const question = String(form.get("question") ?? "").trim();
    if (!question) return;

    setPhase("asking");
    setAnswer("");
    setCitations([]);
    setStats(null);
    setError(null);

    const res = await fetch("/api/core/ask", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (res.status === 402) {
      const body = (await res.json().catch(() => null)) as {
        detail?: { message?: string };
      } | null;
      setError(body?.detail?.message ?? "Your plan limit is reached.");
      setPhase("error");
      return;
    }
    if (!res.ok || !res.body) {
      setError("The question could not be processed. Please try again.");
      setPhase("error");
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const { events, rest } = parseSseEvents(buffer);
      buffer = rest;
      for (const evt of events) {
        const data = JSON.parse(evt.data) as Record<string, unknown>;
        if (evt.name === "answer") setAnswer(String(data.text ?? ""));
        if (evt.name === "citations")
          setCitations((data.items ?? []) as Citation[]);
        if (evt.name === "done") {
          setStats(data as unknown as AskStats);
          setPhase("done");
        }
        if (evt.name === "error") {
          setError(String(data.message ?? "Unknown error"));
          setPhase("error");
        }
      }
    }
    setPhase((p) => (p === "asking" ? "done" : p));
  }

  return (
    <main className="mx-auto min-h-screen w-full max-w-3xl px-6 py-16">
      <header className="flex items-center justify-between">
        <h1 className="text-3xl font-semibold">Ask</h1>
        <a href="/dashboard" className="text-sm text-[var(--muted)] hover:text-white">
          ← Dashboard
        </a>
      </header>

      <form onSubmit={onSubmit} className="mt-8">
        <textarea
          name="question"
          rows={3}
          maxLength={2000}
          required
          placeholder="Ask a complex question over your documents — CoRAG retrieves iteratively until the evidence is sufficient."
          className="w-full rounded border border-white/15 bg-white/5 px-4 py-3 outline-none focus:border-[var(--accent)]"
        />
        <button
          type="submit"
          disabled={phase === "asking"}
          className="mt-3 rounded bg-[var(--accent)] px-5 py-2 font-medium text-black disabled:opacity-60"
        >
          {phase === "asking" ? "Retrieving evidence…" : "Ask"}
        </button>
      </form>

      {error ? (
        <div className="mt-8 rounded border border-red-400/40 bg-red-400/10 p-4 text-sm">
          {error}{" "}
          {error.includes("plan") || error.includes("Upgrade") ? (
            <a href="/billing" className="text-[var(--accent)] underline">
              View plans
            </a>
          ) : null}
        </div>
      ) : null}

      {answer ? (
        <article className="mt-10">
          <div className="whitespace-pre-wrap leading-relaxed">{answer}</div>
          {citations.length > 0 ? (
            <div className="mt-6">
              <h2 className="text-sm uppercase tracking-widest text-[var(--muted)]">
                Sources
              </h2>
              <ul className="mt-2 flex flex-wrap gap-2">
                {citations.map((c) => (
                  <li
                    key={c.id}
                    className="rounded-full border border-white/15 px-3 py-1 text-sm"
                    title={c.chunk_id}
                  >
                    [{c.id}] {c.title}
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
          {stats ? (
            <p className="mt-6 text-xs text-[var(--muted)]">
              {stats.num_steps} retrieval steps · {stats.num_chunks} chunks ·{" "}
              {(stats.latency_ms / 1000).toFixed(1)}s
            </p>
          ) : null}
        </article>
      ) : null}
    </main>
  );
}
