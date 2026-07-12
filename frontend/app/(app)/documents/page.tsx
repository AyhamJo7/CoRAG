"use client";

import { type FormEvent, useCallback, useEffect, useRef, useState } from "react";

interface DocumentRow {
  id: string;
  title: string;
  filename: string;
  mime: string;
  size_bytes: number;
  status: "uploaded" | "processing" | "indexed" | "failed";
  error: string | null;
  created_at: string;
}

const STATUS_STYLES: Record<DocumentRow["status"], string> = {
  uploaded: "text-[var(--muted)]",
  processing: "text-amber-400",
  indexed: "text-[var(--accent)]",
  failed: "text-red-400",
};

function formatBytes(n: number): string {
  if (n >= 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${n} B`;
}

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<DocumentRow[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    const res = await fetch("/api/core/documents", { cache: "no-store" });
    if (res.ok) setDocuments((await res.json()) as DocumentRow[]);
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Poll while anything is still being indexed.
  useEffect(() => {
    const pending = documents.some(
      (d) => d.status === "uploaded" || d.status === "processing",
    );
    if (!pending) return;
    const timer = setInterval(() => void refresh(), 4000);
    return () => clearInterval(timer);
  }, [documents, refresh]);

  async function onUpload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const file = fileInput.current?.files?.[0];
    if (!file) return;
    setError(null);
    setUploading(true);
    const body = new FormData();
    body.append("file", file);
    const res = await fetch("/api/core/documents", { method: "POST", body });
    setUploading(false);
    if (!res.ok) {
      const detail = (await res.json().catch(() => null)) as {
        detail?: { message?: string } | string;
      } | null;
      const message =
        typeof detail?.detail === "string"
          ? detail.detail
          : (detail?.detail?.message ?? "Upload failed.");
      setError(message);
      return;
    }
    if (fileInput.current) fileInput.current.value = "";
    await refresh();
  }

  async function onDelete(id: string) {
    await fetch(`/api/core/documents/${id}`, { method: "DELETE" });
    await refresh();
  }

  return (
    <main className="mx-auto min-h-screen w-full max-w-4xl px-6 py-16">
      <header className="flex items-center justify-between">
        <h1 className="text-3xl font-semibold">Documents</h1>
        <a href="/dashboard" className="text-sm text-[var(--muted)] hover:text-white">
          ← Dashboard
        </a>
      </header>

      <form onSubmit={onUpload} className="mt-8 flex items-center gap-3">
        <input
          ref={fileInput}
          type="file"
          accept=".txt,.md,.pdf"
          required
          className="text-sm text-[var(--muted)] file:mr-3 file:rounded file:border file:border-white/15 file:bg-white/5 file:px-3 file:py-2 file:text-white"
        />
        <button
          type="submit"
          disabled={uploading}
          className="rounded bg-[var(--accent)] px-4 py-2 text-sm font-medium text-black disabled:opacity-60"
        >
          {uploading ? "Uploading…" : "Upload"}
        </button>
      </form>
      {error ? <p className="mt-3 text-sm text-red-400">{error}</p> : null}

      <ul className="mt-8 divide-y divide-white/10">
        {documents.map((doc) => (
          <li key={doc.id} className="flex items-center justify-between py-4">
            <div>
              <p className="font-medium">{doc.title}</p>
              <p className="mt-0.5 text-sm text-[var(--muted)]">
                {doc.filename} · {formatBytes(doc.size_bytes)}
                {doc.status === "failed" && doc.error ? ` · ${doc.error}` : ""}
              </p>
            </div>
            <div className="flex items-center gap-4">
              <span className={`text-sm ${STATUS_STYLES[doc.status]}`}>
                {doc.status}
              </span>
              <button
                type="button"
                onClick={() => void onDelete(doc.id)}
                className="text-sm text-[var(--muted)] hover:text-red-400"
              >
                Delete
              </button>
            </div>
          </li>
        ))}
        {documents.length === 0 ? (
          <li className="py-8 text-sm text-[var(--muted)]">
            No documents yet. Upload a .txt, .md or .pdf to get started.
          </li>
        ) : null}
      </ul>
    </main>
  );
}
