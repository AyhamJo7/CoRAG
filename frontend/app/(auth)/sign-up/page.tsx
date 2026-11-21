"use client";

import { useRouter } from "next/navigation";
import { signIn } from "next-auth/react";
import { type FormEvent, useState } from "react";

export default function SignUpPage() {
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setPending(true);
    const form = new FormData(event.currentTarget);
    const email = String(form.get("email") ?? "");
    const password = String(form.get("password") ?? "");

    const res = await fetch("/api/sign-up", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        workspace: String(form.get("workspace") ?? ""),
        name: String(form.get("name") ?? ""),
        email,
        password,
      }),
    });
    if (!res.ok) {
      const body = (await res.json().catch(() => null)) as {
        error?: string;
      } | null;
      setError(body?.error ?? "Registration failed.");
      setPending(false);
      return;
    }

    const result = await signIn("credentials", {
      email,
      password,
      redirect: false,
    });
    setPending(false);
    if (result?.error) {
      router.push("/sign-in");
      return;
    }
    router.push("/dashboard");
    router.refresh();
  }

  return (
    <main className="flex min-h-screen items-center justify-center px-6">
      <form onSubmit={onSubmit} className="w-full max-w-sm space-y-4">
        <h1 className="text-2xl font-semibold">Create your workspace</h1>
        <label className="block text-sm">
          <span className="text-[var(--muted)]">Your name</span>
          <input
            name="name"
            required
            maxLength={120}
            className="mt-1 w-full rounded border border-white/15 bg-white/5 px-3 py-2 outline-none focus:border-[var(--accent)]"
          />
        </label>
        <label className="block text-sm">
          <span className="text-[var(--muted)]">Workspace name (optional)</span>
          <input
            name="workspace"
            maxLength={120}
            className="mt-1 w-full rounded border border-white/15 bg-white/5 px-3 py-2 outline-none focus:border-[var(--accent)]"
          />
        </label>
        <label className="block text-sm">
          <span className="text-[var(--muted)]">Email</span>
          <input
            name="email"
            type="email"
            required
            autoComplete="email"
            className="mt-1 w-full rounded border border-white/15 bg-white/5 px-3 py-2 outline-none focus:border-[var(--accent)]"
          />
        </label>
        <label className="block text-sm">
          <span className="text-[var(--muted)]">Password (min. 10 characters)</span>
          <input
            name="password"
            type="password"
            required
            minLength={10}
            autoComplete="new-password"
            className="mt-1 w-full rounded border border-white/15 bg-white/5 px-3 py-2 outline-none focus:border-[var(--accent)]"
          />
        </label>
        {error ? <p className="text-sm text-red-400">{error}</p> : null}
        <button
          type="submit"
          disabled={pending}
          className="w-full rounded bg-[var(--accent)] px-3 py-2 font-medium text-black disabled:opacity-60"
        >
          {pending ? "Creating…" : "Create account"}
        </button>
        <p className="text-sm text-[var(--muted)]">
          Already registered?{" "}
          <a href="/sign-in" className="text-[var(--accent)]">
            Sign in
          </a>
        </p>
      </form>
    </main>
  );
}
