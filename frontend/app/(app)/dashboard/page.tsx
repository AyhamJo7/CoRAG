import { redirect } from "next/navigation";

import { auth, signOut } from "@/auth";

export default async function DashboardPage() {
  const session = await auth();
  if (!session?.user) redirect("/sign-in");
  const tenant = session.user.tenants[0];

  return (
    <main className="mx-auto min-h-screen w-full max-w-4xl px-6 py-16">
      <header className="flex items-center justify-between">
        <div>
          <p className="text-sm uppercase tracking-[0.3em] text-[var(--muted)]">
            {tenant?.name ?? "Workspace"}
          </p>
          <h1 className="mt-1 text-3xl font-semibold">
            Welcome, {session.user.name ?? session.user.email}
          </h1>
        </div>
        <form
          action={async () => {
            "use server";
            await signOut({ redirectTo: "/" });
          }}
        >
          <button
            type="submit"
            className="rounded border border-white/15 px-3 py-2 text-sm text-[var(--muted)] hover:text-white"
          >
            Sign out
          </button>
        </form>
      </header>

      <section className="mt-12 grid gap-4 md:grid-cols-2">
        <div className="rounded-lg border border-white/10 p-6">
          <h2 className="font-medium">Documents</h2>
          <p className="mt-2 text-sm text-[var(--muted)]">
            Upload documents and CoRAG will index them for multi-hop questions.
            Coming in the next release.
          </p>
        </div>
        <div className="rounded-lg border border-white/10 p-6">
          <h2 className="font-medium">Ask</h2>
          <p className="mt-2 text-sm text-[var(--muted)]">
            Ask complex questions over your documents with cited answers.
            Coming in the next release.
          </p>
        </div>
      </section>

      <p className="mt-10 text-sm text-[var(--muted)]">
        Plan: <span className="text-white">{tenant?.plan ?? "trial"}</span>
      </p>
    </main>
  );
}
