export default function LandingPage() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center px-6">
      <p className="mb-4 text-sm uppercase tracking-[0.3em] text-[var(--muted)]">
        CoRAG Cloud
      </p>
      <h1 className="max-w-3xl text-center text-4xl font-semibold leading-tight md:text-6xl">
        Multi-hop answers over your documents,{" "}
        <span className="text-[var(--accent)]">with citations</span>.
      </h1>
      <p className="mt-6 max-w-xl text-center text-lg text-[var(--muted)]">
        Upload your documents. Ask complex questions. CoRAG decomposes,
        retrieves iteratively, and answers only when the evidence is
        sufficient.
      </p>
      <p className="mt-10 text-sm text-[var(--muted)]">Coming soon.</p>
    </main>
  );
}
