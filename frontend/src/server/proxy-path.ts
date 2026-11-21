// Path policy for the /api/core BFF proxy. The proxy forwards with the
// internal service token, so the browser must only ever reach tenant-facing
// routes: an allowlist of first segments, and no segment that could make the
// upstream URL parser climb out of it (`..` / `.` / separators).

const ALLOWED_ROOTS = new Set(["api-keys", "ask", "documents", "usage"]);

const UNSAFE_SEGMENT = /[/\\\0]/;

/** Encoded upstream path ("documents/<id>"), or null when it must be refused. */
export function resolveProxyPath(segments: readonly string[]): string | null {
  if (segments.length === 0 || !ALLOWED_ROOTS.has(segments[0])) return null;
  for (const segment of segments) {
    if (
      segment === "" ||
      segment === "." ||
      segment === ".." ||
      UNSAFE_SEGMENT.test(segment)
    ) {
      return null;
    }
  }
  return segments.map(encodeURIComponent).join("/");
}
