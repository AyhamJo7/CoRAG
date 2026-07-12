import { type NextRequest, NextResponse } from "next/server";

import { auth } from "@/auth";

// Catch-all BFF reverse proxy: the browser talks only to /api/core/*; this
// handler authenticates the Auth.js session, binds the active workspace
// against the session's memberships (a forged x-tenant-id selects nothing),
// and forwards to the compose-internal FastAPI with the internal token.
// Bodies are streamed through untouched in both directions (SSE-safe).

export const dynamic = "force-dynamic";

// Service-only surfaces the browser must never reach.
const BLOCKED_PREFIXES = new Set(["internal", "admin", "healthz"]);

const HOP_BY_HOP = new Set([
  "connection",
  "keep-alive",
  "transfer-encoding",
  "upgrade",
  "host",
]);

async function handler(
  req: NextRequest,
  { params }: { params: Promise<{ path: string[] }> },
): Promise<Response> {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { path } = await params;
  if (!path?.length || BLOCKED_PREFIXES.has(path[0])) {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }

  const backendUrl = process.env.BACKEND_URL;
  const internalToken = process.env.INTERNAL_SERVICE_TOKEN;
  if (!backendUrl || !internalToken) {
    return NextResponse.json({ error: "Service unavailable" }, { status: 503 });
  }

  const tenants = session.user.tenants ?? [];
  const requested = req.headers.get("x-tenant-id");
  const tenant = requested
    ? tenants.find((t) => t.id === requested)
    : tenants[0];
  if (!tenant) {
    return NextResponse.json({ error: "No workspace bound" }, { status: 403 });
  }

  const target = `${backendUrl.replace(/\/$/, "")}/${path.join("/")}${req.nextUrl.search}`;
  const headers = new Headers();
  const contentType = req.headers.get("content-type");
  if (contentType) headers.set("content-type", contentType);
  const accept = req.headers.get("accept");
  if (accept) headers.set("accept", accept);
  headers.set("x-internal-token", internalToken);
  headers.set("x-user-id", session.user.id);
  headers.set("x-tenant-id", tenant.id);
  headers.set(
    "x-request-id",
    req.headers.get("x-request-id") ?? crypto.randomUUID(),
  );

  const hasBody = req.method !== "GET" && req.method !== "HEAD";
  let upstream: Response;
  try {
    upstream = await fetch(target, {
      method: req.method,
      headers,
      body: hasBody ? req.body : undefined,
      // @ts-expect-error Node fetch requires duplex for streamed request bodies
      duplex: hasBody ? "half" : undefined,
      cache: "no-store",
    });
  } catch {
    return NextResponse.json({ error: "Upstream unavailable" }, { status: 502 });
  }

  const responseHeaders = new Headers();
  upstream.headers.forEach((value, key) => {
    if (!HOP_BY_HOP.has(key)) responseHeaders.set(key, value);
  });

  return new Response(upstream.body, {
    status: upstream.status,
    headers: responseHeaders,
  });
}

export {
  handler as GET,
  handler as POST,
  handler as PUT,
  handler as PATCH,
  handler as DELETE,
};
