import { NextResponse } from "next/server";
import { z } from "zod";

// Server-only signup: proxies to the internal provisioning endpoint (which
// runs on the admin DB connection) using the internal token. Never exposes
// that token or URL to the client. On success the browser calls signIn()
// with the same credentials to establish the session.

const bodySchema = z.object({
  workspace: z.string().trim().max(120).optional().default(""),
  name: z.string().trim().min(1, "Name is required").max(120),
  email: z.string().trim().email("Invalid email").max(320),
  password: z
    .string()
    .min(10, "Password must be at least 10 characters")
    .max(256),
});

export async function POST(request: Request): Promise<NextResponse> {
  const backendUrl = process.env.BACKEND_URL;
  const internalToken = process.env.INTERNAL_SERVICE_TOKEN;
  if (!backendUrl || !internalToken) {
    return NextResponse.json(
      { error: "Registration is currently unavailable." },
      { status: 503 },
    );
  }

  let parsed: z.infer<typeof bodySchema>;
  try {
    parsed = bodySchema.parse(await request.json());
  } catch (error) {
    const message =
      error instanceof z.ZodError ? error.issues[0]?.message : "Invalid input";
    return NextResponse.json(
      { error: message ?? "Invalid input" },
      { status: 422 },
    );
  }

  let res: Response;
  try {
    res = await fetch(`${backendUrl.replace(/\/$/, "")}/internal/provision`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-internal-token": internalToken,
      },
      body: JSON.stringify(parsed),
      cache: "no-store",
    });
  } catch {
    return NextResponse.json(
      { error: "Registration failed. Please try again later." },
      { status: 502 },
    );
  }

  if (res.status === 409) {
    return NextResponse.json(
      { error: "This email is already registered." },
      { status: 409 },
    );
  }
  if (!res.ok) {
    return NextResponse.json({ error: "Registration failed." }, { status: 502 });
  }

  return NextResponse.json({ ok: true }, { status: 201 });
}
