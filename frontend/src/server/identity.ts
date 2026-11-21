// Credential → identity resolution for Auth.js. Calls the backend's
// internal-token-guarded /internal/login, which checks app_user and returns
// the user's tenant memberships — the authorization source the BFF proxy
// binds the active workspace from.

import { z } from "zod";

import type { TenantMembership } from "@/lib/types";

export interface AuthedUser {
  id: string;
  email: string;
  name: string;
  tenants: TenantMembership[];
}

const loginResponseSchema = z.object({
  user_id: z.string(),
  email: z.string(),
  name: z.string(),
  tenants: z.array(
    z.object({
      id: z.string(),
      name: z.string(),
      role: z.string(),
      plan: z.string(),
    }),
  ),
});

export async function verifyCredentials(
  email: string,
  password: string,
): Promise<AuthedUser | null> {
  const base = process.env.BACKEND_URL;
  const token = process.env.INTERNAL_SERVICE_TOKEN;
  if (!base || !token) {
    throw new Error("BACKEND_URL / INTERNAL_SERVICE_TOKEN are not configured");
  }
  const res = await fetch(`${base.replace(/\/$/, "")}/internal/login`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-internal-token": token,
    },
    body: JSON.stringify({ email, password }),
    cache: "no-store",
  });
  if (res.status === 401) return null;
  if (!res.ok) throw new Error(`backend /internal/login ${res.status}`);
  const parsed = loginResponseSchema.parse(await res.json());
  return {
    id: parsed.user_id,
    email: parsed.email,
    name: parsed.name,
    tenants: parsed.tenants,
  };
}
