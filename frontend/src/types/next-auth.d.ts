import type { TenantMembership } from "@/lib/types";

declare module "next-auth" {
  interface User {
    tenants: TenantMembership[];
  }

  interface Session {
    user: {
      id: string;
      email?: string | null;
      name?: string | null;
      tenants: TenantMembership[];
    };
  }
}
