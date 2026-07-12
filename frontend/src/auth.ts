import NextAuth from "next-auth";
import Credentials from "next-auth/providers/credentials";

import type { TenantMembership } from "@/lib/types";
import { verifyCredentials } from "@/server/identity";

// Auth.js v5, JWT sessions (no DB session tables). The session carries the
// user's tenant memberships; the BFF proxy authorizes the active workspace
// against them — a forged x-tenant-id header selects nothing.
export const { handlers, auth, signIn, signOut } = NextAuth({
  session: { strategy: "jwt" },
  pages: { signIn: "/sign-in" },
  providers: [
    Credentials({
      credentials: { email: {}, password: {} },
      authorize: (credentials) => {
        const email =
          typeof credentials?.email === "string" ? credentials.email : "";
        const password =
          typeof credentials?.password === "string" ? credentials.password : "";
        if (!email || !password) return null;
        return verifyCredentials(email, password);
      },
    }),
  ],
  callbacks: {
    // Consumed by the middleware: unauthenticated requests to matched routes
    // are redirected to the sign-in page.
    authorized: ({ auth }) => Boolean(auth?.user),
    jwt({ token, user }) {
      if (user) {
        token.uid = user.id ?? "";
        token.tenants = user.tenants;
      }
      return token;
    },
    session({ session, token }) {
      if (typeof token.uid === "string") session.user.id = token.uid;
      session.user.tenants = Array.isArray(token.tenants)
        ? (token.tenants as TenantMembership[])
        : [];
      return session;
    },
  },
});
