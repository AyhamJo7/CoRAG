export { auth as middleware } from "@/auth";

// Guard the app pages: unauthenticated requests are redirected to /sign-in.
// API routes are excluded — /api/auth handles its own flow and /api/core
// fails closed (no session → 401). `.+` (not `.*`) keeps "/" public, and the
// file-extension exclusion serves SEO assets without an auth redirect.
export const config = {
  matcher: [
    "/((?!api|sign-in|sign-up|pricing|_next/static|_next/image|.*\\.).+)",
  ],
};
