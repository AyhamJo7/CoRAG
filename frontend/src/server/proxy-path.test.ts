import assert from "node:assert/strict";
import { test } from "node:test";

import { resolveProxyPath } from "./proxy-path.ts";

test("allows tenant-facing routes", () => {
  assert.equal(resolveProxyPath(["usage"]), "usage");
  assert.equal(resolveProxyPath(["ask"]), "ask");
  assert.equal(
    resolveProxyPath(["documents", "6f1e0863-95a9-4a6c-9d61-3f1a1a1a1a1a"]),
    "documents/6f1e0863-95a9-4a6c-9d61-3f1a1a1a1a1a",
  );
});

test("refuses service-only and unknown roots", () => {
  for (const root of ["internal", "admin", "healthz", "Internal", "v1", ""]) {
    assert.equal(resolveProxyPath([root, "billing", "sync"]), null, root);
  }
  assert.equal(resolveProxyPath([]), null);
});

test("refuses traversal out of an allowed root", () => {
  assert.equal(resolveProxyPath(["documents", "..", "internal", "provision"]), null);
  assert.equal(resolveProxyPath(["usage", "."]), null);
  assert.equal(resolveProxyPath(["documents", "a/../../internal"]), null);
  assert.equal(resolveProxyPath(["documents", "..\\internal"]), null);
  assert.equal(resolveProxyPath(["documents", ""]), null);
  assert.equal(resolveProxyPath(["documents", "a\0b"]), null);
});

test("percent-encodes odd characters instead of passing them through", () => {
  assert.equal(resolveProxyPath(["documents", "a b?c"]), "documents/a%20b%3Fc");
});
