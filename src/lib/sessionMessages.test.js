import assert from "node:assert/strict";
import test from "node:test";

import {
  cachedSessionMessages,
  deleteCachedSessionMessages,
  replaceSessionMessages,
  updateCachedSessionMessages,
} from "./sessionMessages.js";

test("updateCachedSessionMessages mutates only the targeted session", () => {
  const cache = new Map([
    ["session-a", [{ role: "user", content: "A" }]],
    ["session-b", [{ role: "user", content: "B" }]],
  ]);

  const next = updateCachedSessionMessages(cache, "session-a", (messages) => [
    ...messages,
    { role: "assistant", content: "A output" },
  ]);

  assert.deepEqual(cachedSessionMessages(next, "session-a"), [
    { role: "user", content: "A" },
    { role: "assistant", content: "A output" },
  ]);
  assert.deepEqual(cachedSessionMessages(next, "session-b"), [
    { role: "user", content: "B" },
  ]);
  assert.deepEqual(cachedSessionMessages(cache, "session-a"), [
    { role: "user", content: "A" },
  ]);
});

test("replaceSessionMessages does not create cache entries without a session", () => {
  const cache = new Map([["session-a", []]]);

  assert.equal(replaceSessionMessages(cache, null, [{ role: "user" }]), cache);
});

test("deleteCachedSessionMessages removes only deleted sessions", () => {
  const cache = new Map([
    ["session-a", [{ role: "user", content: "A" }]],
    ["session-b", [{ role: "user", content: "B" }]],
  ]);

  const next = deleteCachedSessionMessages(cache, ["session-a"]);

  assert.equal(cachedSessionMessages(next, "session-a").length, 0);
  assert.deepEqual(cachedSessionMessages(next, "session-b"), [
    { role: "user", content: "B" },
  ]);
});
