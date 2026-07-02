import assert from "node:assert/strict";
import test from "node:test";

import { formatElapsedDuration, formatElapsedLocalDateTimeRange } from "./format.js";

test("formatElapsedDuration renders seconds and minutes", () => {
  assert.equal(formatElapsedDuration(0), "0s");
  assert.equal(formatElapsedDuration(250), "1s");
  assert.equal(formatElapsedDuration(59_400), "59s");
  assert.equal(formatElapsedDuration(65_100), "1m 5s");
});

test("formatElapsedDuration rejects invalid values", () => {
  assert.equal(formatElapsedDuration(-1), "");
  assert.equal(formatElapsedDuration(Number.NaN), "");
});

test("formatElapsedLocalDateTimeRange formats stored local timestamps", () => {
  assert.equal(
    formatElapsedLocalDateTimeRange("2026-07-02 10:00:00", "2026-07-02 10:03:05"),
    "3m 5s",
  );
  assert.equal(
    formatElapsedLocalDateTimeRange("2026-07-02 10:00:00", "2026-07-02 09:59:59"),
    "",
  );
  assert.equal(formatElapsedLocalDateTimeRange("bad", "2026-07-02 10:00:00"), "");
});
