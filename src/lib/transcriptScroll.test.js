import assert from "node:assert/strict";
import test from "node:test";

import {
  consumeBooleanRef,
  initialTranscriptLocation,
  isUserScrollBlockingFollowOutput,
  isNearTranscriptBottom,
  shouldScrollTranscriptToBottom,
  shouldTreatWheelAsUserScrollIntent,
  TRANSCRIPT_FOLLOW_SCROLL_DELAYS,
} from "./transcriptScroll.js";

test("initialTranscriptLocation starts non-empty transcripts at the bottom", () => {
  assert.deepEqual(initialTranscriptLocation(3), {
    index: "LAST",
    align: "end",
    behavior: "auto",
  });
  assert.equal(initialTranscriptLocation(0), undefined);
});

test("isNearTranscriptBottom accepts small remaining scroll distance", () => {
  assert.equal(
    isNearTranscriptBottom({ scrollHeight: 1_000, scrollTop: 380, clientHeight: 500 }),
    true,
  );
  assert.equal(
    isNearTranscriptBottom({ scrollHeight: 1_000, scrollTop: 300, clientHeight: 500 }),
    false,
  );
});

test("shouldScrollTranscriptToBottom follows only forced or bottom-pinned output", () => {
  assert.equal(
    shouldScrollTranscriptToBottom({
      forceToBottom: true,
      pinnedToBottom: false,
      suppress: false,
    }),
    true,
  );
  assert.equal(
    shouldScrollTranscriptToBottom({
      forceToBottom: false,
      pinnedToBottom: true,
      suppress: false,
    }),
    true,
  );
  assert.equal(
    shouldScrollTranscriptToBottom({
      forceToBottom: false,
      pinnedToBottom: false,
      suppress: false,
    }),
    false,
  );
});

test("shouldScrollTranscriptToBottom suppresses prepended history changes", () => {
  assert.equal(
    shouldScrollTranscriptToBottom({
      forceToBottom: true,
      pinnedToBottom: true,
      suppress: true,
    }),
    false,
  );
});

test("shouldScrollTranscriptToBottom does not scroll after the user leaves bottom", () => {
  assert.equal(
    shouldScrollTranscriptToBottom({
      forceToBottom: true,
      pinnedToBottom: false,
      suppress: false,
      userScrolling: true,
      nearBottom: false,
    }),
    false,
  );
});

test("shouldScrollTranscriptToBottom keeps following when user input is still bottom-pinned", () => {
  assert.equal(
    shouldScrollTranscriptToBottom({
      forceToBottom: false,
      pinnedToBottom: true,
      suppress: false,
      userScrolling: true,
      nearBottom: false,
    }),
    true,
  );
  assert.equal(
    shouldScrollTranscriptToBottom({
      forceToBottom: false,
      pinnedToBottom: false,
      suppress: false,
      userScrolling: true,
      nearBottom: true,
    }),
    true,
  );
});

test("isUserScrollBlockingFollowOutput only blocks after leaving the bottom", () => {
  assert.equal(
    isUserScrollBlockingFollowOutput({
      userScrolling: true,
      pinnedToBottom: false,
      nearBottom: false,
    }),
    true,
  );
  assert.equal(
    isUserScrollBlockingFollowOutput({
      userScrolling: true,
      pinnedToBottom: true,
      nearBottom: false,
    }),
    false,
  );
  assert.equal(
    isUserScrollBlockingFollowOutput({
      userScrolling: true,
      pinnedToBottom: false,
      nearBottom: true,
    }),
    false,
  );
});

test("shouldTreatWheelAsUserScrollIntent ignores bottom-pinned downward wheel input", () => {
  assert.equal(shouldTreatWheelAsUserScrollIntent({ deltaY: 24, nearBottom: true }), false);
  assert.equal(shouldTreatWheelAsUserScrollIntent({ deltaY: 24, nearBottom: false }), true);
  assert.equal(shouldTreatWheelAsUserScrollIntent({ deltaY: -24, nearBottom: true }), true);
});

test("TRANSCRIPT_FOLLOW_SCROLL_DELAYS covers late virtual list measurements", () => {
  assert.equal(TRANSCRIPT_FOLLOW_SCROLL_DELAYS.at(-1) >= 1_000, true);
});

test("consumeBooleanRef clears one-shot scroll flags", () => {
  const ref = { current: true };

  assert.equal(consumeBooleanRef(ref), true);
  assert.equal(ref.current, false);
  assert.equal(consumeBooleanRef(ref), false);
});
