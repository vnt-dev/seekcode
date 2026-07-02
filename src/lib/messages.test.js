import assert from "node:assert/strict";
import test from "node:test";

import { getAssistantBlocks, mapMessageRecords, recordAssistantRoundFinished } from "./messages.js";

test("recordAssistantRoundFinished appends visible round completion metadata", () => {
  const message = {
    role: "assistant",
    content: "",
    reasoning: "",
    events: [],
    blocks: [],
  };

  recordAssistantRoundFinished(
    message,
    {
      round_id: 2,
      usage: { total_tokens: 4_200 },
    },
    "1m 5s",
  );

  assert.deepEqual(getAssistantBlocks(message), [
    {
      id: "round-finished-2",
      type: "round_finished",
      roundId: 2,
      elapsedLabel: "1m 5s",
      usage: { total_tokens: 4_200 },
    },
  ]);
});

test("recordAssistantRoundFinished updates the same round instead of duplicating it", () => {
  const message = {
    role: "assistant",
    content: "",
    reasoning: "",
    events: [],
    blocks: [],
  };

  recordAssistantRoundFinished(message, { round_id: 1, usage: null }, "8s");
  recordAssistantRoundFinished(message, { round_id: 1, usage: null }, "9s");

  assert.equal(getAssistantBlocks(message).length, 1);
  assert.equal(getAssistantBlocks(message)[0].elapsedLabel, "9s");
});

test("mapMessageRecords adds historical turn duration from first to last record time", () => {
  const messages = mapMessageRecords([
    {
      id: 1,
      turn_sequence: 7,
      role: "user",
      content: "Question",
      created_at: "2026-07-02 10:00:00",
    },
    {
      id: 2,
      turn_sequence: 7,
      role: "assistant",
      content: "Answer",
      reasoning_content: null,
      tool_calls: [],
      created_at: "2026-07-02 10:01:12",
    },
  ]);

  assert.equal(messages.length, 2);
  assert.deepEqual(getAssistantBlocks(messages[1]).at(-1), {
    id: "turn-finished-7",
    type: "round_finished",
    label: "本轮",
    elapsedLabel: "1m 12s",
    usage: null,
  });
});
