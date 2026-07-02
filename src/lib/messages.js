// Message record mapping plus assistant block / tool-call assembly helpers.

import { SESSION_MESSAGE_PAGE_TURNS } from "../constants.js";
import { formatElapsedLocalDateTimeRange } from "./format.js";
import { createId } from "./id.js";

// Groups flat message records into ordered turns of user + assistant messages.
export function mapMessageRecords(records) {
  const turns = new Map();

  for (const record of records ?? []) {
    const key = record.turn_sequence;
    if (!turns.has(key)) {
      turns.set(key, {
        users: [],
        assistant: null,
        firstCreatedAt: null,
        lastCreatedAt: null,
      });
    }
    const turn = turns.get(key);
    rememberTurnRecordTime(turn, record.created_at);

    if (record.role === "user") {
      turn.users.push({
        id: record.id,
        role: "user",
        content: record.content,
      });
      continue;
    }

    if (record.role === "assistant") {
      if (!turn.assistant) {
        turn.assistant = createAssistantMessage(key);
      }
      appendAssistantTextBlock(turn.assistant, "reasoning", record.reasoning_content);
      appendAssistantTextBlock(turn.assistant, "content", record.content);
      applyStoredToolCalls(turn.assistant, record.tool_calls ?? []);
      continue;
    }

    if (record.role === "tool") {
      if (!turn.assistant) {
        turn.assistant = createAssistantMessage(key);
      }
      if (record.tool_call_id) {
        const patch = {
          id: record.tool_call_id,
          status: "done",
        };
        try {
          patch.output = JSON.parse(record.content || "null");
        } catch {
          patch.summary = record.content;
        }
        upsertToolCallBlock(turn.assistant, patch);
      }
    }
  }

  return Array.from(turns.entries())
    .sort(([left], [right]) => Number(left) - Number(right))
    .flatMap(([turnKey, turn]) => {
      if (turn.assistant) {
        recordAssistantTurnFinished(
          turn.assistant,
          turnKey,
          formatElapsedLocalDateTimeRange(turn.firstCreatedAt, turn.lastCreatedAt),
        );
      }
      return [...turn.users, turn.assistant].filter(Boolean);
    });
}

function rememberTurnRecordTime(turn, createdAt) {
  if (!createdAt) return;
  if (!turn.firstCreatedAt) turn.firstCreatedAt = createdAt;
  turn.lastCreatedAt = createdAt;
}

// Creates an empty assistant message keyed by turn sequence.
function createAssistantMessage(turnKey) {
  return {
    id: `assistant-${turnKey}`,
    role: "assistant",
    content: "",
    reasoning: "",
    events: [],
    blocks: [],
  };
}

// Builds pagination state from the loaded records (earliest turn, has-more).
export function messagePageStateFromRecords(records, fallbackEarliestTurn = null) {
  const turns = messageRecordTurnSequences(records);
  return {
    earliestTurn: turns[0] ?? fallbackEarliestTurn,
    hasMore: turns.length >= SESSION_MESSAGE_PAGE_TURNS,
    loading: false,
  };
}

// Returns the sorted, unique turn sequences present in the records.
export function messageRecordTurnSequences(records) {
  return Array.from(
    new Set(
      (records ?? [])
        .map((record) => Number(record.turn_sequence))
        .filter((turnSequence) => Number.isFinite(turnSequence)),
    ),
  ).sort((left, right) => left - right);
}

// Rehydrates tool-call blocks from stored assistant tool_calls.
export function applyStoredToolCalls(message, toolCalls) {
  for (const call of toolCalls) {
    const id = call.id;
    if (!id) continue;
    upsertToolCallBlock(message, {
      id,
      name: call.function?.name,
      status: "running",
      argumentsDelta: call.function?.arguments ?? "",
      display: call.display,
    });
  }
}

// Appends streamed reasoning/content text to the latest matching block.
export function appendAssistantTextBlock(message, type, text) {
  if (!text) return;

  const field = type === "reasoning" ? "reasoning" : "content";
  message[field] = `${message[field] ?? ""}${text}`;

  const blocks = ensureAssistantBlocks(message);
  let block = blocks[blocks.length - 1];
  if (!block || block.type !== type) {
    block = {
      id: createId(),
      type,
      text: "",
    };
    blocks.push(block);
  }
  block.text += text;
}

// Clears streamed output for a failed attempt and records the retry notice.
export function recordAssistantModelRetry(message, payload) {
  message.content = "";
  message.reasoning = "";
  message.events = [];

  const blocks = ensureAssistantBlocks(message).filter((block) => block.type === "retry");
  blocks.push({
    id: `retry-${payload.round_id}-${payload.retry_count}`,
    type: "retry",
    retryCount: payload.retry_count,
    maxRetries: payload.max_retries,
    error: String(payload.error ?? "Model request failed"),
  });
  message.blocks = blocks;
}

// Records model round completion metadata without merging it into assistant text.
export function recordAssistantRoundFinished(message, payload, elapsedLabel) {
  const roundId = payload?.round_id;
  if (!roundId || !elapsedLabel) return;

  const blocks = ensureAssistantBlocks(message);
  const id = `round-finished-${roundId}`;
  const usage = payload?.usage ?? null;
  const existing = blocks.find((block) => block.id === id);
  const nextBlock = {
    id,
    type: "round_finished",
    roundId,
    elapsedLabel,
    usage,
  };

  if (existing) Object.assign(existing, nextBlock);
  else blocks.push(nextBlock);
}

// Records historical turn completion metadata reconstructed from persisted rows.
export function recordAssistantTurnFinished(message, turnKey, elapsedLabel) {
  if (!elapsedLabel) return;

  const blocks = ensureAssistantBlocks(message);
  const id = `turn-finished-${turnKey}`;
  const existing = blocks.find((block) => block.id === id);
  const nextBlock = {
    id,
    type: "round_finished",
    label: "本轮",
    elapsedLabel,
    usage: null,
  };

  if (existing) Object.assign(existing, nextBlock);
  else blocks.push(nextBlock);
}

// Creates or updates a tool-call entry and its associated block.
export function upsertToolCallBlock(message, patch) {
  if (!patch.id) return;

  if (!Array.isArray(message.events)) message.events = [];
  let item = message.events.find((tool) => tool.id === patch.id);
  if (!item) {
    item = {
      id: patch.id,
      name: "",
      status: "running",
      arguments: "",
      display: null,
    };
    message.events.push(item);
  }

  if (patch.name) item.name = patch.name;
  if (patch.status) item.status = patch.status;
  if (Object.prototype.hasOwnProperty.call(patch, "arguments")) {
    item.arguments = patch.arguments;
  }
  if (patch.argumentsDelta) {
    item.arguments = `${item.arguments ?? ""}${patch.argumentsDelta}`;
  }
  if (Object.prototype.hasOwnProperty.call(patch, "display")) item.display = patch.display;
  if (Object.prototype.hasOwnProperty.call(patch, "summary")) item.summary = patch.summary;
  if (Object.prototype.hasOwnProperty.call(patch, "output")) item.output = patch.output;
  if (Object.prototype.hasOwnProperty.call(patch, "error")) item.error = patch.error;

  const blocks = ensureAssistantBlocks(message);
  let block = blocks.find(
    (candidate) => candidate.type === "tool" && candidate.tool?.id === patch.id,
  );
  if (!block) {
    block = {
      id: `tool-${patch.id}`,
      type: "tool",
      tool: item,
    };
    blocks.push(block);
  } else {
    block.tool = item;
  }
}

// Returns the renderable assistant blocks (non-empty text or resolved tools).
export function getAssistantBlocks(message) {
  return ensureAssistantBlocks(message).filter((block) => {
    if (block.type === "tool") return Boolean(block.tool);
    if (block.type === "retry") return Boolean(block.error);
    if (block.type === "round_finished") return Boolean(block.elapsedLabel);
    return Boolean(block.text);
  });
}

// Extracts the expandable display info from a tool call, if present.
export function toolDisplayInfo(tool) {
  const display = tool?.display;
  if (!display || typeof display !== "object") return null;

  const title = String(display.title ?? "").trim() || "Details";
  const preview = String(display.preview ?? "").trim();
  const detail = String(display.detail ?? "");
  if (!preview || !detail) return null;

  return { title, preview, detail };
}

// Lazily builds the ordered block list from legacy message fields.
export function ensureAssistantBlocks(message) {
  if (Array.isArray(message.blocks)) return message.blocks;

  message.blocks = [];
  if (message.reasoning) {
    message.blocks.push({
      id: createId(),
      type: "reasoning",
      text: message.reasoning,
    });
  }
  if (message.content) {
    message.blocks.push({
      id: createId(),
      type: "content",
      text: message.content,
    });
  }
  for (const tool of message.events ?? []) {
    message.blocks.push({
      id: `tool-${tool.id}`,
      type: "tool",
      tool,
    });
  }
  return message.blocks;
}

// Reports whether an assistant message has no reasoning, content, or tools.
export function isAssistantMessageEmpty(message) {
  if (message.content || message.reasoning || message.events?.length) return false;
  return getAssistantBlocks(message).length === 0;
}
