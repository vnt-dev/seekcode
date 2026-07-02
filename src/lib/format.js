// Formatting and small mapping helpers for tokens, numbers, dates, and tones.

// Parses a context window expression like "1M", "500k", or "64000" into tokens.
export function parseContextWindow(raw) {
  const text = String(raw ?? "").trim().toLowerCase();
  if (!text) return 1_000_000;
  let multiplier = 1;
  let numberPart = text;
  if (text.endsWith("k")) {
    multiplier = 1_000;
    numberPart = text.slice(0, -1);
  } else if (text.endsWith("m")) {
    multiplier = 1_000_000;
    numberPart = text.slice(0, -1);
  }
  const value = Number.parseFloat(numberPart.trim());
  if (!Number.isFinite(value) || value <= 0) return 1_000_000;
  return Math.round(value * multiplier);
}

// Formats a token count with a k/M unit, one decimal, rounded up.
// A positive value below 0.1k is shown as 0.1k.
export function formatContextTokens(tokens) {
  const value = Math.max(0, Number(tokens) || 0);
  if (value >= 1_000_000) {
    const millions = Math.ceil((value / 1_000_000) * 10) / 10;
    return `${millions.toFixed(1)}M`;
  }
  const thousands = Math.ceil((value / 1_000) * 10) / 10;
  return `${thousands.toFixed(1)}k`;
}

// Formats a number with a compact k/M unit for inline labels.
export function formatCompactNumber(value) {
  const number = Math.max(0, Number(value) || 0);
  if (number >= 1_000_000) return `${(number / 1_000_000).toFixed(1)}M`;
  if (number >= 1_000) return `${(number / 1_000).toFixed(1)}k`;
  return String(Math.round(number));
}

// Formats elapsed milliseconds as a compact model-call duration.
export function formatElapsedDuration(value) {
  const milliseconds = Number(value);
  if (!Number.isFinite(milliseconds) || milliseconds < 0) return "";
  let totalSeconds = Math.round(milliseconds / 1_000);
  if (milliseconds > 0 && totalSeconds === 0) totalSeconds = 1;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  if (minutes > 0) return `${minutes}m ${seconds}s`;
  return `${seconds}s`;
}

// Formats the elapsed time between two local "yyyy-MM-dd HH:mm:ss" timestamps.
export function formatElapsedLocalDateTimeRange(start, end) {
  const startMs = parseLocalDateTime(start);
  const endMs = parseLocalDateTime(end);
  if (!Number.isFinite(startMs) || !Number.isFinite(endMs) || endMs < startMs) return "";
  return formatElapsedDuration(endMs - startMs);
}

function parseLocalDateTime(value) {
  const match = String(value ?? "").match(
    /^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})$/,
  );
  if (!match) return Number.NaN;
  const [, year, month, day, hour, minute, second] = match.map(Number);
  return new Date(year, month - 1, day, hour, minute, second).getTime();
}

// Extracts the "HH:MM" portion from a stored "date time" string.
export function shortDateTime(value) {
  if (!value) return "";
  const [, time = value] = String(value).split(" ");
  return time.slice(0, 5) || value;
}

// Maps a session state to a timeline tone.
export function stateTone(state) {
  if (state === "completed") return "success";
  if (state === "failed" || state === "canceled") return "danger";
  if (state === "thinking" || state === "running_tool") return "active";
  return "neutral";
}
