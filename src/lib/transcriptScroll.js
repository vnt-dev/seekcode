// Scroll policy helpers for the virtual transcript.

export const TRANSCRIPT_BOTTOM_THRESHOLD = 120;
export const TRANSCRIPT_FOLLOW_SCROLL_DELAYS = [40, 120, 250, 500, 900, 1400];

// Returns the initial Virtuoso position used when a session transcript mounts.
export function initialTranscriptLocation(messageCount) {
  if (messageCount <= 0) return undefined;
  return {
    index: "LAST",
    align: "end",
    behavior: "auto",
  };
}

// Reports whether the scroll container is already at, or close to, the bottom.
export function isNearTranscriptBottom(scroller, threshold = TRANSCRIPT_BOTTOM_THRESHOLD) {
  if (!scroller) return true;
  const remaining = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight;
  return remaining <= threshold;
}

// Reports whether a wheel gesture should interrupt automatic follow-output.
export function shouldTreatWheelAsUserScrollIntent({ deltaY, nearBottom }) {
  const wheelDelta = Number(deltaY);
  if (!Number.isFinite(wheelDelta)) return !nearBottom;
  if (wheelDelta < 0) return true;
  return !nearBottom;
}

// User gestures only block follow-output after the transcript has left bottom.
export function isUserScrollBlockingFollowOutput({
  userScrolling = false,
  pinnedToBottom = false,
  nearBottom = false,
}) {
  return Boolean(userScrolling && !pinnedToBottom && !nearBottom);
}

// Decides whether the next rendered change should move to the true scroll end.
export function shouldScrollTranscriptToBottom({
  forceToBottom,
  pinnedToBottom,
  suppress,
  userScrolling = false,
  nearBottom = false,
}) {
  if (suppress) return false;
  if (isUserScrollBlockingFollowOutput({ userScrolling, pinnedToBottom, nearBottom })) {
    return false;
  }
  return forceToBottom || pinnedToBottom || nearBottom;
}

// Consumes a boolean ref-style flag exactly once.
export function consumeBooleanRef(ref) {
  const value = Boolean(ref.current);
  ref.current = false;
  return value;
}
