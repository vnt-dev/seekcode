// Session-scoped transcript cache helpers.

// Returns the cached transcript for a session without exposing the mutable map.
export function cachedSessionMessages(cache, sessionId) {
  if (!sessionId) return [];
  return cache.get(sessionId) ?? [];
}

// Stores one session transcript in a new map so callers can keep immutable
// update boundaries while a ref owns the latest cache.
export function replaceSessionMessages(cache, sessionId, messages) {
  if (!sessionId) return cache;
  const next = new Map(cache);
  next.set(sessionId, messages);
  return next;
}

// Applies a transcript mutation to only one session.
export function updateCachedSessionMessages(cache, sessionId, updater) {
  if (!sessionId) return cache;
  return replaceSessionMessages(cache, sessionId, updater(cachedSessionMessages(cache, sessionId)));
}

// Drops deleted sessions from the transcript cache.
export function deleteCachedSessionMessages(cache, sessionIds) {
  const ids = new Set((sessionIds ?? []).filter(Boolean));
  if (ids.size === 0) return cache;

  const next = new Map(cache);
  for (const sessionId of ids) next.delete(sessionId);
  return next;
}
