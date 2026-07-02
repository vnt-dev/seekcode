// Helpers for tracking which workspace groups hide their session lists.

export function isWorkspaceCollapsed(collapsedIds, workspaceId) {
  return Boolean(workspaceId && collapsedIds?.has(workspaceId));
}

export function toggleWorkspaceCollapsedIds(collapsedIds, workspaceId) {
  const next = new Set(collapsedIds ?? []);
  if (!workspaceId) return next;

  if (next.has(workspaceId)) next.delete(workspaceId);
  else next.add(workspaceId);
  return next;
}

export function expandWorkspaceInCollapsedIds(collapsedIds, workspaceId) {
  const next = new Set(collapsedIds ?? []);
  if (workspaceId) next.delete(workspaceId);
  return next;
}
