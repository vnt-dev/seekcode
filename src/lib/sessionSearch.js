// Builds the flattened session search result list used by the sidebar dialog.

function normalizeSearchText(value) {
  return String(value ?? "").trim().toLowerCase();
}

export function searchSessionsByTitle(workspaces, query) {
  const normalizedQuery = normalizeSearchText(query);
  const results = [];

  for (const workspace of workspaces ?? []) {
    for (const session of workspace.sessions ?? []) {
      const title = session.title || "New chat";
      const normalizedTitle = normalizeSearchText(title);
      if (normalizedQuery && !normalizedTitle.includes(normalizedQuery)) continue;

      results.push({
        id: session.id,
        title,
        workspaceId: workspace.id,
        workspaceName: workspace.name,
      });
    }
  }

  return results;
}
