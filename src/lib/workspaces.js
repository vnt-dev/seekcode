// Workspace/session record mapping and filesystem path helpers.

import { DEFAULT_REASONING_EFFORT } from "../constants.js";
import { shortDateTime } from "./format.js";

// Maps a backend workspace bundle into the UI workspace shape.
export function mapWorkspaceBundle(bundle) {
  const workspace = bundle.workspace;
  return {
    id: workspace.id,
    name: workspace.name,
    path: workspace.absolute_path,
    sessions: (bundle.sessions ?? []).map(mapSessionRecord),
  };
}

// Maps a backend session record into the UI session shape.
export function mapSessionRecord(session) {
  return {
    id: session.id,
    title: session.name || "New chat",
    updated: shortDateTime(session.updated_at),
    model: session.model,
    modelProvider: session.model_provider,
    thinkingEnabled: session.thinking_enabled ?? true,
    reasoningEffort: session.reasoning_effort || DEFAULT_REASONING_EFFORT,
  };
}

// Finds a session by id across all workspaces.
export function findSessionById(workspaces, sessionId) {
  if (!sessionId) return null;
  for (const workspace of workspaces ?? []) {
    const session = workspace.sessions.find((item) => item.id === sessionId);
    if (session) return session;
  }
  return null;
}

// Derives a directory path from a selected file when the native path is known.
export function getSelectedDirectoryPath(file, relativePath, rootName) {
  if (!file?.path) return rootName;

  const relativeParts = relativePath.split(/[\\/]/).filter(Boolean);
  let directoryPath = file.path;
  for (let index = 1; index < relativeParts.length; index += 1) {
    directoryPath = directoryPath.replace(/[\\/][^\\/]*$/, "");
  }
  return directoryPath || rootName;
}

// Normalizes a workspace path for comparison (slashes, trailing, case).
export function normalizeWorkspacePath(path) {
  return String(path ?? "")
    .replace(/\\/g, "/")
    .replace(/\/+$/, "")
    .toLowerCase();
}

// Derives a display name from the final segment of a workspace path.
export function getWorkspaceName(path) {
  const normalized = String(path ?? "").replace(/\\/g, "/").replace(/\/+$/, "");
  return normalized.split("/").filter(Boolean).at(-1) || "Untitled Workspace";
}

