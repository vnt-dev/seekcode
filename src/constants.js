// Shared constants for models, providers, and pagination used across the UI.

export const DEFAULT_MODELS = [
  { id: "deepseek-v4-pro", label: "DeepSeek V4 Pro" },
  { id: "deepseek-v4-flash", label: "DeepSeek V4 Flash" },
];

export const DEFAULT_PROVIDER_ID = "default";
export const DEFAULT_PROVIDER_NAME = "默认供应商";
export const DEFAULT_REASONING_EFFORT = "high";

// Number of conversation turns fetched per session message page.
export const SESSION_MESSAGE_PAGE_TURNS = 20;

export const REASONING_EFFORTS = [
  { id: "high", label: "High" },
  { id: "max", label: "Max" },
];
