// Model/provider option helpers and settings normalization.

import {
  DEFAULT_MODELS,
  DEFAULT_PROVIDER_ID,
  DEFAULT_PROVIDER_NAME,
  DEFAULT_REASONING_EFFORT,
} from "../constants.js";

// Maps a persisted settings record into the shape used by the UI.
export function mapLoadedSettings(settings) {
  return {
    base_url: settings?.base_url ?? "https://api.deepseek.com",
    api_key: settings?.api_key ?? "",
    title_model: settings?.title_model ?? "deepseek-v4-flash",
    context_window: settings?.context_window ?? "1M",
    models: normalizeModelOptions(settings?.models),
    providers: normalizeAdditionalProviders(settings?.providers),
    minimize_to_tray: settings?.minimize_to_tray ?? true,
    close_behavior_configured: settings?.close_behavior_configured ?? false,
  };
}

// Deduplicates and trims a model list, falling back when nothing is valid.
export function normalizeModelOptions(models, fallback = DEFAULT_MODELS) {
  const source = Array.isArray(models) ? models : fallback;
  const seen = new Set();
  const normalized = [];

  for (const item of source) {
    const id = String(item?.id ?? "").trim();
    if (!id || seen.has(id)) continue;
    seen.add(id);
    const label = String(item?.label ?? "").trim();
    normalized.push({ id, label: label || id });
  }

  return normalized.length > 0 ? normalized : fallback;
}

// Normalizes the additional (non-default) provider list for editing.
export function normalizeAdditionalProviders(providers) {
  if (!Array.isArray(providers)) return [];
  return providers.map((provider) => ({
    id: String(provider?.id ?? "").trim(),
    name: String(provider?.name ?? "").trim(),
    base_url: String(provider?.base_url ?? "").trim(),
    api_key: String(provider?.api_key ?? "").trim(),
    models:
      Array.isArray(provider?.models) && provider.models.length > 0
        ? provider.models.map((model) => ({
            id: String(model?.id ?? ""),
            label: String(model?.label ?? ""),
          }))
        : [{ id: "", label: "" }],
  }));
}

// Builds the flat list of selectable model options across all providers.
export function buildModelOptions(settings) {
  const providers = [
    {
      id: DEFAULT_PROVIDER_ID,
      name: DEFAULT_PROVIDER_NAME,
      models: normalizeModelOptions(settings?.models),
      isDefault: true,
    },
    ...normalizeAdditionalProviders(settings?.providers)
      .filter((provider) => provider.id)
      .map((provider) => ({
        id: provider.id,
        name: provider.name || provider.id,
        models: normalizeModelOptions(provider.models, []),
        isDefault: false,
      })),
  ];

  const seen = new Set();
  const options = [];
  for (const provider of providers) {
    for (const item of provider.models) {
      const key = modelKey(provider.id, item.id);
      if (seen.has(key)) continue;
      seen.add(key);
      options.push({
        key,
        providerId: provider.id,
        providerName: provider.name,
        modelId: item.id,
        modelLabel: item.label,
        label: provider.isDefault ? item.label : `${provider.name}-${item.label}`,
      });
    }
  }
  return options.length > 0
    ? options
    : [
        {
          key: modelKey(DEFAULT_PROVIDER_ID, DEFAULT_MODELS[0].id),
          providerId: DEFAULT_PROVIDER_ID,
          providerName: DEFAULT_PROVIDER_NAME,
          modelId: DEFAULT_MODELS[0].id,
          modelLabel: DEFAULT_MODELS[0].label,
          label: DEFAULT_MODELS[0].label,
        },
      ];
}

// Encodes a provider/model pair into a stable option key.
export function modelKey(providerId, modelId) {
  return `${encodeURIComponent(providerId)}::${encodeURIComponent(modelId)}`;
}

// Resolves an option by key, falling back to the first option.
export function resolveModelOptionByKey(key, modelOptions) {
  const normalizedKey = String(key ?? "");
  return modelOptions.find((item) => item.key === normalizedKey) ?? modelOptions[0] ?? null;
}

// Resolves the option key that matches a session's stored provider/model.
export function resolveModelForSession(session, modelOptions) {
  const options =
    Array.isArray(modelOptions) && modelOptions.length > 0 ? modelOptions : buildModelOptions({});
  const sessionProvider = String(session?.modelProvider ?? "").trim();
  const sessionModel = String(session?.model ?? "").trim();
  const match = options.find(
    (item) => item.providerId === sessionProvider && item.modelId === sessionModel,
  );
  return match?.key ?? options[0].key;
}

// Checks whether a session already uses the given model configuration.
export function sessionUsesModelConfig(session, option, thinkingEnabled, reasoningEffort) {
  return (
    String(session?.modelProvider ?? "").trim() === option.providerId &&
    String(session?.model ?? "").trim() === option.modelId &&
    Boolean(session?.thinkingEnabled) === Boolean(thinkingEnabled) &&
    String(session?.reasoningEffort || DEFAULT_REASONING_EFFORT) ===
      String(reasoningEffort || DEFAULT_REASONING_EFFORT)
  );
}

// Returns the base_url/api_key pair for a provider id from settings.
export function settingsProviderById(settings, providerId) {
  if (!providerId || providerId === DEFAULT_PROVIDER_ID) {
    return {
      id: DEFAULT_PROVIDER_ID,
      base_url: settings?.base_url ?? "",
      api_key: settings?.api_key ?? "",
    };
  }
  return (
    normalizeAdditionalProviders(settings?.providers).find(
      (provider) => provider.id === providerId,
    ) ?? {
      id: providerId,
      base_url: "",
      api_key: "",
    }
  );
}

// Returns settings with the model list replaced for the given provider.
export function setProviderModels(settings, providerId, models) {
  if (!providerId || providerId === DEFAULT_PROVIDER_ID) {
    return { ...settings, models };
  }

  return {
    ...settings,
    providers: normalizeAdditionalProviders(settings.providers).map((provider) =>
      provider.id === providerId ? { ...provider, models } : provider,
    ),
  };
}

// Returns settings with a patch applied to the provider at the given index.
export function updateProviderAt(settings, providerIndex, patch) {
  const providers = normalizeAdditionalProviders(settings.providers);
  providers[providerIndex] = { ...providers[providerIndex], ...patch };
  return { ...settings, providers };
}
