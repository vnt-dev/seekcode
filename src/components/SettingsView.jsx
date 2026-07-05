// Application settings page: default provider config plus additional providers,
// each with an editable model collection.

import React from "react";
import { CloudDownload, Loader2, Plus, Save, Settings2, Trash2 } from "lucide-react";

import { DEFAULT_PROVIDER_ID } from "../constants.js";
import { normalizeAdditionalProviders, updateProviderAt } from "../lib/models.js";

export function SettingsView({
  settings,
  setSettings,
  settingsStatus,
  fetchingModels,
  onFetchModels,
  onSubmit,
  onBack,
}) {
  const canFetchDefault =
    Boolean(settings.base_url?.trim()) &&
    Boolean(settings.api_key?.trim()) &&
    !fetchingModels;
  const providers = normalizeAdditionalProviders(settings.providers);

  return (
    <main className="settings-page">
      <header className="settings-header">
        <div>
          <div className="crumb">
            <Settings2 size={15} />
            应用设置
          </div>
          <h1>设置</h1>
        </div>
        <button className="secondary-button" onClick={onBack}>
          返回
        </button>
      </header>

      <div className="settings-layout">
        <nav className="settings-nav">
          <button className="settings-nav-item is-active" type="button">
            <Settings2 size={16} />
            <span>配置</span>
          </button>
        </nav>

        <form className="settings-form" onSubmit={onSubmit}>
          <section className="settings-panel">
            <div className="settings-panel-header"></div>

            <div className="settings-subsection-title">默认模型供应商</div>

            <label className="field">
              <span>base_url</span>
              <input
                value={settings.base_url}
                onChange={(event) =>
                  setSettings((current) => ({ ...current, base_url: event.target.value }))
                }
                placeholder="https://api.deepseek.com"
              />
            </label>

            <label className="field">
              <span>api_key</span>
              <input
                value={settings.api_key}
                onChange={(event) =>
                  setSettings((current) => ({ ...current, api_key: event.target.value }))
                }
                placeholder="sk-..."
                type="password"
              />
            </label>

            <label className="field">
              <span>title_model</span>
              <input
                value={settings.title_model}
                onChange={(event) =>
                  setSettings((current) => ({ ...current, title_model: event.target.value }))
                }
                placeholder="deepseek-v4-flash"
              />
            </label>

            <label className="field">
              <span>模型上下文大小</span>
              <input
                value={settings.context_window}
                onChange={(event) =>
                  setSettings((current) => ({ ...current, context_window: event.target.value }))
                }
                placeholder="1M"
              />
              <small className="field-hint">支持 k / M 单位（忽略大小写），例如 1M、500k、64000</small>
            </label>

            <ModelCollectionEditor
              title="默认供应商模型集合"
              models={settings.models}
              fetching={fetchingModels === DEFAULT_PROVIDER_ID}
              canFetch={canFetchDefault}
              onFetch={() => onFetchModels(DEFAULT_PROVIDER_ID)}
              onChange={(models) => setSettings((current) => ({ ...current, models }))}
            />

            <div className="provider-actions">
              <button
                className="secondary-button"
                type="button"
                onClick={() =>
                  setSettings((current) => ({
                    ...current,
                    providers: [
                      ...normalizeAdditionalProviders(current.providers),
                      {
                        id: "",
                        name: "",
                        base_url: "",
                        api_key: "",
                        models: [{ id: "", label: "" }],
                      },
                    ],
                  }))
                }
              >
                <Plus size={15} />
                <span>添加供应商</span>
              </button>
            </div>

            {providers.length > 0 ? (
              <div className="providers-field">
                <div className="models-field-header">
                  <span>其他模型供应商</span>
                </div>
              <div className="provider-config-list">
                {providers.map((provider, providerIndex) => {
                  const canFetchProvider =
                    Boolean(provider.id?.trim()) &&
                    Boolean(provider.base_url?.trim()) &&
                    Boolean(provider.api_key?.trim()) &&
                    !fetchingModels;
                  return (
                    <div className="provider-config" key={providerIndex}>
                      <div className="provider-config-header">
                        <strong>{provider.name || provider.id || "新供应商"}</strong>
                        <button
                          className="model-remove-button"
                          type="button"
                          title="移除供应商"
                          onClick={() =>
                            setSettings((current) => ({
                              ...current,
                              providers: normalizeAdditionalProviders(current.providers).filter(
                                (_, index) => index !== providerIndex,
                              ),
                            }))
                          }
                        >
                          <Trash2 size={15} />
                        </button>
                      </div>
                      <div className="provider-grid">
                        <label className="field">
                          <span>供应商 ID</span>
                          <input
                            value={provider.id}
                            onChange={(event) =>
                              setSettings((current) =>
                                updateProviderAt(current, providerIndex, { id: event.target.value }),
                              )
                            }
                            placeholder="openai"
                          />
                        </label>
                        <label className="field">
                          <span>供应商名称</span>
                          <input
                            value={provider.name}
                            onChange={(event) =>
                              setSettings((current) =>
                                updateProviderAt(current, providerIndex, { name: event.target.value }),
                              )
                            }
                            placeholder="OpenAI"
                          />
                        </label>
                        <label className="field">
                          <span>base_url</span>
                          <input
                            value={provider.base_url}
                            onChange={(event) =>
                              setSettings((current) =>
                                updateProviderAt(current, providerIndex, {
                                  base_url: event.target.value,
                                }),
                              )
                            }
                            placeholder="https://api.openai.com/v1"
                          />
                        </label>
                        <label className="field">
                          <span>api_key</span>
                          <input
                            value={provider.api_key}
                            onChange={(event) =>
                              setSettings((current) =>
                                updateProviderAt(current, providerIndex, {
                                  api_key: event.target.value,
                                }),
                              )
                            }
                            placeholder="sk-..."
                            type="password"
                          />
                        </label>
                      </div>
                      <ModelCollectionEditor
                        title="模型集合"
                        models={provider.models}
                        fetching={fetchingModels === provider.id}
                        canFetch={canFetchProvider}
                        onFetch={() => onFetchModels(provider.id)}
                        onChange={(models) =>
                          setSettings((current) =>
                            updateProviderAt(current, providerIndex, { models }),
                          )
                        }
                      />
                    </div>
                  );
                })}
              </div>
              </div>
            ) : null}

            <div className="settings-actions">
              {settingsStatus !== "idle" ? (
                <span className="settings-status">{settingsStatus}</span>
              ) : null}
              <button className="save-button" type="submit">
                <Save size={16} />
                <span>保存</span>
              </button>
            </div>
          </section>
        </form>
      </div>
    </main>
  );
}

// Editable list of model id/label pairs with a "fetch models" action.
function ModelCollectionEditor({ title, models, fetching, canFetch, onFetch, onChange }) {
  const editableModels =
    Array.isArray(models) && models.length > 0 ? models : [{ id: "", label: "" }];

  return (
    <div className="models-field">
      <div className="models-field-header">
        <span>{title}</span>
        <div className="models-field-actions">
          <button
            className={`secondary-button fetch-models-button ${fetching ? "is-fetching" : ""}`}
            type="button"
            disabled={!canFetch}
            onClick={onFetch}
          >
            {fetching ? <Loader2 size={15} /> : <CloudDownload size={15} />}
            <span>获取模型</span>
          </button>
          <button
            className="secondary-button"
            type="button"
            onClick={() => onChange([...editableModels, { id: "", label: "" }])}
          >
            <Plus size={15} />
            <span>添加模型</span>
          </button>
        </div>
      </div>
      <div className="model-config-list">
        {editableModels.map((item, index) => (
          <div className="model-config-row" key={index}>
            <input
              value={item.id}
              onChange={(event) => {
                const next = [...editableModels];
                next[index] = { ...next[index], id: event.target.value };
                onChange(next);
              }}
              placeholder="deepseek-v4-pro"
            />
            <input
              value={item.label}
              onChange={(event) => {
                const next = [...editableModels];
                next[index] = { ...next[index], label: event.target.value };
                onChange(next);
              }}
              placeholder="DeepSeek V4 Pro"
            />
            <button
              className="model-remove-button"
              type="button"
              title="移除模型"
              onClick={() => {
                const next = editableModels.filter((_, itemIndex) => itemIndex !== index);
                onChange(next.length > 0 ? next : [{ id: "", label: "" }]);
              }}
            >
              <Trash2 size={15} />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
