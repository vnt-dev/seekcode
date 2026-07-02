// Right-hand panel showing aggregated model-call statistics for the session.

import React from "react";
import { PanelLeft, PanelRight } from "lucide-react";

import { formatElapsedDuration } from "../lib/format.js";

export function WorkspacePanel({ stats, collapsed, onToggle }) {
  if (collapsed) {
    return (
      <aside className="workspace is-collapsed">
        <button
          className="workspace-toggle"
          type="button"
          onClick={onToggle}
          title="展开会话统计"
          aria-label="展开会话统计"
        >
          <PanelLeft size={18} />
        </button>
      </aside>
    );
  }

  const callCount = Number(stats?.call_count ?? 0);
  const inputTokens = Number(stats?.input_tokens ?? 0);
  const outputTokens = Number(stats?.output_tokens ?? 0);
  const totalTokens = inputTokens + outputTokens;
  const cacheHitTokens = Number(stats?.cache_hit_tokens ?? 0);
  const averageCallElapsedMs = Number(stats?.average_call_elapsed_ms ?? 0);
  const averageTurnElapsedMs = Number(stats?.average_turn_elapsed_ms ?? 0);
  const cacheHitRate = inputTokens > 0 ? (cacheHitTokens / inputTokens) * 100 : 0;
  const cacheHitPercent = Math.min(Math.max(cacheHitRate, 0), 100);

  return (
    <aside className="workspace">
      <header className="workspace-header">
        <div>
          <span>会话看板</span>
          <strong>当前对话</strong>
        </div>
        <button
          className="workspace-toggle"
          type="button"
          onClick={onToggle}
          title="收起会话统计"
          aria-label="收起会话统计"
        >
          <PanelRight size={18} />
        </button>
      </header>

      <section className="workspace-section">
        <div className="metric-panel">
          <div className="metric-stack">
            <div className="metric-tile is-primary">
              <span>调用次数</span>
              <strong>{callCount.toLocaleString()}</strong>
            </div>
            <div className="metric-tile is-primary">
              <span>总 token</span>
              <strong>{totalTokens.toLocaleString()}</strong>
            </div>
          </div>
          <div className="metric-pair">
            <div className="metric-tile">
              <span>平均调用</span>
              <strong>{formatElapsedDuration(averageCallElapsedMs)}</strong>
            </div>
            <div className="metric-tile">
              <span>平均对话</span>
              <strong>{formatElapsedDuration(averageTurnElapsedMs)}</strong>
            </div>
          </div>
        </div>
      </section>

      <section className="workspace-section stats-detail-section">
        <div className="section-title">Token 用量</div>
        <div className="stat-list">
          <div className="stat-row">
            <span>输入 token</span>
            <strong>{inputTokens.toLocaleString()}</strong>
          </div>
          <div className="stat-row">
            <span>输出 token</span>
            <strong>{outputTokens.toLocaleString()}</strong>
          </div>
          <div className="stat-row">
            <span>缓存命中</span>
            <strong>{cacheHitTokens.toLocaleString()}</strong>
          </div>
          <div className="cache-meter">
            <div className="cache-meter-header">
              <span>缓存命中率</span>
              <strong>{cacheHitRate.toFixed(1)}%</strong>
            </div>
            <div
              className="cache-meter-track"
              aria-label={`缓存命中率 ${cacheHitRate.toFixed(1)}%`}
            >
              <span style={{ width: `${cacheHitPercent}%` }} />
            </div>
          </div>
        </div>
      </section>
    </aside>
  );
}
