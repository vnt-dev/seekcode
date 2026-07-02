// Renders a single transcript message: user text, assistant blocks, tool calls,
// or a context-compaction divider.

import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Check,
  CheckCircle2,
  Clock,
  Copy,
  Hammer,
  Loader2,
  Sparkles,
  XCircle,
} from "lucide-react";

import { formatCompactNumber } from "../lib/format.js";
import { getAssistantBlocks, toolDisplayInfo } from "../lib/messages.js";

export function MessageBubble({ message }) {
  if (message.role === "compaction") {
    const isDone = message.status === "done";
    const sizeLabel =
      isDone && Number.isFinite(Number(message.summaryChars))
        ? `摘要 ${formatCompactNumber(message.summaryChars)} 字`
        : "";
    return (
      <div className={`compaction-divider ${isDone ? "is-done" : "is-running"}`}>
        <span className="compaction-divider-line" />
        <span className="compaction-divider-label">
          <span className="compaction-divider-text">
            {isDone ? "已压缩上下文" : "正在压缩上下文"}
          </span>
          {sizeLabel ? <span className="compaction-divider-size">{sizeLabel}</span> : null}
        </span>
        <span className="compaction-divider-line" />
      </div>
    );
  }

  const isUser = message.role === "user";
  const blocks = isUser ? [] : getAssistantBlocks(message);
  return (
    <article className={`message ${isUser ? "is-user" : "is-assistant"}`}>
      <div className="message-body">
        {isUser ? <div className="message-content">{message.content}</div> : null}
        {!isUser
          ? blocks.map((block) => {
              if (block.type === "reasoning") {
                return (
                  <details className="reasoning message-block" open key={block.id}>
                    <summary>
                      <Sparkles size={14} />
                      Reasoning
                    </summary>
                    <div className="reasoning-content markdown-content">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{block.text}</ReactMarkdown>
                    </div>
                  </details>
                );
              }

              if (block.type === "tool") {
                return <ToolCallBlock key={block.id} tool={block.tool} />;
              }

              if (block.type === "retry") {
                return <ModelRetryBlock key={block.id} retry={block} />;
              }

              if (block.type === "round_finished") {
                return <RoundFinishedBlock key={block.id} block={block} />;
              }

              return (
                <div className="message-content markdown-content message-block" key={block.id}>
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{block.text}</ReactMarkdown>
                </div>
              );
            })
          : null}
      </div>
    </article>
  );
}

// Renders completion metadata for one model request round.
function RoundFinishedBlock({ block }) {
  const roundLabel = block.label || (block.roundId ? `Round ${block.roundId}` : "本轮");
  const tokenLabel =
    block.usage && Number.isFinite(Number(block.usage.total_tokens))
      ? `${formatCompactNumber(block.usage.total_tokens)} tokens`
      : "";

  return (
    <div className="round-finished message-block">
      <Clock size={13} />
      <span>{roundLabel}</span>
      <span>处理用时 {block.elapsedLabel}</span>
      {tokenLabel ? <span>{tokenLabel}</span> : null}
    </div>
  );
}

// Renders a compact notice for a failed model request attempt.
function ModelRetryBlock({ retry }) {
  return (
    <div className="model-retry message-block">
      <div className="model-retry-title">
        <XCircle size={14} />
        <span>
          模型调用失败，正在重试 {retry.retryCount}/{retry.maxRetries}
        </span>
      </div>
      <code>{retry.error}</code>
    </div>
  );
}

// Renders a tool call chip with an optional expandable command panel.
function ToolCallBlock({ tool }) {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);
  const display = toolDisplayInfo(tool);
  const expandable = Boolean(display);

  function toggleExpanded() {
    if (!expandable) return;
    setExpanded((value) => !value);
  }

  function handleKeyDown(event) {
    if (!expandable) return;
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    toggleExpanded();
  }

  async function copyCommand(event) {
    event.stopPropagation();
    if (!display?.detail) return;
    await navigator.clipboard?.writeText(display.detail);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  }

  return (
    <div
      className={`tool-strip message-block ${expandable ? "is-expandable" : ""} ${expanded ? "is-expanded" : ""}`}
    >
      <div
        className="tool-strip-head"
        onClick={toggleExpanded}
        onKeyDown={handleKeyDown}
        role={expandable ? "button" : undefined}
        tabIndex={expandable ? 0 : undefined}
        aria-expanded={expandable ? expanded : undefined}
      >
        <div className={`tool-chip is-${tool.status}`}>
          <Hammer size={14} />
          <span>{tool.name || "tool"}</span>
          {tool.status === "running" ? <Loader2 size={13} /> : null}
          {tool.status === "done" ? <CheckCircle2 size={13} /> : null}
          {tool.status === "failed" ? <XCircle size={13} /> : null}
        </div>
        {display?.preview ? (
          <code className="tool-command-line" title={display.preview}>
            {display.preview}
          </code>
        ) : null}
      </div>

      {expanded && display ? (
        <div className="tool-command-panel">
          <div className="tool-command-panel-bar">
            <span>{display.title}</span>
            <button
              className="tool-command-copy"
              type="button"
              title="复制内容"
              aria-label="复制内容"
              onClick={copyCommand}
            >
              {copied ? <Check size={14} /> : <Copy size={14} />}
            </button>
          </div>
          <pre>
            <code>{display.detail}</code>
          </pre>
        </div>
      ) : null}
    </div>
  );
}
