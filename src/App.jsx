// Root application component: owns workspace/session state, streams agent
// events, and orchestrates the chat, settings, and stats views.

import React, { useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import { Virtuoso } from "react-virtuoso";
import {
  Bot,
  Boxes,
  Brain,
  Check,
  ChevronDown,
  ChevronRight,
  Loader2,
  MessageSquare,
  MessageSquarePlus,
  Plus,
  Search,
  Send,
  Settings2,
  Square,
  Trash2,
  XCircle,
} from "lucide-react";

import {
  DEFAULT_MODELS,
  DEFAULT_PROVIDER_ID,
  DEFAULT_REASONING_EFFORT,
  REASONING_EFFORTS,
  SESSION_MESSAGE_PAGE_TURNS,
} from "./constants.js";
import { createId } from "./lib/id.js";
import {
  formatContextTokens,
  formatElapsedDuration,
  parseContextWindow,
  stateTone,
} from "./lib/format.js";
import {
  buildModelOptions,
  mapLoadedSettings,
  modelKey,
  normalizeModelOptions,
  resolveModelForSession,
  resolveModelOptionByKey,
  sessionUsesModelConfig,
  setProviderModels,
  settingsProviderById,
} from "./lib/models.js";
import {
  findSessionById,
  getSelectedDirectoryPath,
  getWorkspaceName,
  mapSessionRecord,
  mapWorkspaceBundle,
} from "./lib/workspaces.js";
import {
  expandWorkspaceInCollapsedIds,
  isWorkspaceCollapsed,
  toggleWorkspaceCollapsedIds,
} from "./lib/workspaceCollapse.js";
import { searchSessionsByTitle } from "./lib/sessionSearch.js";
import {
  appendAssistantTextBlock,
  isAssistantMessageEmpty,
  mapMessageRecords,
  messagePageStateFromRecords,
  recordAssistantModelRetry,
  recordAssistantRoundFinished,
  upsertToolCallBlock,
} from "./lib/messages.js";
import {
  cachedSessionMessages,
  deleteCachedSessionMessages,
  replaceSessionMessages,
  updateCachedSessionMessages,
} from "./lib/sessionMessages.js";
import {
  consumeBooleanRef,
  isNearTranscriptBottom,
  initialTranscriptLocation,
  shouldScrollTranscriptToBottom,
  shouldTreatWheelAsUserScrollIntent,
  TRANSCRIPT_FOLLOW_SCROLL_DELAYS,
} from "./lib/transcriptScroll.js";
import { MessageBubble } from "./components/MessageBubble.jsx";
import { SettingsView } from "./components/SettingsView.jsx";
import { WorkspacePanel } from "./components/WorkspacePanel.jsx";

// Base virtual index for the transcript. Virtuoso keeps scroll position stable
// while prepending older messages by tracking a decreasing firstItemIndex, so we
// start high enough that many pages can be prepended without reaching zero.
const TRANSCRIPT_START_INDEX = 1_000_000;

// Top/bottom breathing room for the virtual transcript. Kept as a stable
// component so Virtuoso does not remount the Header/Footer on every render.
function TranscriptSpacer() {
  return <div className="transcript-spacer" />;
}
const TRANSCRIPT_COMPONENTS = { Header: TranscriptSpacer, Footer: TranscriptSpacer };

export function App() {
  const [workspaces, setWorkspaces] = useState([]);
  const [activeWorkspaceId, setActiveWorkspaceId] = useState(null);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [draftSession, setDraftSession] = useState(null);
  const [draggedWorkspaceId, setDraggedWorkspaceId] = useState(null);
  const [workspaceDragPreview, setWorkspaceDragPreview] = useState(null);
  const [collapsedWorkspaceIds, setCollapsedWorkspaceIds] = useState(() => new Set());
  const [contextMenu, setContextMenu] = useState(null);
  const [sessionSearchOpen, setSessionSearchOpen] = useState(false);
  const [sessionSearchQuery, setSessionSearchQuery] = useState("");
  const [workspacePromptOpen, setWorkspacePromptOpen] = useState(false);
  const [view, setView] = useState("chat");
  const [model, setModel] = useState(modelKey(DEFAULT_PROVIDER_ID, DEFAULT_MODELS[0].id));
  const [modelMenuOpen, setModelMenuOpen] = useState(false);
  const [thinkingEnabled, setThinkingEnabled] = useState(true);
  const [reasoningEffort, setReasoningEffort] = useState(DEFAULT_REASONING_EFFORT);
  const [prompt, setPrompt] = useState("");
  const [runningSessionIds, setRunningSessionIds] = useState(() => new Set());
  const [cancelingSessionIds, setCancelingSessionIds] = useState(() => new Set());
  const [taskIdsBySession, setTaskIdsBySession] = useState(() => new Map());
  const [activeTaskId, setActiveTaskId] = useState(null);
  const [modelError, setModelError] = useState(null);
  const [settings, setSettings] = useState({
    base_url: "https://api.deepseek.com",
    api_key: "",
    title_model: "deepseek-v4-flash",
    context_window: "1M",
    models: DEFAULT_MODELS,
    providers: [],
  });
  const [settingsStatus, setSettingsStatus] = useState("idle");
  const [fetchingModels, setFetchingModels] = useState(null);
  const [messages, setMessages] = useState([
    {
      id: "welcome",
      role: "assistant",
      content:
        "SeekCode is ready. Start a coding task to see answer tokens, reasoning, and tool calls stream in real time.",
      reasoning: "",
      events: [],
      blocks: [
        {
          id: "welcome-content",
          type: "content",
          text:
            "SeekCode is ready. Start a coding task to see answer tokens, reasoning, and tool calls stream in real time.",
        },
      ],
    },
  ]);
  const [timeline, setTimeline] = useState([]);
  const [selectedEventId, setSelectedEventId] = useState(null);
  const [contextUsageBySession, setContextUsageBySession] = useState(() => new Map());
  const [statsBySession, setStatsBySession] = useState(() => new Map());
  const [panelCollapsed, setPanelCollapsed] = useState(true);
  // Virtual index of the first rendered message; decreased when older messages
  // are prepended so Virtuoso can preserve the scroll position.
  const [firstItemIndex, setFirstItemIndex] = useState(TRANSCRIPT_START_INDEX);
  const [transcriptResetKey, setTranscriptResetKey] = useState(0);
  const virtuosoRef = useRef(null);
  const transcriptScrollerRef = useRef(null);
  // Signals the next messages render to jump to the bottom after a fresh load.
  const pendingBottomScrollRef = useRef(false);
  // Suppresses follow-output once when older history is prepended at the top.
  const suppressNextFollowOutputRef = useRef(false);
  // Mirrors the user's scroll position before new output changes list height.
  const transcriptPinnedToBottomRef = useRef(true);
  const transcriptUserScrollingRef = useRef(false);
  const transcriptUserScrollTimerRef = useRef(null);
  const transcriptBottomScrollFrameRef = useRef(null);
  const transcriptBottomScrollTimersRef = useRef([]);
  const activeTaskIdRef = useRef(null);
  const activeSessionIdRef = useRef(null);
  const runningSessionIdsRef = useRef(new Set());
  const cancelingSessionIdsRef = useRef(new Set());
  const taskIdsBySessionRef = useRef(new Map());
  const messagesBySessionRef = useRef(new Map());
  const hiddenCompactionTaskIdsRef = useRef(new Set());
  const messagePageStateRef = useRef(new Map());
  const modelRoundStartedAtRef = useRef(new Map());
  const titleAnimationTimersRef = useRef(new Map());
  const directoryInputRef = useRef(null);
  const workspaceListRef = useRef(null);
  const workspaceDragRef = useRef(null);
  const modelMenuRef = useRef(null);
  const sessionSearchInputRef = useRef(null);
  const suppressWorkspaceClickRef = useRef(false);

  const activeWorkspace = workspaces.find((workspace) => workspace.id === activeWorkspaceId);
  const draggedWorkspace = workspaces.find((workspace) => workspace.id === draggedWorkspaceId);
  const draggedWorkspaceCollapsed = isWorkspaceCollapsed(
    collapsedWorkspaceIds,
    draggedWorkspace?.id,
  );
  const activeSessionRunning = activeSessionId ? runningSessionIds.has(activeSessionId) : false;
  const activeSessionCanceling = activeSessionId ? cancelingSessionIds.has(activeSessionId) : false;
  const activeSessionTaskId = activeSessionId ? taskIdsBySession.get(activeSessionId) : null;
  const modelOptions = buildModelOptions(settings);
  const selectedModelOption = resolveModelOptionByKey(model, modelOptions);
  const modelContextWindow = parseContextWindow(settings.context_window);
  const activeContextUsed = activeSessionId
    ? contextUsageBySession.get(activeSessionId) ?? 0
    : 0;
  const activeContextPercent =
    modelContextWindow > 0
      ? Math.ceil((activeContextUsed / modelContextWindow) * 100)
      : 0;
  const activeSessionStats = activeSessionId
    ? statsBySession.get(activeSessionId) ?? null
    : null;
  const sessionSearchResults = useMemo(
    () => searchSessionsByTitle(workspaces, sessionSearchQuery),
    [workspaces, sessionSearchQuery],
  );

  useEffect(() => {
    activeTaskIdRef.current = activeTaskId;
  }, [activeTaskId]);

  useEffect(() => {
    activeSessionIdRef.current = activeSessionId;
  }, [activeSessionId]);

  useEffect(() => {
    loadSettingsFromDisk({ showStatus: false });
    refreshWorkspaces();
  }, []);

  useEffect(() => {
    const activeSession = findSessionById(workspaces, activeSessionId);
    const nextModel = resolveModelForSession(activeSession, modelOptions);
    if (model !== nextModel) setModel(nextModel);
    if (activeSession) {
      setThinkingEnabled(activeSession.thinkingEnabled);
      setReasoningEffort(activeSession.reasoningEffort || DEFAULT_REASONING_EFFORT);
    }
  }, [activeSessionId, model, settings.models, settings.providers, workspaces]);

  useEffect(() => {
    if (!modelMenuOpen) return;

    function closeModelMenu(event) {
      if (modelMenuRef.current?.contains(event.target)) return;
      setModelMenuOpen(false);
    }

    function closeModelMenuOnEscape(event) {
      if (event.key === "Escape") setModelMenuOpen(false);
    }

    window.addEventListener("pointerdown", closeModelMenu);
    window.addEventListener("keydown", closeModelMenuOnEscape);
    window.addEventListener("scroll", closeModelMenu, true);
    return () => {
      window.removeEventListener("pointerdown", closeModelMenu);
      window.removeEventListener("keydown", closeModelMenuOnEscape);
      window.removeEventListener("scroll", closeModelMenu, true);
    };
  }, [modelMenuOpen]);

  useEffect(() => {
    let unlisten;
    listen("agent:event", (event) => applyAgentEvent(event.payload)).then((dispose) => {
      unlisten = dispose;
    });
    return () => {
      if (unlisten) unlisten();
    };
  }, []);

  useEffect(() => {
    let unlisten;
    listen("session:title_changed", (event) => animateSessionTitle(event.payload)).then(
      (dispose) => {
        unlisten = dispose;
      },
    );
    return () => {
      if (unlisten) unlisten();
    };
  }, []);

  useEffect(() => {
    return () => {
      for (const timers of titleAnimationTimersRef.current.values()) {
        timers.forEach((timer) => window.clearTimeout(timer));
      }
      titleAnimationTimersRef.current.clear();
      cancelScheduledTranscriptBottomScroll();
      clearTranscriptUserScrollTimer();
    };
  }, []);

  useEffect(() => {
    const scroller = transcriptScrollerRef.current;
    if (!scroller) return undefined;

    function rememberTranscriptPinState() {
      transcriptPinnedToBottomRef.current = isNearTranscriptBottom(scroller);
    }

    function handleUserScrollIntent() {
      markTranscriptUserScrolling();
      cancelScheduledTranscriptBottomScroll();
    }

    function handleWheelIntent(event) {
      const nearBottom = isNearTranscriptBottom(scroller);
      if (!shouldTreatWheelAsUserScrollIntent({ deltaY: event.deltaY, nearBottom })) {
        transcriptPinnedToBottomRef.current = true;
        return;
      }
      transcriptPinnedToBottomRef.current = false;
      handleUserScrollIntent();
    }

    function handleTranscriptKeyDown(event) {
      const scrollKeys = new Set([
        "ArrowDown",
        "ArrowUp",
        "End",
        "Home",
        "PageDown",
        "PageUp",
        " ",
      ]);
      if (scrollKeys.has(event.key)) handleUserScrollIntent();
    }

    rememberTranscriptPinState();
    scroller.addEventListener("scroll", rememberTranscriptPinState, { passive: true });
    scroller.addEventListener("wheel", handleWheelIntent, { passive: true });
    scroller.addEventListener("touchstart", handleUserScrollIntent, { passive: true });
    scroller.addEventListener("pointerdown", handleUserScrollIntent);
    scroller.addEventListener("keydown", handleTranscriptKeyDown);
    window.addEventListener("pointerup", finishTranscriptUserScrolling);
    window.addEventListener("pointercancel", finishTranscriptUserScrolling);
    return () => {
      scroller.removeEventListener("scroll", rememberTranscriptPinState);
      scroller.removeEventListener("wheel", handleWheelIntent);
      scroller.removeEventListener("touchstart", handleUserScrollIntent);
      scroller.removeEventListener("pointerdown", handleUserScrollIntent);
      scroller.removeEventListener("keydown", handleTranscriptKeyDown);
      window.removeEventListener("pointerup", finishTranscriptUserScrolling);
      window.removeEventListener("pointercancel", finishTranscriptUserScrolling);
    };
  }, [transcriptResetKey]);

  useEffect(() => {
    const forceToBottom = consumeBooleanRef(pendingBottomScrollRef);
    const suppress = consumeBooleanRef(suppressNextFollowOutputRef);
    if (
      !shouldScrollTranscriptToBottom({
        forceToBottom,
        pinnedToBottom: transcriptPinnedToBottomRef.current,
        suppress,
        userScrolling: transcriptUserScrollingRef.current,
        nearBottom: isNearTranscriptBottom(transcriptScrollerRef.current),
      })
    ) {
      return;
    }

    scheduleTranscriptBottomScroll({ forceToBottom });
  }, [messages]);

  useEffect(() => {
    if (!contextMenu) return;

    function closeContextMenu() {
      setContextMenu(null);
    }

    window.addEventListener("click", closeContextMenu);
    window.addEventListener("keydown", closeContextMenu);
    window.addEventListener("scroll", closeContextMenu, true);
    return () => {
      window.removeEventListener("click", closeContextMenu);
      window.removeEventListener("keydown", closeContextMenu);
      window.removeEventListener("scroll", closeContextMenu, true);
    };
  }, [contextMenu]);

  useEffect(() => {
    if (!sessionSearchOpen) return undefined;

    sessionSearchInputRef.current?.focus();

    function closeSessionSearchOnEscape(event) {
      if (event.key === "Escape") closeSessionSearch();
    }

    window.addEventListener("keydown", closeSessionSearchOnEscape);
    return () => window.removeEventListener("keydown", closeSessionSearchOnEscape);
  }, [sessionSearchOpen]);

  useEffect(() => {
    if (view !== "settings") return;

    loadSettingsFromDisk({ showStatus: true });
  }, [view]);

  function appendTimeline(type, title, detail, payload, tone = "neutral") {
    const item = {
      id: createId(),
      type,
      title,
      detail,
      payload,
      tone,
      time: new Date().toLocaleTimeString("en-US", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      }),
    };
    setTimeline((items) => [...items, item]);
    setSelectedEventId((current) => current ?? item.id);
  }

  function modelRoundKey(payload) {
    return `${payload?.session_id ?? ""}:${payload?.task_id ?? ""}:${payload?.round_id ?? ""}`;
  }

  function rememberModelRoundStart(payload) {
    if (!payload?.session_id || !payload?.task_id || !payload?.round_id) return;
    modelRoundStartedAtRef.current.set(modelRoundKey(payload), Date.now());
  }

  function finishModelRoundTiming(payload) {
    const key = modelRoundKey(payload);
    const startedAt = modelRoundStartedAtRef.current.get(key);
    modelRoundStartedAtRef.current.delete(key);
    if (!Number.isFinite(startedAt)) return "";
    return formatElapsedDuration(Date.now() - startedAt);
  }

  function clearModelRoundTimingsForTask(sessionId, taskId) {
    const prefix = `${sessionId ?? ""}:${taskId ?? ""}:`;
    for (const key of modelRoundStartedAtRef.current.keys()) {
      if (key.startsWith(prefix)) modelRoundStartedAtRef.current.delete(key);
    }
  }

  async function loadSettingsFromDisk({ showStatus } = { showStatus: false }) {
    if (showStatus) setSettingsStatus("loading");
    try {
      const loaded = await invoke("load_app_settings");
      setSettings(mapLoadedSettings(loaded));
      if (showStatus) setSettingsStatus("idle");
    } catch (error) {
      if (showStatus) setSettingsStatus(`Load failed: ${error}`);
      else setModelError(String(error));
    }
  }

  function setSessionRunning(sessionId, running) {
    if (!sessionId) return;

    const next = new Set(runningSessionIdsRef.current);
    if (running) next.add(sessionId);
    else next.delete(sessionId);
    runningSessionIdsRef.current = next;
    setRunningSessionIds(next);

  }

  function setSessionCanceling(sessionId, canceling) {
    if (!sessionId) return;

    const next = new Set(cancelingSessionIdsRef.current);
    if (canceling) next.add(sessionId);
    else next.delete(sessionId);
    cancelingSessionIdsRef.current = next;
    setCancelingSessionIds(next);
  }

  function setSessionTask(sessionId, taskId) {
    if (!sessionId) return;

    const next = new Map(taskIdsBySessionRef.current);
    if (taskId) next.set(sessionId, taskId);
    else next.delete(sessionId);
    taskIdsBySessionRef.current = next;
    setTaskIdsBySession(next);

    if (activeSessionIdRef.current === sessionId) {
      activeTaskIdRef.current = taskId ?? null;
      setActiveTaskId(taskId ?? null);
    }
  }

  // Records the latest input token count used by a session's context.
  function setSessionContextUsage(sessionId, tokens) {
    if (!sessionId) return;
    const value = Number(tokens);
    if (!Number.isFinite(value) || value < 0) return;
    setContextUsageBySession((current) => {
      if (current.get(sessionId) === value) return current;
      const next = new Map(current);
      next.set(sessionId, value);
      return next;
    });
  }

  // Stores aggregated model-call stats for a session's dashboard.
  function setSessionStats(sessionId, stats) {
    if (!sessionId || !stats) return;
    setStatsBySession((current) => {
      const next = new Map(current);
      next.set(sessionId, stats);
      return next;
    });
  }

  function resetMessagePageState(sessionId) {
    if (!sessionId) return;
    messagePageStateRef.current.delete(sessionId);
  }

  function showEmptyMessages(sessionId = null) {
    if (sessionId) resetMessagePageState(sessionId);
    else messagePageStateRef.current.clear();
    setFirstItemIndex(TRANSCRIPT_START_INDEX);
    pendingBottomScrollRef.current = false;
    suppressNextFollowOutputRef.current = false;
    transcriptPinnedToBottomRef.current = true;
    setTranscriptResetKey((current) => current + 1);
    if (sessionId) {
      messagesBySessionRef.current = replaceSessionMessages(
        messagesBySessionRef.current,
        sessionId,
        [],
      );
    }
    setMessages([]);
  }

  function bindTranscriptScroller(scroller) {
    transcriptScrollerRef.current = scroller;
  }

  function clearTranscriptUserScrollTimer() {
    if (transcriptUserScrollTimerRef.current != null) {
      window.clearTimeout(transcriptUserScrollTimerRef.current);
      transcriptUserScrollTimerRef.current = null;
    }
  }

  function markTranscriptUserScrolling() {
    transcriptUserScrollingRef.current = true;
    clearTranscriptUserScrollTimer();
    transcriptUserScrollTimerRef.current = window.setTimeout(() => {
      transcriptUserScrollingRef.current = false;
      transcriptUserScrollTimerRef.current = null;
    }, 250);
  }

  function finishTranscriptUserScrolling() {
    clearTranscriptUserScrollTimer();
    transcriptUserScrollTimerRef.current = window.setTimeout(() => {
      transcriptUserScrollingRef.current = false;
      transcriptUserScrollTimerRef.current = null;
      const scroller = transcriptScrollerRef.current;
      if (scroller) transcriptPinnedToBottomRef.current = isNearTranscriptBottom(scroller);
    }, 80);
  }

  function cancelScheduledTranscriptBottomScroll() {
    if (transcriptBottomScrollFrameRef.current != null) {
      window.cancelAnimationFrame(transcriptBottomScrollFrameRef.current);
      transcriptBottomScrollFrameRef.current = null;
    }
    transcriptBottomScrollTimersRef.current.forEach((timer) => window.clearTimeout(timer));
    transcriptBottomScrollTimersRef.current = [];
  }

  // Scrolls to the container's true maximum offset, not merely to the last
  // virtual item. Repeating the scroll handles late Virtuoso measurements and
  // growing streamed content.
  function scheduleTranscriptBottomScroll({ forceToBottom = false } = {}) {
    cancelScheduledTranscriptBottomScroll();

    const scrollToBottom = () => {
      const scroller = transcriptScrollerRef.current;
      if (!scroller) return;
      if (
        !shouldScrollTranscriptToBottom({
          forceToBottom,
          pinnedToBottom: transcriptPinnedToBottomRef.current,
          suppress: false,
          userScrolling: transcriptUserScrollingRef.current,
          nearBottom: isNearTranscriptBottom(scroller),
        })
      ) {
        cancelScheduledTranscriptBottomScroll();
        return;
      }
      scroller.scrollTo({ top: scroller.scrollHeight, behavior: "auto" });
      transcriptPinnedToBottomRef.current = true;
    };

    transcriptBottomScrollFrameRef.current = window.requestAnimationFrame(() => {
      scrollToBottom();
      transcriptBottomScrollFrameRef.current = window.requestAnimationFrame(() => {
        scrollToBottom();
        transcriptBottomScrollFrameRef.current = null;
      });
    });
    transcriptBottomScrollTimersRef.current = TRANSCRIPT_FOLLOW_SCROLL_DELAYS.map((delay) =>
      window.setTimeout(scrollToBottom, delay),
    );
  }

  // Loads aggregated model-call telemetry for the session dashboard.
  async function loadSessionStats(sessionId) {
    if (!sessionId) return;
    try {
      const stats = await invoke("session_model_call_stats", { sessionId });
      setSessionStats(sessionId, stats);
    } catch (error) {
      console.warn("failed to load session stats", error);
    }
  }

  function replaceMessagesForSession(sessionId, nextMessages) {
    if (!sessionId) return;
    messagesBySessionRef.current = replaceSessionMessages(
      messagesBySessionRef.current,
      sessionId,
      nextMessages,
    );
    if (activeSessionIdRef.current === sessionId) setMessages(nextMessages);
  }

  // Applies stream updates to the owning session, even when another session is
  // currently visible.
  function updateMessagesForSession(sessionId, updater) {
    if (!sessionId) return;
    messagesBySessionRef.current = updateCachedSessionMessages(
      messagesBySessionRef.current,
      sessionId,
      updater,
    );
    if (activeSessionIdRef.current === sessionId) {
      setMessages(cachedSessionMessages(messagesBySessionRef.current, sessionId));
    }
  }

  function showCachedMessagesForSession(sessionId) {
    setFirstItemIndex(TRANSCRIPT_START_INDEX);
    pendingBottomScrollRef.current = true;
    suppressNextFollowOutputRef.current = false;
    transcriptPinnedToBottomRef.current = true;
    setTranscriptResetKey((current) => current + 1);
    setMessages(cachedSessionMessages(messagesBySessionRef.current, sessionId));
  }

  function updateAssistantMessage(sessionId, taskId, mutator) {
    updateMessagesForSession(sessionId, (items) => {
      const next = [...items];
      const last = next[next.length - 1];
      const currentTaskId = taskId ?? taskIdsBySessionRef.current.get(sessionId) ?? null;
      if (!last || last.role !== "assistant" || last.taskId !== currentTaskId) {
        next.push({
          id: createId(),
          role: "assistant",
          taskId: currentTaskId,
          content: "",
          reasoning: "",
          events: [],
          blocks: [],
        });
      }
      mutator(next[next.length - 1]);
      return next;
    });
  }

  function applyAgentEvent(event) {
    const { type, payload } = event;
    if (payload?.session_id && payload?.task_id) {
      setSessionTask(payload.session_id, payload.task_id);
    } else if (payload?.task_id && !activeTaskIdRef.current) {
      activeTaskIdRef.current = payload.task_id;
      setActiveTaskId(payload.task_id);
    }

    switch (type) {
      case "task_started":
        setSessionRunning(payload.session_id, true);
        setSessionCanceling(payload.session_id, false);
        setSessionTask(payload.session_id, payload.task_id);
        appendTimeline("task", "Task started", `model ${payload.model}`, payload, "active");
        break;
      case "state_changed":
        appendTimeline("state", "State changed", payload.state, payload, stateTone(payload.state));
        break;
      case "model_request_started":
        rememberModelRoundStart(payload);
        appendTimeline(
          "model",
          `Model round ${payload.round_id}`,
          `${payload.message_count} messages / ${payload.tool_count} tools`,
          payload,
          "active",
        );
        break;
      case "model_request_retrying":
        appendTimeline(
          "model",
          `Model retry ${payload.retry_count}/${payload.max_retries}`,
          payload.error,
          payload,
          "danger",
        );
        updateAssistantMessage(payload.session_id, payload.task_id, (message) => {
          recordAssistantModelRetry(message, payload);
        });
        break;
      case "assistant_message_delta":
        updateAssistantMessage(payload.session_id, payload.task_id, (message) => {
          appendAssistantTextBlock(message, "reasoning", payload.reasoning_content);
          appendAssistantTextBlock(message, "content", payload.content);
        });
        break;
      case "tool_call_started":
        appendTimeline(
          "tool",
          `Call ${payload.name}`,
          JSON.stringify(payload.arguments ?? {}),
          payload,
          "active",
        );
        updateAssistantMessage(payload.session_id, payload.task_id, (message) => {
          upsertToolCallBlock(message, {
            id: payload.tool_call_id,
            name: payload.name,
            status: "running",
            arguments: payload.arguments,
            display: payload.display,
          });
        });
        break;
      case "tool_call_finished":
        appendTimeline(
          "tool",
          `${payload.name} ${payload.ok ? "finished" : "failed"}`,
          payload.summary ?? payload.error ?? "",
          payload,
          payload.ok ? "success" : "danger",
        );
        updateAssistantMessage(payload.session_id, payload.task_id, (message) => {
          upsertToolCallBlock(message, {
            id: payload.tool_call_id,
            name: payload.name,
            status: payload.ok ? "done" : "failed",
            summary: payload.summary,
            output: payload.output,
            error: payload.error,
          });
        });
        break;
      case "model_round_finished":
        if (payload.usage) {
          setSessionContextUsage(payload.session_id, payload.usage.prompt_tokens);
        }
        const elapsedLabel = finishModelRoundTiming(payload);
        updateAssistantMessage(payload.session_id, payload.task_id, (message) => {
          recordAssistantRoundFinished(message, payload, elapsedLabel);
        });
        loadSessionStats(payload.session_id);
        const usageLabel = payload.usage ? `${payload.usage.total_tokens} tokens` : "no usage";
        appendTimeline(
          "model",
          `Round ${payload.round_id} finished`,
          elapsedLabel ? `${elapsedLabel} / ${usageLabel}` : usageLabel,
          payload,
          "success",
        );
        break;
      case "finished":
        setSessionRunning(payload.session_id, false);
        setSessionCanceling(payload.session_id, false);
        setSessionTask(payload.session_id, null);
        clearModelRoundTimingsForTask(payload.session_id, payload.task_id);
        loadSessionStats(payload.session_id);
        appendTimeline("task", "Task finished", payload.task_id, payload, "success");
        break;
      case "failed":
        setSessionRunning(payload.session_id, false);
        setSessionCanceling(payload.session_id, false);
        setSessionTask(payload.session_id, null);
        clearModelRoundTimingsForTask(payload.session_id, payload.task_id);
        hideCompactionDivider(payload.task_id, payload.session_id);
        appendTimeline("task", "Task failed", payload.error, payload, "danger");
        if (activeSessionIdRef.current === payload.session_id) {
          setModelError(String(payload.error ?? "Model call failed"));
        }
        removeEmptyAssistantPlaceholder(payload.session_id);
        break;
      case "canceled":
        setSessionRunning(payload.session_id, false);
        setSessionCanceling(payload.session_id, false);
        setSessionTask(payload.session_id, null);
        clearModelRoundTimingsForTask(payload.session_id, payload.task_id);
        hideCompactionDivider(payload.task_id, payload.session_id);
        appendTimeline("task", "Task canceled", payload.task_id, payload, "danger");
        break;
      case "context_compaction_started":
        upsertCompactionDivider("running", payload);
        appendTimeline("compaction", "正在压缩上下文", "", payload, "active");
        break;
      case "context_compaction_canceled":
        hideCompactionDivider(payload.task_id, payload.session_id);
        appendTimeline("compaction", "上下文压缩已取消", "", payload, "neutral");
        break;
      case "context_compaction_finished":
        upsertCompactionDivider("done", payload);
        appendTimeline(
          "compaction",
          "已压缩上下文",
          `已压缩 ${payload.compacted_rounds} 轮，摘要 ${payload.summary_chars} 字`,
          payload,
          "success",
        );
        break;
      default:
        appendTimeline("event", type, "", payload);
    }
  }

  async function submitPrompt(event) {
    event.preventDefault();
    const text = prompt.trim();
    if (!text) return;

    let sessionId = activeSessionId;
    if (!sessionId) {
      if (!activeWorkspaceId) {
        showWorkspaceRequiredDialog();
        return;
      }
      try {
        sessionId = await createSessionForWorkspace(activeWorkspaceId);
      } catch (error) {
        appendTimeline("task", "Start failed", String(error), { error }, "danger");
        setModelError(String(error));
        return;
      }
    }
    if (runningSessionIdsRef.current.has(sessionId)) return;

    const selectedModel = resolveModelOptionByKey(model, modelOptions) ?? modelOptions[0];
    if (!selectedModel) {
      setModelError("没有可用模型");
      return;
    }
    const currentSession = findSessionById(workspaces, sessionId);
    if (
      !sessionUsesModelConfig(
        currentSession,
        selectedModel,
        thinkingEnabled,
        reasoningEffort,
      )
    ) {
      try {
        const saved = await invoke("update_session_model", {
          sessionId,
          modelProvider: selectedModel.providerId,
          model: selectedModel.modelId,
          thinkingEnabled,
          reasoningEffort,
        });
        const mappedSession = mapSessionRecord(saved);
        setSessionModel(mappedSession.id, mappedSession);
      } catch (error) {
        appendTimeline("task", "Start failed", String(error), { error }, "danger");
        setModelError(String(error));
        return;
      }
    }

    setPrompt("");
    setModelError(null);
    activeSessionIdRef.current = sessionId;
    updateMessagesForSession(sessionId, (items) => [
      ...items,
      { id: createId(), role: "user", content: text },
      {
        id: createId(),
        role: "assistant",
        taskId: null,
        content: "",
        reasoning: "",
        events: [],
        blocks: [],
      },
    ]);
    setTimeline([]);
    setSelectedEventId(null);
    setSessionRunning(sessionId, true);
    setSessionCanceling(sessionId, false);
    setSessionTask(sessionId, null);
    activeTaskIdRef.current = null;
    setActiveTaskId(null);

    try {
      const task = await invoke("start_agent_task", {
        request: {
          session_id: sessionId,
          prompt: text,
          model: selectedModel.modelId,
          thinking: thinkingEnabled,
          reasoning_effort: reasoningEffort,
        },
      });
      setSessionTask(sessionId, task.id);
      updateMessagesForSession(sessionId, (items) => {
        const next = [...items];
        const last = next[next.length - 1];
        if (last?.role === "assistant") last.taskId = task.id;
        return next;
      });
    } catch (error) {
      setSessionRunning(sessionId, false);
      setSessionCanceling(sessionId, false);
      setSessionTask(sessionId, null);
      appendTimeline("task", "Start failed", String(error), { error }, "danger");
      setModelError(String(error));
      removeEmptyAssistantPlaceholder(sessionId);
    }
  }

  async function stopActiveTask() {
    const sessionId = activeSessionIdRef.current;
    const taskId = sessionId ? taskIdsBySessionRef.current.get(sessionId) : activeTaskIdRef.current;
    if (!sessionId || !taskId || cancelingSessionIdsRef.current.has(sessionId)) return;

    setSessionCanceling(sessionId, true);
    try {
      await invoke("cancel_agent_task", { taskId });
      hideCompactionDivider(taskId, sessionId);
    } catch (error) {
      setSessionCanceling(sessionId, false);
      hiddenCompactionTaskIdsRef.current.delete(taskId);
      appendTimeline("task", "Cancel failed", String(error), { error }, "danger");
      setModelError(String(error));
    }
  }

  function removeEmptyAssistantPlaceholder(sessionId = activeSessionIdRef.current) {
    updateMessagesForSession(sessionId, (items) => {
      const next = [...items];
      const last = next[next.length - 1];
      if (last?.role === "assistant" && isAssistantMessageEmpty(last)) {
        next.pop();
      }
      return next;
    });
  }

  function upsertCompactionDivider(status, payload) {
    const taskId = payload?.task_id ?? activeTaskIdRef.current;
    const sessionId = payload?.session_id ?? activeSessionIdRef.current;
    if (!taskId || !sessionId) return;
    if (status === "done") {
      hiddenCompactionTaskIdsRef.current.delete(taskId);
    } else if (hiddenCompactionTaskIdsRef.current.has(taskId)) {
      return;
    }

    updateMessagesForSession(sessionId, (items) => {
      const next = [...items];
      const existingIndex = next.findIndex(
        (message) => message.role === "compaction" && message.taskId === taskId,
      );
      const divider = {
        id: existingIndex >= 0 ? next[existingIndex].id : `compaction-${taskId}`,
        role: "compaction",
        taskId,
        status,
        summaryChars: payload?.summary_chars,
      };

      if (existingIndex >= 0) {
        next[existingIndex] = { ...next[existingIndex], ...divider };
        return next;
      }

      const placeholderIndex = next.findIndex(
        (message, index) =>
          index === next.length - 1 &&
          message.role === "assistant" &&
          (!message.taskId || message.taskId === taskId) &&
          isAssistantMessageEmpty(message),
      );
      if (placeholderIndex >= 0) {
        next[placeholderIndex] = { ...next[placeholderIndex], taskId };
        next.splice(placeholderIndex, 0, divider);
      } else {
        next.push(divider);
      }
      return next;
    });
  }

  function hideCompactionDivider(taskId, knownSessionId = null) {
    if (!taskId) return;
    const sessionId =
      knownSessionId ??
      [...taskIdsBySessionRef.current.entries()].find(([, id]) => id === taskId)?.[0] ??
      activeSessionIdRef.current;
    updateMessagesForSession(sessionId, (items) => {
      const divider = items.find(
        (message) => message.role === "compaction" && message.taskId === taskId,
      );
      if (divider?.status === "done") return items;

      hiddenCompactionTaskIdsRef.current.add(taskId);
      return items.filter(
        (message) => message.role !== "compaction" || message.taskId !== taskId,
      );
    });
  }

  async function refreshWorkspaces(preferredWorkspaceId, preferredSessionId) {
    try {
      const loaded = await invoke("list_visible_workspaces");
      const nextWorkspaces = loaded.map(mapWorkspaceBundle);
      setWorkspaces(nextWorkspaces);

      const nextWorkspace =
        nextWorkspaces.find((workspace) => workspace.id === preferredWorkspaceId) ??
        nextWorkspaces.find((workspace) => workspace.id === activeWorkspaceId) ??
        nextWorkspaces[0] ??
        null;
      const nextSession =
        nextWorkspace?.sessions.find((session) => session.id === preferredSessionId) ??
        nextWorkspace?.sessions.find((session) => session.id === activeSessionId) ??
        nextWorkspace?.sessions[0] ??
        null;

      setActiveWorkspaceId(nextWorkspace?.id ?? null);
      setActiveSessionId(nextSession?.id ?? null);
      activeSessionIdRef.current = nextSession?.id ?? null;
      setModel(resolveModelForSession(nextSession, modelOptions));
      if (nextSession && runningSessionIdsRef.current.has(nextSession.id)) {
        showCachedMessagesForSession(nextSession.id);
      } else if (nextSession) await loadSessionMessages(nextSession.id);
      else showEmptyMessages();
    } catch (error) {
      setModelError(String(error));
    }
  }

  async function createBlankConversation(workspaceId) {
    try {
      const sessionId = await createSessionForWorkspace(workspaceId);
      setActiveWorkspaceId(workspaceId);
      setActiveSessionId(sessionId);
      expandWorkspaceGroup(workspaceId);
      setDraftSession(null);
      setView("chat");
      setPrompt("");
      activeSessionIdRef.current = sessionId;
      setActiveTaskId(null);
      activeTaskIdRef.current = null;
      setModelError(null);
      setTimeline([]);
      setSelectedEventId(null);
      showEmptyMessages(sessionId);
    } catch (error) {
      setModelError(String(error));
    }
  }

  function toggleWorkspaceGroup(workspaceId) {
    setCollapsedWorkspaceIds((current) => toggleWorkspaceCollapsedIds(current, workspaceId));
  }

  function expandWorkspaceGroup(workspaceId) {
    setCollapsedWorkspaceIds((current) => expandWorkspaceInCollapsedIds(current, workspaceId));
  }

  function openSessionSearch() {
    setSessionSearchQuery("");
    setSessionSearchOpen(true);
  }

  function closeSessionSearch() {
    setSessionSearchOpen(false);
    setSessionSearchQuery("");
  }

  async function chooseSessionSearchResult(result) {
    closeSessionSearch();
    expandWorkspaceGroup(result.workspaceId);
    await openSession(result.workspaceId, result.id);
  }

  function showWorkspaceRequiredDialog() {
    setWorkspacePromptOpen(true);
  }

  function animateSessionTitle(event) {
    const sessionId = event?.session_id;
    const title = String(event?.title ?? "").trim();
    if (!sessionId || !title) return;

    const existingTimers = titleAnimationTimersRef.current.get(sessionId) ?? [];
    existingTimers.forEach((timer) => window.clearTimeout(timer));

    const characters = Array.from(title);
    const timers = [];

    setSessionTitle(sessionId, characters.slice(0, 1).join(""));
    if (characters.length === 1) {
      titleAnimationTimersRef.current.delete(sessionId);
      return;
    }

    for (let index = 2; index <= characters.length; index += 1) {
      const timer = window.setTimeout(() => {
        setSessionTitle(sessionId, characters.slice(0, index).join(""));
        if (index === characters.length) {
          titleAnimationTimersRef.current.delete(sessionId);
        }
      }, (index - 1) * 34);
      timers.push(timer);
    }

    titleAnimationTimersRef.current.set(sessionId, timers);
  }

  function setSessionTitle(sessionId, title) {
    setWorkspaces((items) =>
      items.map((workspace) => ({
        ...workspace,
        sessions: workspace.sessions.map((session) =>
          session.id === sessionId ? { ...session, title } : session,
        ),
      })),
    );
  }

  function setSessionModel(sessionId, patch) {
    setWorkspaces((items) =>
      items.map((workspace) => ({
        ...workspace,
        sessions: workspace.sessions.map((session) =>
          session.id === sessionId
            ? {
                ...session,
                ...patch,
              }
            : session,
        ),
      })),
    );
  }

  async function changeModel(nextModel) {
    const selected = resolveModelOptionByKey(nextModel, modelOptions);
    if (!selected) return;

    setModel(selected.key);
    setModelMenuOpen(false);
    const sessionId = activeSessionIdRef.current;
    if (!sessionId) return;

    setSessionModel(sessionId, {
      modelProvider: selected.providerId,
      model: selected.modelId,
      thinkingEnabled,
      reasoningEffort,
    });
    try {
      const saved = await invoke("update_session_model", {
        sessionId,
        modelProvider: selected.providerId,
        model: selected.modelId,
        thinkingEnabled,
        reasoningEffort,
      });
      const mappedSession = mapSessionRecord(saved);
      setSessionModel(mappedSession.id, mappedSession);
    } catch (error) {
      setModelError(String(error));
    }
  }

  async function changeThinkingEnabled(nextThinkingEnabled) {
    setThinkingEnabled(nextThinkingEnabled);
    await saveActiveModelConfig({ thinking: nextThinkingEnabled });
  }

  async function changeReasoningEffort(nextReasoningEffort) {
    setReasoningEffort(nextReasoningEffort);
    await saveActiveModelConfig({ effort: nextReasoningEffort });
  }

  async function saveActiveModelConfig({ thinking = thinkingEnabled, effort = reasoningEffort } = {}) {
    const sessionId = activeSessionIdRef.current;
    const selected = resolveModelOptionByKey(model, modelOptions);
    if (!sessionId || !selected) return;

    setSessionModel(sessionId, {
      modelProvider: selected.providerId,
      model: selected.modelId,
      thinkingEnabled: thinking,
      reasoningEffort: effort,
    });
    try {
      const saved = await invoke("update_session_model", {
        sessionId,
        modelProvider: selected.providerId,
        model: selected.modelId,
        thinkingEnabled: thinking,
        reasoningEffort: effort,
      });
      const mappedSession = mapSessionRecord(saved);
      setSessionModel(mappedSession.id, mappedSession);
    } catch (error) {
      setModelError(String(error));
    }
  }

  async function chooseWorkspaceFromPrompt() {
    setWorkspacePromptOpen(false);
    await openDirectoryPicker();
  }

  async function createSessionForWorkspace(workspaceId, name = "") {
    if (!workspaceId) throw new Error("Open a workspace before creating a session");
    const selected = resolveModelOptionByKey(model, modelOptions) ?? modelOptions[0];

    const session = await invoke("create_session", {
      request: {
        workspace_id: workspaceId,
        name,
        model_provider: selected.providerId,
        model: selected.modelId,
        thinking_enabled: thinkingEnabled,
        reasoning_effort: reasoningEffort,
      },
    });
    const mappedSession = mapSessionRecord(session);

    setWorkspaces((items) =>
      items.map((workspace) =>
        workspace.id === workspaceId
          ? { ...workspace, sessions: [mappedSession, ...workspace.sessions] }
          : workspace,
      ),
    );
    setActiveSessionId(mappedSession.id);
    return mappedSession.id;
  }

  async function openDirectoryPicker() {
    try {
      const selected = await open({ directory: true, multiple: false });
      const selectedPath = Array.isArray(selected) ? selected[0] : selected;
      if (selectedPath) await addWorkspaceFromPath(String(selectedPath));
    } catch {
      directoryInputRef.current?.click();
    }
  }

  function handleDirectorySelected(event) {
    const files = Array.from(event.target.files ?? []);
    event.target.value = "";
    if (files.length === 0) return;

    const firstFile = files[0];
    const relativePath = firstFile.webkitRelativePath || firstFile.name;
    const rootName = relativePath.split(/[\\/]/).filter(Boolean)[0] || "Untitled Workspace";
    const workspacePath = getSelectedDirectoryPath(firstFile, relativePath, rootName);
    addWorkspaceFromPath(workspacePath, rootName);
  }

  async function addWorkspaceFromPath(workspacePath, fallbackName) {
    const workspaceName = fallbackName || getWorkspaceName(workspacePath);
    try {
      const bundle = await invoke("open_workspace", {
        request: {
          name: workspaceName,
          absolute_path: workspacePath,
        },
      });
      const openedWorkspace = mapWorkspaceBundle(bundle);

      setWorkspaces((items) => [
        openedWorkspace,
        ...items.filter((workspace) => workspace.id !== openedWorkspace.id),
      ]);
      setActiveWorkspaceId(openedWorkspace.id);
      setActiveSessionId(openedWorkspace.sessions[0]?.id ?? null);
      expandWorkspaceGroup(openedWorkspace.id);
      setDraftSession(null);
      setView("chat");
      setTimeline([]);
      setSelectedEventId(null);
      if (openedWorkspace.sessions[0]) await loadSessionMessages(openedWorkspace.sessions[0].id);
      else showEmptyMessages();
    } catch (error) {
      setModelError(String(error));
    }
  }

  function beginWorkspaceDrag(event, workspaceId) {
    if (event.button !== 0 || event.target.closest("button")) return;

    const rect = event.currentTarget.getBoundingClientRect();
    workspaceDragRef.current = {
      id: workspaceId,
      pointerId: event.pointerId,
      startY: event.clientY,
      currentY: event.clientY,
      offsetY: event.clientY - rect.top,
      left: rect.left,
      width: rect.width,
      hasMoved: false,
    };
    window.addEventListener("pointermove", handleWorkspacePointerMove);
    window.addEventListener("pointerup", finishWorkspaceDrag, { once: true });
    window.addEventListener("pointercancel", finishWorkspaceDrag, { once: true });
  }

  function handleWorkspacePointerMove(event) {
    const drag = workspaceDragRef.current;
    if (!drag || event.pointerId !== drag.pointerId) return;

    if (!drag.hasMoved && Math.abs(event.clientY - drag.startY) < 4) return;

    drag.hasMoved = true;
    drag.currentY = event.clientY;
    suppressWorkspaceClickRef.current = true;
    setDraggedWorkspaceId(drag.id);
    setWorkspaceDragPreview({
      id: drag.id,
      top: event.clientY - drag.offsetY,
      left: drag.left,
      width: drag.width,
    });
    moveWorkspaceToPointer(drag.id, event.clientY);
  }

  function finishWorkspaceDrag(event) {
    const drag = workspaceDragRef.current;
    if (event?.pointerId && drag?.pointerId !== event.pointerId) return;

    window.removeEventListener("pointermove", handleWorkspacePointerMove);
    window.removeEventListener("pointerup", finishWorkspaceDrag);
    window.removeEventListener("pointercancel", finishWorkspaceDrag);
    workspaceDragRef.current = null;
    setDraggedWorkspaceId(null);
    setWorkspaceDragPreview(null);
  }

  function moveWorkspaceToPointer(workspaceId, pointerY) {
    const list = workspaceListRef.current;
    if (!list) return;

    const groups = Array.from(list.querySelectorAll(".workspace-group"));
    const target = groups.find((group) => {
      if (group.dataset.workspaceId === workspaceId) return false;
      const rect = group.getBoundingClientRect();
      return pointerY < rect.top + rect.height / 2;
    });
    const targetId = target?.dataset.workspaceId ?? null;

    setWorkspaces((items) => {
      const draggedWorkspace = items.find((workspace) => workspace.id === workspaceId);
      if (!draggedWorkspace) return items;

      const withoutDragged = items.filter((workspace) => workspace.id !== workspaceId);
      const targetIndex = targetId
        ? withoutDragged.findIndex((workspace) => workspace.id === targetId)
        : withoutDragged.length;
      if (targetIndex < 0) return items;

      const next = [...withoutDragged];
      next.splice(targetIndex, 0, draggedWorkspace);
      if (next.every((workspace, index) => workspace.id === items[index]?.id)) return items;
      return next;
    });
  }

  function openWorkspaceContextMenu(event, workspaceId) {
    event.preventDefault();
    setContextMenu({
      type: "workspace",
      workspaceId,
      x: event.clientX,
      y: event.clientY,
    });
  }

  function openSessionContextMenu(event, workspaceId, sessionId) {
    event.preventDefault();
    event.stopPropagation();
    setContextMenu({
      type: "session",
      workspaceId,
      sessionId,
      x: event.clientX,
      y: event.clientY,
    });
  }

  async function removeWorkspace(workspaceId) {
    try {
      await invoke("hide_workspace", { workspaceId });
    } catch (error) {
      setModelError(String(error));
      return;
    }

    const nextWorkspaces = workspaces.filter((workspace) => workspace.id !== workspaceId);
    setWorkspaces(nextWorkspaces);
    expandWorkspaceGroup(workspaceId);
    setContextMenu(null);

    if (activeWorkspaceId === workspaceId) {
      const nextWorkspace = nextWorkspaces[0] ?? null;
      setActiveWorkspaceId(nextWorkspace?.id ?? null);
      setActiveSessionId(nextWorkspace?.sessions[0]?.id ?? null);
      setDraftSession(null);
      if (nextWorkspace?.sessions[0]) {
        const sessionId = nextWorkspace.sessions[0].id;
        if (runningSessionIdsRef.current.has(sessionId)) showCachedMessagesForSession(sessionId);
        else loadSessionMessages(sessionId);
      }
      else showEmptyMessages();
    }
  }

  async function deleteWorkspaceSessions(workspaceId) {
    try {
      await invoke("delete_workspace_sessions", { workspaceId });
    } catch (error) {
      setModelError(String(error));
      return;
    }

    setWorkspaces((items) =>
      items.map((workspace) =>
        workspace.id === workspaceId ? { ...workspace, sessions: [] } : workspace,
      ),
    );
    const removedSessionIds =
      workspaces.find((workspace) => workspace.id === workspaceId)?.sessions.map((session) => session.id) ??
      [];
    const nextRunning = new Set(runningSessionIdsRef.current);
    const nextCanceling = new Set(cancelingSessionIdsRef.current);
    const nextTasks = new Map(taskIdsBySessionRef.current);
    for (const sessionId of removedSessionIds) nextRunning.delete(sessionId);
    for (const sessionId of removedSessionIds) {
      nextCanceling.delete(sessionId);
      nextTasks.delete(sessionId);
    }
    messagesBySessionRef.current = deleteCachedSessionMessages(
      messagesBySessionRef.current,
      removedSessionIds,
    );
    runningSessionIdsRef.current = nextRunning;
    cancelingSessionIdsRef.current = nextCanceling;
    taskIdsBySessionRef.current = nextTasks;
    setRunningSessionIds(nextRunning);
    setCancelingSessionIds(nextCanceling);
    setTaskIdsBySession(nextTasks);
    setContextMenu(null);

    if (activeWorkspaceId === workspaceId) {
      setActiveSessionId(null);
      activeSessionIdRef.current = null;
      setDraftSession(null);
      setModel(resolveModelForSession(null, modelOptions));
      showEmptyMessages();
    }
  }

  async function deleteSession(workspaceId, sessionId) {
    try {
      await invoke("delete_session", { sessionId });
    } catch (error) {
      setModelError(String(error));
      return;
    }

    const workspace = workspaces.find((item) => item.id === workspaceId);
    const remainingSessions = workspace?.sessions.filter((session) => session.id !== sessionId) ?? [];

    setWorkspaces((items) =>
      items.map((item) =>
        item.id === workspaceId ? { ...item, sessions: remainingSessions } : item,
      ),
    );
    setContextMenu(null);
    setSessionRunning(sessionId, false);
    setSessionCanceling(sessionId, false);
    setSessionTask(sessionId, null);
    messagesBySessionRef.current = deleteCachedSessionMessages(messagesBySessionRef.current, [
      sessionId,
    ]);

    if (activeWorkspaceId === workspaceId && activeSessionId === sessionId) {
      setActiveSessionId(remainingSessions[0]?.id ?? null);
      activeSessionIdRef.current = remainingSessions[0]?.id ?? null;
      setDraftSession(null);
      setModel(resolveModelForSession(remainingSessions[0], modelOptions));
      if (remainingSessions[0]) {
        const nextSessionId = remainingSessions[0].id;
        if (runningSessionIdsRef.current.has(nextSessionId)) {
          showCachedMessagesForSession(nextSessionId);
        } else {
          loadSessionMessages(nextSessionId);
        }
      }
      else showEmptyMessages(sessionId);
    }
  }

  async function openSession(workspaceId, sessionId) {
    setActiveWorkspaceId(workspaceId);
    setActiveSessionId(sessionId);
    activeSessionIdRef.current = sessionId;
    setDraftSession(null);
    setView("chat");
    const taskId = taskIdsBySessionRef.current.get(sessionId) ?? null;
    setActiveTaskId(taskId);
    activeTaskIdRef.current = taskId;
    setModel(resolveModelForSession(findSessionById(workspaces, sessionId), modelOptions));
    setModelError(null);
    setTimeline([]);
    setSelectedEventId(null);
    if (runningSessionIdsRef.current.has(sessionId)) showCachedMessagesForSession(sessionId);
    else await loadSessionMessages(sessionId);
  }

  async function openWorkspaceInSidebar(workspace) {
    setActiveWorkspaceId(workspace.id);
    setActiveSessionId(workspace.sessions[0]?.id ?? null);
    activeSessionIdRef.current = workspace.sessions[0]?.id ?? null;
    setDraftSession(null);
    setView("chat");
    setModel(resolveModelForSession(workspace.sessions[0], modelOptions));
    setTimeline([]);
    setSelectedEventId(null);
    if (workspace.sessions[0]) {
      const sessionId = workspace.sessions[0].id;
      if (runningSessionIdsRef.current.has(sessionId)) showCachedMessagesForSession(sessionId);
      else await loadSessionMessages(sessionId);
    }
    else showEmptyMessages();
  }

  async function loadSessionMessages(sessionId) {
    try {
      const records = await invoke("list_session_messages", {
        sessionId,
        turnLimit: SESSION_MESSAGE_PAGE_TURNS,
      });
      if (activeSessionIdRef.current !== sessionId) return;
      messagePageStateRef.current.set(sessionId, messagePageStateFromRecords(records));
      setFirstItemIndex(TRANSCRIPT_START_INDEX);
      pendingBottomScrollRef.current = true;
      suppressNextFollowOutputRef.current = false;
      transcriptPinnedToBottomRef.current = true;
      setTranscriptResetKey((current) => current + 1);
      replaceMessagesForSession(sessionId, mapMessageRecords(records));
      loadSessionContextUsage(sessionId);
      loadSessionStats(sessionId);
    } catch (error) {
      setModelError(String(error));
    }
  }

  async function loadOlderSessionMessages(sessionId) {
    if (!sessionId) return;
    const pageState = messagePageStateRef.current.get(sessionId);
    if (!pageState?.hasMore || pageState.loading || pageState.earliestTurn == null) return;

    messagePageStateRef.current.set(sessionId, { ...pageState, loading: true });

    try {
      const records = await invoke("list_session_messages", {
        sessionId,
        beforeTurnSequence: pageState.earliestTurn,
        turnLimit: SESSION_MESSAGE_PAGE_TURNS,
      });
      const nextPageState = messagePageStateFromRecords(records, pageState.earliestTurn);
      messagePageStateRef.current.set(sessionId, nextPageState);
      if (activeSessionIdRef.current !== sessionId || records.length === 0) return;

      // Prepend older messages and shift the virtual index so Virtuoso keeps the
      // current message under the viewport instead of jumping.
      const olderMessages = mapMessageRecords(records);
      setFirstItemIndex((current) => current - olderMessages.length);
      suppressNextFollowOutputRef.current = true;
      updateMessagesForSession(sessionId, (items) => [...olderMessages, ...items]);
    } catch (error) {
      messagePageStateRef.current.set(sessionId, { ...pageState, loading: false });
      setModelError(String(error));
    }
  }

  // Loads the persisted input-token watermark so the usage indicator is
  // populated even before a new round runs in the reopened session.
  async function loadSessionContextUsage(sessionId) {
    try {
      const tokens = await invoke("session_context_usage", { sessionId });
      setSessionContextUsage(sessionId, tokens);
    } catch (error) {
      // Non-fatal: the indicator simply stays at its previous value.
      console.warn("failed to load session context usage", error);
    }
  }

  async function saveSettings(event) {
    event.preventDefault();
    setSettingsStatus("saving");
    try {
      const saved = await invoke("save_app_settings", { settings });
      setSettings(mapLoadedSettings(saved));
      setSettingsStatus("saved");
    } catch (error) {
      setSettingsStatus(`Save failed: ${error}`);
    }
  }

  async function fetchModelsForSettings(providerId = DEFAULT_PROVIDER_ID) {
    const provider = settingsProviderById(settings, providerId);
    const baseUrl = provider.base_url.trim();
    const apiKey = provider.api_key.trim();
    if (!baseUrl || !apiKey || fetchingModels) return;

    setFetchingModels(provider.id);
    setSettingsStatus("正在获取模型...");
    try {
      const models = await invoke("fetch_provider_models", {
        baseUrl,
        apiKey,
      });
      const normalizedModels = normalizeModelOptions(models);
      setSettings((current) => setProviderModels(current, provider.id, normalizedModels));
      setSettingsStatus(`已获取 ${normalizedModels.length} 个模型`);
    } catch (error) {
      setSettingsStatus(`获取模型失败: ${error}`);
    } finally {
      setFetchingModels(null);
    }
  }

  return (
    <div
      className={`app-shell ${view === "settings" ? "is-settings-view" : ""} ${
        panelCollapsed ? "is-panel-collapsed" : ""
      }`}
    >
      {view !== "settings" ? (
      <aside className="sidebar">
        <button className="new-chat-button" type="button" onClick={openDirectoryPicker}>
          <MessageSquarePlus size={17} />
          <span>新对话</span>
        </button>
        <input
          ref={directoryInputRef}
          className="directory-picker"
          type="file"
          webkitdirectory="true"
          directory=""
          multiple
          onChange={handleDirectorySelected}
        />

        <button
          className="sidebar-search-button"
          type="button"
          onClick={openSessionSearch}
          aria-haspopup="dialog"
        >
          <Search size={16} />
          <span>搜索对话</span>
        </button>

        <div
          className={`workspace-list ${draggedWorkspaceId ? "is-dragging-workspace" : ""}`}
          ref={workspaceListRef}
        >
          {workspaces.map((workspace) => {
            const workspaceCollapsed = isWorkspaceCollapsed(collapsedWorkspaceIds, workspace.id);
            return (
              <section
                className={`workspace-group ${draggedWorkspaceId === workspace.id ? "is-dragging" : ""} ${workspaceCollapsed ? "is-collapsed" : ""}`}
                key={workspace.id}
                data-workspace-id={workspace.id}
                onPointerDown={(event) => beginWorkspaceDrag(event, workspace.id)}
                onContextMenu={(event) => openWorkspaceContextMenu(event, workspace.id)}
              >
                <div
                  className={`workspace-nav-header ${activeWorkspaceId === workspace.id ? "is-active" : ""}`}
                  role="button"
                  tabIndex={0}
                  onClick={(event) => {
                    if (suppressWorkspaceClickRef.current) {
                      event.preventDefault();
                      suppressWorkspaceClickRef.current = false;
                      return;
                    }
                    openWorkspaceInSidebar(workspace);
                  }}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      openWorkspaceInSidebar(workspace);
                    }
                  }}
                >
                  <button
                    className="workspace-collapse-button"
                    type="button"
                    title={workspaceCollapsed ? "展开项目对话" : "收起项目对话"}
                    aria-label={workspaceCollapsed ? "展开项目对话" : "收起项目对话"}
                    aria-expanded={!workspaceCollapsed}
                    onClick={(event) => {
                      event.stopPropagation();
                      toggleWorkspaceGroup(workspace.id);
                    }}
                  >
                    {workspaceCollapsed ? <ChevronRight size={15} /> : <ChevronDown size={15} />}
                  </button>
                  <span className="workspace-nav-name">{workspace.name}</span>
                  <button
                    className="workspace-nav-action"
                    type="button"
                    title="New chat"
                    onClick={(event) => {
                      event.stopPropagation();
                      createBlankConversation(workspace.id);
                    }}
                  >
                    <Plus size={15} />
                  </button>
                </div>
                {workspaceCollapsed ? null : (
                  <div className="session-list">
                    {workspace.sessions.map((session) => {
                      const isRunning = runningSessionIds.has(session.id);
                      return (
                        <button
                          key={session.id}
                          className={`session-item ${activeSessionId === session.id ? "is-selected" : ""} ${isRunning ? "is-running" : ""}`}
                          onContextMenu={(event) =>
                            openSessionContextMenu(event, workspace.id, session.id)
                          }
                          onClick={() => openSession(workspace.id, session.id)}
                        >
                          <MessageSquare size={15} />
                          <span>{session.title}</span>
                          {isRunning ? null : <small>{session.updated}</small>}
                          <span
                            className="session-running-indicator"
                            aria-label={isRunning ? "Session is running" : undefined}
                          >
                            {isRunning ? <Loader2 size={12} /> : null}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                )}
              </section>
            );
          })}
        </div>

        {draggedWorkspace && workspaceDragPreview ? (
          <div
            className="workspace-drag-preview"
            style={{
              top: `${workspaceDragPreview.top}px`,
              left: `${workspaceDragPreview.left}px`,
              width: `${workspaceDragPreview.width}px`,
            }}
          >
            <div className={`workspace-nav-header ${activeWorkspaceId === draggedWorkspace.id ? "is-active" : ""}`}>
              <button className="workspace-collapse-button" type="button" tabIndex={-1}>
                {draggedWorkspaceCollapsed ? (
                  <ChevronRight size={15} />
                ) : (
                  <ChevronDown size={15} />
                )}
              </button>
              <span className="workspace-nav-name">{draggedWorkspace.name}</span>
              <button className="workspace-nav-action" type="button" tabIndex={-1}>
                <Plus size={15} />
              </button>
            </div>
            {draggedWorkspaceCollapsed ? null : (
              <div className="session-list">
                {draggedWorkspace.sessions.map((session) => {
                  const isRunning = runningSessionIds.has(session.id);
                  return (
                    <div
                      key={session.id}
                      className={`session-item ${activeSessionId === session.id ? "is-selected" : ""} ${isRunning ? "is-running" : ""}`}
                    >
                      <MessageSquare size={15} />
                      <span>{session.title}</span>
                      {isRunning ? null : <small>{session.updated}</small>}
                      <span className="session-running-indicator" aria-hidden="true">
                        {isRunning ? <Loader2 size={12} /> : null}
                      </span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        ) : null}

        {contextMenu ? (
          <div
            className="context-menu"
            role="menu"
            style={{ left: `${contextMenu.x}px`, top: `${contextMenu.y}px` }}
            onClick={(event) => event.stopPropagation()}
            onContextMenu={(event) => event.preventDefault()}
          >
            {contextMenu.type === "workspace" ? (
              <>
                <button
                  className="context-menu-item"
                  type="button"
                  role="menuitem"
                  onClick={() => removeWorkspace(contextMenu.workspaceId)}
                >
                  <XCircle size={15} />
                  <span>{"移除工作区"}</span>
                </button>
                <button
                  className="context-menu-item is-danger"
                  type="button"
                  role="menuitem"
                  onClick={() => deleteWorkspaceSessions(contextMenu.workspaceId)}
                >
                  <Trash2 size={15} />
                  <span>{"删除所有会话"}</span>
                </button>
              </>
            ) : (
              <button
                className="context-menu-item is-danger"
                type="button"
                role="menuitem"
                onClick={() => deleteSession(contextMenu.workspaceId, contextMenu.sessionId)}
              >
                <Trash2 size={15} />
                <span>{"删除会话"}</span>
              </button>
            )}
          </div>
        ) : null}

        <div className="sidebar-footer">
          <button
            className={`settings-button ${view === "settings" ? "is-active" : ""}`}
            title="设置"
            onClick={() => setView("settings")}
          >
            <Settings2 size={17} />
            <span>设置</span>
          </button>
        </div>
      </aside>
      ) : null}

      {view === "settings" ? (
        <SettingsView
          settings={settings}
          setSettings={setSettings}
          settingsStatus={settingsStatus}
          fetchingModels={fetchingModels}
          onFetchModels={fetchModelsForSettings}
          onSubmit={saveSettings}
          onBack={() => {
            setView("chat");
            loadSettingsFromDisk({ showStatus: false });
          }}
        />
      ) : (
        <>
          <main className="conversation">
            <Virtuoso
              key={transcriptResetKey}
              ref={virtuosoRef}
              className="transcript"
              style={{ height: "100%", overflowX: "hidden" }}
              data={messages}
              firstItemIndex={firstItemIndex}
              initialTopMostItemIndex={initialTranscriptLocation(messages.length)}
              scrollerRef={bindTranscriptScroller}
              startReached={() => loadOlderSessionMessages(activeSessionIdRef.current)}
              increaseViewportBy={{ top: 600, bottom: 600 }}
              components={TRANSCRIPT_COMPONENTS}
              computeItemKey={(_, message) => message.id}
              itemContent={(_, message) => <MessageBubble message={message} />}
            />


            <div className="composer-stack">
              {modelError ? (
                <div className="model-error-card" role="alert">
                  <XCircle size={17} />
                  <div>
                    <strong>调用失败</strong>
                    <span>{modelError}</span>
                  </div>
                </div>
              ) : null}

              {activeSessionRunning && !modelError ? (
                <div className="generating-card" role="status" aria-live="polite">
                  <span>{"正在生成内容"}</span>
                  <span className="generating-dots" aria-hidden="true">
                    <span />
                    <span />
                    <span />
                  </span>
                </div>
              ) : null}

              <form className="composer" onSubmit={submitPrompt}>
                <textarea
                  value={prompt}
                  onChange={(event) => setPrompt(event.target.value)}
                  placeholder=""
                  rows={4}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
                      submitPrompt(event);
                    }
                  }}
                />
                <div className="composer-bar">
                  <div className="composer-hint">
                    <Bot size={15} />
                    <span>Ctrl + Enter to send</span>
                  </div>
                  <div className="composer-actions">
                    <span
                      className="context-usage"
                      title={`已用 ${activeContextUsed} / 上下文 ${modelContextWindow} tokens（${activeContextPercent}%）`}
                    >
                      {formatContextTokens(activeContextUsed)}/{formatContextTokens(modelContextWindow)} ({activeContextPercent}%)
                    </span>
                    <div className="model-picker" ref={modelMenuRef}>
                      <button
                        className="model-trigger"
                        type="button"
                        aria-haspopup="menu"
                        aria-expanded={modelMenuOpen}
                        onClick={() => setModelMenuOpen((open) => !open)}
                      >
                        <span>{selectedModelOption?.label ?? "选择模型"}</span>
                        <ChevronDown size={14} />
                      </button>
                      {modelMenuOpen ? (
                        <div className="model-popover" role="menu">
                          <div className="model-popover-title">Models</div>
                          <div className="model-list">
                            {modelOptions.map((item, index) => (
                              <button
                                className={`model-option ${item.key === model ? "is-selected" : ""}`}
                                key={item.key}
                                type="button"
                                role="menuitemradio"
                                aria-checked={item.key === model}
                                onClick={() => changeModel(item.key)}
                              >
                                <span>{item.label}</span>
                                {item.key === model ? <Check size={15} /> : <small>{index + 1}</small>}
                              </button>
                            ))}
                          </div>
                          <div className="model-popover-divider" />
                          <div className="model-setting-row">
                            <div>
                              <span>思考模式</span>
                              <small>{`thinking: ${thinkingEnabled ? "enabled" : "disabled"}`}</small>
                            </div>
                            <button
                              className={`switch-button ${thinkingEnabled ? "is-on" : ""}`}
                              type="button"
                              role="switch"
                              aria-checked={thinkingEnabled}
                              onClick={() => changeThinkingEnabled(!thinkingEnabled)}
                            >
                              <span />
                            </button>
                          </div>
                          <div className="model-setting-block">
                            <div className="model-setting-heading">
                              <Brain size={14} />
                              <span>思考强度</span>
                            </div>
                            <div className="effort-segments" aria-disabled={!thinkingEnabled}>
                              {REASONING_EFFORTS.map((item) => (
                                <button
                                  className={reasoningEffort === item.id ? "is-selected" : ""}
                                  key={item.id}
                                  type="button"
                                  disabled={!thinkingEnabled}
                                  onClick={() => changeReasoningEffort(item.id)}
                                >
                                  {item.label}
                                </button>
                              ))}
                            </div>
                          </div>
                        </div>
                      ) : null}
                    </div>
                    <button
                      className="send-button"
                      type={activeSessionRunning ? "button" : "submit"}
                      disabled={
                        activeSessionRunning
                          ? activeSessionCanceling || !activeSessionTaskId
                          : !prompt.trim()
                      }
                      onClick={activeSessionRunning ? stopActiveTask : undefined}
                    >
                      {activeSessionRunning ? (
                        activeSessionCanceling ? (
                          <Loader2 size={17} />
                        ) : (
                          <Square size={17} />
                        )
                      ) : (
                        <Send size={17} />
                      )}
                      <span>
                        {activeSessionRunning
                          ? activeSessionCanceling
                            ? "Stopping"
                            : "Stop"
                          : "Send"}
                      </span>
                    </button>
                  </div>
                </div>
              </form>
            </div>
          </main>

          <WorkspacePanel
            stats={activeSessionStats}
            collapsed={panelCollapsed}
            onToggle={() => setPanelCollapsed((value) => !value)}
          />
        </>
      )}

      {sessionSearchOpen ? (
        <div
          className="session-search-backdrop"
          role="presentation"
          onPointerDown={closeSessionSearch}
        >
          <section
            className="session-search-dialog"
            role="dialog"
            aria-modal="true"
            aria-label="搜索对话"
            onPointerDown={(event) => event.stopPropagation()}
          >
            <div className="session-search-input-shell">
              <Search size={17} />
              <input
                ref={sessionSearchInputRef}
                value={sessionSearchQuery}
                placeholder="搜索对话名称"
                onChange={(event) => setSessionSearchQuery(event.target.value)}
              />
            </div>
            <div className="session-search-results">
              {sessionSearchResults.length > 0 ? (
                sessionSearchResults.map((result) => (
                  <button
                    className={`session-search-item ${
                      activeSessionId === result.id ? "is-selected" : ""
                    }`}
                    key={result.id}
                    type="button"
                    onClick={() => chooseSessionSearchResult(result)}
                  >
                    <span className="session-search-title">{result.title}</span>
                    <span className="session-search-workspace">{result.workspaceName}</span>
                  </button>
                ))
              ) : (
                <div className="session-search-empty">没有匹配的对话</div>
              )}
            </div>
          </section>
        </div>
      ) : null}

      {workspacePromptOpen ? (
        <div className="modal-backdrop" role="presentation">
          <section
            className="workspace-required-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="workspace-required-title"
          >
            <header className="workspace-required-header">
              <div className="workspace-required-icon">
                <Boxes size={20} />
              </div>
              <div>
                <h2 id="workspace-required-title">需要选择工作区</h2>
                <p>发送消息前，请先选择一个工作区。</p>
              </div>
            </header>
            <div className="workspace-required-actions">
              <button
                className="secondary-button"
                type="button"
                onClick={() => setWorkspacePromptOpen(false)}
              >
                取消
              </button>
              <button className="save-button" type="button" onClick={chooseWorkspaceFromPrompt}>
                选择工作区
              </button>
            </div>
          </section>
        </div>
      ) : null}
    </div>
  );
}
