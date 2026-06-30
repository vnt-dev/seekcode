import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Activity,
  Bot,
  Boxes,
  CheckCircle2,
  Clock3,
  CloudDownload,
  Code2,
  GitBranch,
  Hammer,
  Loader2,
  MessageSquare,
  MessageSquarePlus,
  PanelRight,
  Play,
  Plus,
  Save,
  Search,
  Send,
  Settings2,
  Sparkles,
  Square,
  SquareTerminal,
  Trash2,
  XCircle,
} from "lucide-react";
import "./styles.css";

const DEFAULT_MODELS = [
  { id: "deepseek-v4-pro", label: "DeepSeek V4 Pro" },
  { id: "deepseek-v4-flash", label: "DeepSeek V4 Flash" },
];
const DEFAULT_PROVIDER_ID = "default";
const DEFAULT_PROVIDER_NAME = "默认供应商";

const WORKSPACE_ITEMS = [
  { icon: Code2, label: "agent-core", value: "events, tasks, stream loop" },
  { icon: Boxes, label: "tool-system", value: "8 system tools" },
  { icon: GitBranch, label: "deepseek-client", value: "SSE + tool calls" },
  { icon: SquareTerminal, label: "src-tauri", value: "command bridge" },
];

function App() {
  const [workspaces, setWorkspaces] = useState([]);
  const [activeWorkspaceId, setActiveWorkspaceId] = useState(null);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [draftSession, setDraftSession] = useState(null);
  const [draggedWorkspaceId, setDraggedWorkspaceId] = useState(null);
  const [workspaceDragPreview, setWorkspaceDragPreview] = useState(null);
  const [contextMenu, setContextMenu] = useState(null);
  const [workspacePromptOpen, setWorkspacePromptOpen] = useState(false);
  const [view, setView] = useState("chat");
  const [model, setModel] = useState(modelKey(DEFAULT_PROVIDER_ID, DEFAULT_MODELS[0].id));
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
  const transcriptRef = useRef(null);
  const activeTaskIdRef = useRef(null);
  const activeSessionIdRef = useRef(null);
  const runningSessionIdsRef = useRef(new Set());
  const cancelingSessionIdsRef = useRef(new Set());
  const taskIdsBySessionRef = useRef(new Map());
  const titleAnimationTimersRef = useRef(new Map());
  const directoryInputRef = useRef(null);
  const workspaceListRef = useRef(null);
  const workspaceDragRef = useRef(null);
  const suppressWorkspaceClickRef = useRef(false);

  const activeWorkspace = workspaces.find((workspace) => workspace.id === activeWorkspaceId);
  const draggedWorkspace = workspaces.find((workspace) => workspace.id === draggedWorkspaceId);
  const selectedEvent = timeline.find((event) => event.id === selectedEventId);
  const activeSessionRunning = activeSessionId ? runningSessionIds.has(activeSessionId) : false;
  const activeSessionCanceling = activeSessionId ? cancelingSessionIds.has(activeSessionId) : false;
  const activeSessionTaskId = activeSessionId ? taskIdsBySession.get(activeSessionId) : null;
  const modelOptions = buildModelOptions(settings);

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
  }, [activeSessionId, model, settings.models, settings.providers, workspaces]);

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
    };
  }, []);

  useEffect(() => {
    transcriptRef.current?.scrollTo({
      top: transcriptRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, timeline.length]);

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

  function updateAssistantMessage(mutator) {
    setMessages((items) => {
      const next = [...items];
      const last = next[next.length - 1];
      const currentTaskId = activeTaskIdRef.current;
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
        appendTimeline(
          "model",
          `Model round ${payload.round_id}`,
          `${payload.message_count} messages / ${payload.tool_count} tools`,
          payload,
          "active",
        );
        break;
      case "model_choice":
        updateAssistantMessage((message) => {
          const delta = payload.choice?.delta ?? {};
          appendAssistantTextBlock(message, "reasoning", delta.reasoning_content);
          appendAssistantTextBlock(message, "content", delta.content);
          applyToolCallDeltas(message, delta.tool_calls ?? []);
        });
        break;
      case "assistant_token":
        updateAssistantMessage((message) => {
          appendAssistantTextBlock(message, "content", payload.text);
        });
        break;
      case "assistant_reasoning":
        updateAssistantMessage((message) => {
          appendAssistantTextBlock(message, "reasoning", payload.text);
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
        updateAssistantMessage((message) => {
          upsertToolCallBlock(message, {
            id: payload.tool_call_id,
            name: payload.name,
            status: "running",
            arguments: payload.arguments,
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
        updateAssistantMessage((message) => {
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
        appendTimeline(
          "model",
          `Round ${payload.round_id} finished`,
          payload.usage ? `${payload.usage.total_tokens} tokens` : "no usage",
          payload,
          "success",
        );
        break;
      case "finished":
        setSessionRunning(payload.session_id, false);
        setSessionCanceling(payload.session_id, false);
        setSessionTask(payload.session_id, null);
        appendTimeline("task", "Task finished", payload.task_id, payload, "success");
        break;
      case "failed":
        setSessionRunning(payload.session_id, false);
        setSessionCanceling(payload.session_id, false);
        setSessionTask(payload.session_id, null);
        appendTimeline("task", "Task failed", payload.error, payload, "danger");
        setModelError(String(payload.error ?? "Model call failed"));
        removeEmptyAssistantPlaceholder();
        break;
      case "canceled":
        setSessionRunning(payload.session_id, false);
        setSessionCanceling(payload.session_id, false);
        setSessionTask(payload.session_id, null);
        appendTimeline("task", "Task canceled", payload.task_id, payload, "danger");
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
    if (!sessionUsesModelOption(currentSession, selectedModel)) {
      try {
        const saved = await invoke("update_session_model", {
          sessionId,
          modelProvider: selectedModel.providerId,
          model: selectedModel.modelId,
        });
        const mappedSession = mapSessionRecord(saved);
        setSessionModel(
          mappedSession.id,
          mappedSession.modelProvider,
          mappedSession.model,
          mappedSession.updated,
        );
      } catch (error) {
        appendTimeline("task", "Start failed", String(error), { error }, "danger");
        setModelError(String(error));
        return;
      }
    }

    setPrompt("");
    setModelError(null);
    setMessages((items) => [
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
    activeSessionIdRef.current = sessionId;
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
        },
      });
      setSessionTask(sessionId, task.id);
      setMessages((items) => {
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
      removeEmptyAssistantPlaceholder();
    }
  }

  async function stopActiveTask() {
    const sessionId = activeSessionIdRef.current;
    const taskId = sessionId ? taskIdsBySessionRef.current.get(sessionId) : activeTaskIdRef.current;
    if (!sessionId || !taskId || cancelingSessionIdsRef.current.has(sessionId)) return;

    setSessionCanceling(sessionId, true);
    try {
      await invoke("cancel_agent_task", { taskId });
    } catch (error) {
      setSessionCanceling(sessionId, false);
      appendTimeline("task", "Cancel failed", String(error), { error }, "danger");
      setModelError(String(error));
    }
  }

  function removeEmptyAssistantPlaceholder() {
    setMessages((items) => {
      const next = [...items];
      const last = next[next.length - 1];
      if (last?.role === "assistant" && isAssistantMessageEmpty(last)) {
        next.pop();
      }
      return next;
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
      if (nextSession) await loadSessionMessages(nextSession.id);
      else setMessages([]);
    } catch (error) {
      setModelError(String(error));
    }
  }

  async function createBlankConversation(workspaceId) {
    try {
      const sessionId = await createSessionForWorkspace(workspaceId);
      setActiveWorkspaceId(workspaceId);
      setActiveSessionId(sessionId);
      setDraftSession(null);
      setView("chat");
      setPrompt("");
      activeSessionIdRef.current = sessionId;
      setActiveTaskId(null);
      activeTaskIdRef.current = null;
      setModelError(null);
      setTimeline([]);
      setSelectedEventId(null);
      setMessages([]);
    } catch (error) {
      setModelError(String(error));
    }
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

  function setSessionModel(sessionId, nextProvider, nextModel, nextUpdated) {
    setWorkspaces((items) =>
      items.map((workspace) => ({
        ...workspace,
        sessions: workspace.sessions.map((session) =>
          session.id === sessionId
            ? {
                ...session,
                modelProvider: nextProvider,
                model: nextModel,
                ...(nextUpdated ? { updated: nextUpdated } : {}),
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
    const sessionId = activeSessionIdRef.current;
    if (!sessionId) return;

    setSessionModel(sessionId, selected.providerId, selected.modelId);
    try {
      const saved = await invoke("update_session_model", {
        sessionId,
        modelProvider: selected.providerId,
        model: selected.modelId,
      });
      const mappedSession = mapSessionRecord(saved);
      setSessionModel(
        mappedSession.id,
        mappedSession.modelProvider,
        mappedSession.model,
        mappedSession.updated,
      );
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
        thinking_enabled: true,
        reasoning_effort: null,
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
      setDraftSession(null);
      setView("chat");
      setTimeline([]);
      setSelectedEventId(null);
      if (openedWorkspace.sessions[0]) await loadSessionMessages(openedWorkspace.sessions[0].id);
      else setMessages([]);
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
    setContextMenu(null);

    if (activeWorkspaceId === workspaceId) {
      const nextWorkspace = nextWorkspaces[0] ?? null;
      setActiveWorkspaceId(nextWorkspace?.id ?? null);
      setActiveSessionId(nextWorkspace?.sessions[0]?.id ?? null);
      setDraftSession(null);
      if (nextWorkspace?.sessions[0]) loadSessionMessages(nextWorkspace.sessions[0].id);
      else setMessages([]);
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
      setMessages([]);
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

    if (activeWorkspaceId === workspaceId && activeSessionId === sessionId) {
      setActiveSessionId(remainingSessions[0]?.id ?? null);
      activeSessionIdRef.current = remainingSessions[0]?.id ?? null;
      setDraftSession(null);
      setModel(resolveModelForSession(remainingSessions[0], modelOptions));
      if (remainingSessions[0]) loadSessionMessages(remainingSessions[0].id);
      else setMessages([]);
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
    await loadSessionMessages(sessionId);
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
    if (workspace.sessions[0]) await loadSessionMessages(workspace.sessions[0].id);
    else setMessages([]);
  }

  async function loadSessionMessages(sessionId) {
    try {
      const records = await invoke("list_session_messages", { sessionId });
      setMessages(mapMessageRecords(records));
    } catch (error) {
      setModelError(String(error));
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
    <div className={`app-shell ${view === "settings" ? "is-settings-view" : ""}`}>
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

        <div className="sidebar-search">
          <Search size={16} />
          <input placeholder="Search sessions or workspaces" />
        </div>

        <div
          className={`workspace-list ${draggedWorkspaceId ? "is-dragging-workspace" : ""}`}
          ref={workspaceListRef}
        >
          {workspaces.map((workspace) => (
            <section
              className={`workspace-group ${draggedWorkspaceId === workspace.id ? "is-dragging" : ""}`}
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
                <span className="workspace-nav-dot" style={{ backgroundColor: workspace.accent }} />
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
            </section>
          ))}
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
              <span className="workspace-nav-dot" style={{ backgroundColor: draggedWorkspace.accent }} />
              <span className="workspace-nav-name">{draggedWorkspace.name}</span>
              <button className="workspace-nav-action" type="button" tabIndex={-1}>
                <Plus size={15} />
              </button>
            </div>
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
            <section className="transcript" ref={transcriptRef}>
              {messages.map((message) => (
                <MessageBubble key={message.id} message={message} />
              ))}
            </section>

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
                    <select value={model} onChange={(event) => changeModel(event.target.value)}>
                      {modelOptions.map((item) => (
                        <option key={item.key} value={item.key}>
                          {item.label}
                        </option>
                      ))}
                    </select>
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

          <WorkspacePanel activeWorkspace={activeWorkspace} timeline={timeline} selectedEvent={selectedEvent} selectedEventId={selectedEventId} setSelectedEventId={setSelectedEventId} />
        </>
      )}

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

function WorkspacePanel({ activeWorkspace, timeline, selectedEvent, selectedEventId, setSelectedEventId }) {
  return (
    <aside className="workspace">
      <header className="workspace-header">
        <div>
          <span>Workspace</span>
          <strong>{activeWorkspace?.name}</strong>
        </div>
        <PanelRight size={18} />
      </header>

      <section className="workspace-section">
        <div className="section-title">Structure</div>
        <div className="workspace-items">
          {WORKSPACE_ITEMS.map((item) => (
            <WorkspaceItem item={item} key={item.label} />
          ))}
        </div>
      </section>

      <section className="workspace-section timeline-section">
        <div className="section-title">Run Timeline</div>
        <div className="timeline">
          {timeline.length === 0 ? (
            <div className="empty-state">
              <Activity size={22} />
              <span>Model output, reasoning, and tool calls appear here</span>
            </div>
          ) : (
            timeline.map((item) => (
              <button
                key={item.id}
                className={`timeline-item tone-${item.tone} ${selectedEventId === item.id ? "is-selected" : ""}`}
                onClick={() => setSelectedEventId(item.id)}
              >
                <EventIcon type={item.type} tone={item.tone} />
                <div>
                  <strong>{item.title}</strong>
                  <span>{item.detail}</span>
                </div>
                <time>{item.time}</time>
              </button>
            ))
          )}
        </div>
      </section>

      <section className="workspace-section detail-section">
        <div className="section-title">Event Detail</div>
        {selectedEvent ? (
          <pre>{JSON.stringify(selectedEvent.payload, null, 2)}</pre>
        ) : (
          <div className="empty-detail">Select a timeline event</div>
        )}
      </section>
    </aside>
  );
}

function SettingsView({
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
            <div className="settings-panel-header">
              <div>
                <h2>配置</h2>
                <p>配置 SeekCode 使用的默认模型供应商和额外模型供应商。</p>
              </div>
            </div>

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

function ModelCollectionEditor({ title, models, fetching, canFetch, onFetch, onChange }) {
  const editableModels = Array.isArray(models) && models.length > 0 ? models : [{ id: "", label: "" }];

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

function WorkspaceItem({ item }) {
  const Icon = item.icon;
  return (
    <div className="workspace-item">
      <Icon size={17} />
      <div>
        <strong>{item.label}</strong>
        <span>{item.value}</span>
      </div>
    </div>
  );
}

function MessageBubble({ message }) {
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
                    <p>{block.text}</p>
                  </details>
                );
              }

              if (block.type === "tool") {
                const tool = block.tool;
                return (
                  <div className="tool-strip message-block" key={block.id}>
                    <div className={`tool-chip is-${tool.status}`}>
                      <Hammer size={14} />
                      <span>{tool.name || "tool"}</span>
                      {tool.status === "running" ? <Loader2 size={13} /> : null}
                      {tool.status === "done" ? <CheckCircle2 size={13} /> : null}
                      {tool.status === "failed" ? <XCircle size={13} /> : null}
                    </div>
                  </div>
                );
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

function EventIcon({ type, tone }) {
  if (tone === "danger") return <XCircle size={16} />;
  if (tone === "success") return <CheckCircle2 size={16} />;
  if (type === "tool") return <Hammer size={16} />;
  if (type === "model") return <Bot size={16} />;
  if (type === "state") return <Clock3 size={16} />;
  if (type === "task") return <Play size={16} />;
  return <Activity size={16} />;
}

function stateTone(state) {
  if (state === "completed") return "success";
  if (state === "failed" || state === "canceled") return "danger";
  if (state === "thinking" || state === "running_tool") return "active";
  return "neutral";
}

function mapWorkspaceBundle(bundle) {
  const workspace = bundle.workspace;
  return {
    id: workspace.id,
    name: workspace.name,
    path: workspace.absolute_path,
    accent: getWorkspaceAccent(workspace.name),
    sessions: (bundle.sessions ?? []).map(mapSessionRecord),
  };
}

function mapSessionRecord(session) {
  return {
    id: session.id,
    title: session.name || "New chat",
    updated: shortDateTime(session.updated_at),
    model: session.model,
    modelProvider: session.model_provider,
  };
}

function mapLoadedSettings(settings) {
  return {
    base_url: settings?.base_url ?? "https://api.deepseek.com",
    api_key: settings?.api_key ?? "",
    title_model: settings?.title_model ?? "deepseek-v4-flash",
    models: normalizeModelOptions(settings?.models),
    providers: normalizeAdditionalProviders(settings?.providers),
  };
}

function normalizeModelOptions(models, fallback = DEFAULT_MODELS) {
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

function normalizeAdditionalProviders(providers) {
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

function buildModelOptions(settings) {
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

function modelKey(providerId, modelId) {
  return `${encodeURIComponent(providerId)}::${encodeURIComponent(modelId)}`;
}

function resolveModelOptionByKey(key, modelOptions) {
  const normalizedKey = String(key ?? "");
  return modelOptions.find((item) => item.key === normalizedKey) ?? modelOptions[0] ?? null;
}

function findSessionById(workspaces, sessionId) {
  if (!sessionId) return null;
  for (const workspace of workspaces ?? []) {
    const session = workspace.sessions.find((item) => item.id === sessionId);
    if (session) return session;
  }
  return null;
}

function resolveModelForSession(session, modelOptions) {
  const options = Array.isArray(modelOptions) && modelOptions.length > 0 ? modelOptions : buildModelOptions({});
  const sessionProvider = String(session?.modelProvider ?? "").trim();
  const sessionModel = String(session?.model ?? "").trim();
  const match = options.find(
    (item) => item.providerId === sessionProvider && item.modelId === sessionModel,
  );
  return match?.key ?? options[0].key;
}

function sessionUsesModelOption(session, option) {
  return (
    String(session?.modelProvider ?? "").trim() === option.providerId &&
    String(session?.model ?? "").trim() === option.modelId
  );
}

function settingsProviderById(settings, providerId) {
  if (!providerId || providerId === DEFAULT_PROVIDER_ID) {
    return {
      id: DEFAULT_PROVIDER_ID,
      base_url: settings?.base_url ?? "",
      api_key: settings?.api_key ?? "",
    };
  }
  return (
    normalizeAdditionalProviders(settings?.providers).find((provider) => provider.id === providerId) ?? {
      id: providerId,
      base_url: "",
      api_key: "",
    }
  );
}

function setProviderModels(settings, providerId, models) {
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

function updateProviderAt(settings, providerIndex, patch) {
  const providers = normalizeAdditionalProviders(settings.providers);
  providers[providerIndex] = { ...providers[providerIndex], ...patch };
  return { ...settings, providers };
}

function mapMessageRecords(records) {
  const turns = new Map();

  for (const record of records ?? []) {
    const key = record.turn_sequence;
    if (!turns.has(key)) turns.set(key, { users: [], assistant: null });
    const turn = turns.get(key);

    if (record.role === "user") {
      turn.users.push({
        id: record.id,
        role: "user",
        content: record.content,
      });
      continue;
    }

    if (record.role === "assistant") {
      if (!turn.assistant) {
        turn.assistant = {
          id: `assistant-${key}`,
          role: "assistant",
          content: "",
          reasoning: "",
          events: [],
          blocks: [],
        };
      }
      appendAssistantTextBlock(turn.assistant, "reasoning", record.reasoning_content);
      appendAssistantTextBlock(turn.assistant, "content", record.content);
      applyStoredToolCalls(turn.assistant, record.tool_calls ?? []);
      continue;
    }

    if (record.role === "tool") {
      if (!turn.assistant) {
        turn.assistant = {
          id: `assistant-${key}`,
          role: "assistant",
          content: "",
          reasoning: "",
          events: [],
          blocks: [],
        };
      }
      if (record.tool_call_id) {
        const patch = {
          id: record.tool_call_id,
          status: "done",
        };
        try {
          patch.output = JSON.parse(record.content || "null");
        } catch {
          patch.summary = record.content;
        }
        upsertToolCallBlock(turn.assistant, patch);
      }
    }
  }

  return Array.from(turns.entries())
    .sort(([left], [right]) => Number(left) - Number(right))
    .flatMap(([, turn]) => [...turn.users, turn.assistant].filter(Boolean));
}

function applyToolCallDeltas(message, toolCalls) {
  for (const call of toolCalls) {
    const id = call.id;
    if (!id) continue;
    upsertToolCallBlock(message, {
      id,
      name: call.name ?? call.function?.name,
      status: "running",
      argumentsDelta: call.arguments ?? call.function?.arguments ?? "",
    });
  }
}

function applyStoredToolCalls(message, toolCalls) {
  for (const call of toolCalls) {
    const id = call.id;
    if (!id) continue;
    upsertToolCallBlock(message, {
      id,
      name: call.function?.name,
      status: "running",
      argumentsDelta: call.function?.arguments ?? "",
    });
  }
}

function appendAssistantTextBlock(message, type, text) {
  if (!text) return;

  const field = type === "reasoning" ? "reasoning" : "content";
  message[field] = `${message[field] ?? ""}${text}`;

  const blocks = ensureAssistantBlocks(message);
  let block = blocks[blocks.length - 1];
  if (!block || block.type !== type) {
    block = {
      id: createId(),
      type,
      text: "",
    };
    blocks.push(block);
  }
  block.text += text;
}

function upsertToolCallBlock(message, patch) {
  if (!patch.id) return;

  if (!Array.isArray(message.events)) message.events = [];
  let item = message.events.find((tool) => tool.id === patch.id);
  if (!item) {
    item = {
      id: patch.id,
      name: "",
      status: "running",
      arguments: "",
    };
    message.events.push(item);
  }

  if (patch.name) item.name = patch.name;
  if (patch.status) item.status = patch.status;
  if (Object.prototype.hasOwnProperty.call(patch, "arguments")) {
    item.arguments = patch.arguments;
  }
  if (patch.argumentsDelta) {
    item.arguments = `${item.arguments ?? ""}${patch.argumentsDelta}`;
  }
  if (Object.prototype.hasOwnProperty.call(patch, "summary")) item.summary = patch.summary;
  if (Object.prototype.hasOwnProperty.call(patch, "output")) item.output = patch.output;
  if (Object.prototype.hasOwnProperty.call(patch, "error")) item.error = patch.error;

  const blocks = ensureAssistantBlocks(message);
  let block = blocks.find(
    (candidate) => candidate.type === "tool" && candidate.tool?.id === patch.id,
  );
  if (!block) {
    block = {
      id: `tool-${patch.id}`,
      type: "tool",
      tool: item,
    };
    blocks.push(block);
  } else {
    block.tool = item;
  }
}

function getAssistantBlocks(message) {
  return ensureAssistantBlocks(message).filter((block) => {
    if (block.type === "tool") return Boolean(block.tool);
    return Boolean(block.text);
  });
}

function ensureAssistantBlocks(message) {
  if (Array.isArray(message.blocks)) return message.blocks;

  message.blocks = [];
  if (message.reasoning) {
    message.blocks.push({
      id: createId(),
      type: "reasoning",
      text: message.reasoning,
    });
  }
  if (message.content) {
    message.blocks.push({
      id: createId(),
      type: "content",
      text: message.content,
    });
  }
  for (const tool of message.events ?? []) {
    message.blocks.push({
      id: `tool-${tool.id}`,
      type: "tool",
      tool,
    });
  }
  return message.blocks;
}

function isAssistantMessageEmpty(message) {
  if (message.content || message.reasoning || message.events?.length) return false;
  return getAssistantBlocks(message).length === 0;
}

function shortDateTime(value) {
  if (!value) return "";
  const [, time = value] = String(value).split(" ");
  return time.slice(0, 5) || value;
}

function getSelectedDirectoryPath(file, relativePath, rootName) {
  if (!file?.path) return rootName;

  const relativeParts = relativePath.split(/[\\/]/).filter(Boolean);
  let directoryPath = file.path;
  for (let index = 1; index < relativeParts.length; index += 1) {
    directoryPath = directoryPath.replace(/[\\/][^\\/]*$/, "");
  }
  return directoryPath || rootName;
}

function normalizeWorkspacePath(path) {
  return String(path ?? "")
    .replace(/\\/g, "/")
    .replace(/\/+$/, "")
    .toLowerCase();
}

function getWorkspaceName(path) {
  const normalized = String(path ?? "").replace(/\\/g, "/").replace(/\/+$/, "");
  return normalized.split("/").filter(Boolean).at(-1) || "Untitled Workspace";
}

function getWorkspaceAccent(name) {
  const palette = ["#4f7cff", "#0f9f7a", "#b35c32", "#7c5cc4", "#ca3f5f", "#2b8ca3"];
  const total = Array.from(name).reduce((sum, char) => sum + char.charCodeAt(0), 0);
  return palette[total % palette.length];
}

function createId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

createRoot(document.getElementById("root")).render(<App />);
