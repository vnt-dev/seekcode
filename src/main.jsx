import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import {
  Activity,
  Bot,
  Boxes,
  CheckCircle2,
  Clock3,
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
  SquareTerminal,
  Trash2,
  XCircle,
} from "lucide-react";
import "./styles.css";

const MODELS = [
  { id: "deepseek-v4-pro", label: "DeepSeek V4 Pro" },
  { id: "deepseek-v4-flash", label: "DeepSeek V4 Flash" },
];

const INITIAL_PROJECTS = [
  {
    id: "seekcode",
    name: "SeekCode",
    path: "D:/rust/SeekCode",
    accent: "#4f7cff",
    sessions: [
      { id: "s-1", title: "Agent event stream", updated: "now" },
      { id: "s-2", title: "DeepSeek provider", updated: "today" },
    ],
  },
  {
    id: "examples",
    name: "Examples",
    path: "local workspace",
    accent: "#0f9f7a",
    sessions: [{ id: "s-3", title: "Patch engine notes", updated: "yesterday" }],
  },
];

const WORKSPACE_ITEMS = [
  { icon: Code2, label: "agent-core", value: "events, tasks, stream loop" },
  { icon: Boxes, label: "tool-system", value: "8 system tools" },
  { icon: GitBranch, label: "deepseek-client", value: "SSE + tool calls" },
  { icon: SquareTerminal, label: "src-tauri", value: "command bridge" },
];

function App() {
  const [projects, setProjects] = useState(INITIAL_PROJECTS);
  const [activeProjectId, setActiveProjectId] = useState("seekcode");
  const [activeSessionId, setActiveSessionId] = useState("s-1");
  const [draftSession, setDraftSession] = useState(null);
  const [draggedProjectId, setDraggedProjectId] = useState(null);
  const [projectDragPreview, setProjectDragPreview] = useState(null);
  const [contextMenu, setContextMenu] = useState(null);
  const [view, setView] = useState("chat");
  const [model, setModel] = useState(MODELS[0].id);
  const [prompt, setPrompt] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const [activeTaskId, setActiveTaskId] = useState(null);
  const [modelError, setModelError] = useState(null);
  const [settings, setSettings] = useState({
    base_url: "https://api.deepseek.com",
    api_key: "",
  });
  const [settingsStatus, setSettingsStatus] = useState("idle");
  const [messages, setMessages] = useState([
    {
      id: "welcome",
      role: "assistant",
      content:
        "SeekCode is ready. Start a coding task to see answer tokens, reasoning, and tool calls stream in real time.",
      reasoning: "",
      events: [],
    },
  ]);
  const [timeline, setTimeline] = useState([]);
  const [selectedEventId, setSelectedEventId] = useState(null);
  const transcriptRef = useRef(null);
  const activeTaskIdRef = useRef(null);
  const directoryInputRef = useRef(null);
  const projectListRef = useRef(null);
  const projectDragRef = useRef(null);
  const suppressProjectClickRef = useRef(false);

  const activeProject = projects.find((project) => project.id === activeProjectId);
  const draggedProject = projects.find((project) => project.id === draggedProjectId);
  const selectedEvent = timeline.find((event) => event.id === selectedEventId);

  useEffect(() => {
    activeTaskIdRef.current = activeTaskId;
  }, [activeTaskId]);

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

    let canceled = false;
    setSettingsStatus("loading");
    invoke("load_app_settings")
      .then((loaded) => {
        if (canceled) return;
        setSettings({
          base_url: loaded.base_url ?? "https://api.deepseek.com",
          api_key: loaded.api_key ?? "",
        });
        setSettingsStatus("idle");
      })
      .catch((error) => {
        if (!canceled) setSettingsStatus(`Load failed: ${error}`);
      });

    return () => {
      canceled = true;
    };
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
        });
      }
      mutator(next[next.length - 1]);
      return next;
    });
  }

  function applyAgentEvent(event) {
    const { type, payload } = event;
    if (payload?.task_id && !activeTaskIdRef.current) {
      activeTaskIdRef.current = payload.task_id;
      setActiveTaskId(payload.task_id);
    }

    switch (type) {
      case "task_started":
        setIsRunning(true);
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
      case "assistant_token":
        updateAssistantMessage((message) => {
          message.content += payload.text;
        });
        break;
      case "assistant_reasoning":
        updateAssistantMessage((message) => {
          message.reasoning += payload.text;
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
          message.events.push({
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
          const item = message.events.find((tool) => tool.id === payload.tool_call_id);
          if (item) {
            item.status = payload.ok ? "done" : "failed";
            item.summary = payload.summary;
            item.output = payload.output;
            item.error = payload.error;
          }
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
        setIsRunning(false);
        appendTimeline("task", "Task finished", payload.task_id, payload, "success");
        break;
      case "failed":
        setIsRunning(false);
        appendTimeline("task", "Task failed", payload.error, payload, "danger");
        setModelError(String(payload.error ?? "Model call failed"));
        removeEmptyAssistantPlaceholder();
        break;
      case "canceled":
        setIsRunning(false);
        appendTimeline("task", "Task canceled", payload.task_id, payload, "danger");
        break;
      default:
        appendTimeline("event", type, "", payload);
    }
  }

  async function submitPrompt(event) {
    event.preventDefault();
    const text = prompt.trim();
    if (!text || isRunning) return;

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
      },
    ]);
    setTimeline([]);
    setSelectedEventId(null);
    setIsRunning(true);
    activeTaskIdRef.current = null;
    setActiveTaskId(null);

    try {
      const task = await invoke("start_agent_task", {
        request: {
          prompt: text,
          workspace_id: null,
          model,
        },
      });
      activeTaskIdRef.current = task.id;
      setActiveTaskId(task.id);
      setMessages((items) => {
        const next = [...items];
        const last = next[next.length - 1];
        if (last?.role === "assistant") last.taskId = task.id;
        return next;
      });
    } catch (error) {
      setIsRunning(false);
      appendTimeline("task", "Start failed", String(error), { error }, "danger");
      setModelError(String(error));
      removeEmptyAssistantPlaceholder();
    }
  }

  function removeEmptyAssistantPlaceholder() {
    setMessages((items) => {
      const next = [...items];
      const last = next[next.length - 1];
      if (last?.role === "assistant" && !last.content && !last.reasoning && !last.events?.length) {
        next.pop();
      }
      return next;
    });
  }

  function createBlankConversation(projectId) {
    const id = `draft-${createId()}`;
    setActiveProjectId(projectId);
    setActiveSessionId(id);
    setDraftSession({ id, projectId });
    setView("chat");
    setPrompt("");
    setIsRunning(false);
    setActiveTaskId(null);
    activeTaskIdRef.current = null;
    setModelError(null);
    setTimeline([]);
    setSelectedEventId(null);
    setMessages([]);
  }

  async function openDirectoryPicker() {
    try {
      const selected = await open({ directory: true, multiple: false });
      const selectedPath = Array.isArray(selected) ? selected[0] : selected;
      if (selectedPath) addProjectFromPath(String(selectedPath));
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
    const rootName = relativePath.split(/[\\/]/).filter(Boolean)[0] || "Untitled Project";
    const projectPath = getSelectedDirectoryPath(firstFile, relativePath, rootName);
    addProjectFromPath(projectPath, rootName);
  }

  function addProjectFromPath(projectPath, fallbackName) {
    const normalizedPath = normalizeProjectPath(projectPath);
    const projectName = fallbackName || getProjectName(projectPath);
    const existing = projects.find((project) => normalizeProjectPath(project.path) === normalizedPath);
    const id = `project-${createId()}`;

    setProjects((items) => {
      const existingProject = items.find(
        (project) => normalizeProjectPath(project.path) === normalizedPath,
      );
      if (existingProject) {
        return [existingProject, ...items.filter((project) => project.id !== existingProject.id)];
      }

      return [
        {
          id,
          name: projectName,
          path: projectPath,
          accent: getProjectAccent(projectName),
          sessions: [],
        },
        ...items,
      ];
    });
    setActiveProjectId(existing?.id ?? id);
    setActiveSessionId(null);
    setDraftSession(null);
    setView("chat");
  }

  function beginProjectDrag(event, projectId) {
    if (event.button !== 0 || event.target.closest("button")) return;

    const rect = event.currentTarget.getBoundingClientRect();
    projectDragRef.current = {
      id: projectId,
      pointerId: event.pointerId,
      startY: event.clientY,
      currentY: event.clientY,
      offsetY: event.clientY - rect.top,
      left: rect.left,
      width: rect.width,
      hasMoved: false,
    };
    window.addEventListener("pointermove", handleProjectPointerMove);
    window.addEventListener("pointerup", finishProjectDrag, { once: true });
    window.addEventListener("pointercancel", finishProjectDrag, { once: true });
  }

  function handleProjectPointerMove(event) {
    const drag = projectDragRef.current;
    if (!drag || event.pointerId !== drag.pointerId) return;

    if (!drag.hasMoved && Math.abs(event.clientY - drag.startY) < 4) return;

    drag.hasMoved = true;
    drag.currentY = event.clientY;
    suppressProjectClickRef.current = true;
    setDraggedProjectId(drag.id);
    setProjectDragPreview({
      id: drag.id,
      top: event.clientY - drag.offsetY,
      left: drag.left,
      width: drag.width,
    });
    moveProjectToPointer(drag.id, event.clientY);
  }

  function finishProjectDrag(event) {
    const drag = projectDragRef.current;
    if (event?.pointerId && drag?.pointerId !== event.pointerId) return;

    window.removeEventListener("pointermove", handleProjectPointerMove);
    window.removeEventListener("pointerup", finishProjectDrag);
    window.removeEventListener("pointercancel", finishProjectDrag);
    projectDragRef.current = null;
    setDraggedProjectId(null);
    setProjectDragPreview(null);
  }

  function moveProjectToPointer(projectId, pointerY) {
    const list = projectListRef.current;
    if (!list) return;

    const groups = Array.from(list.querySelectorAll(".project-group"));
    const target = groups.find((group) => {
      if (group.dataset.projectId === projectId) return false;
      const rect = group.getBoundingClientRect();
      return pointerY < rect.top + rect.height / 2;
    });
    const targetId = target?.dataset.projectId ?? null;

    setProjects((items) => {
      const draggedProject = items.find((project) => project.id === projectId);
      if (!draggedProject) return items;

      const withoutDragged = items.filter((project) => project.id !== projectId);
      const targetIndex = targetId
        ? withoutDragged.findIndex((project) => project.id === targetId)
        : withoutDragged.length;
      if (targetIndex < 0) return items;

      const next = [...withoutDragged];
      next.splice(targetIndex, 0, draggedProject);
      if (next.every((project, index) => project.id === items[index]?.id)) return items;
      return next;
    });
  }

  function openProjectContextMenu(event, projectId) {
    event.preventDefault();
    setContextMenu({
      type: "project",
      projectId,
      x: event.clientX,
      y: event.clientY,
    });
  }

  function openSessionContextMenu(event, projectId, sessionId) {
    event.preventDefault();
    event.stopPropagation();
    setContextMenu({
      type: "session",
      projectId,
      sessionId,
      x: event.clientX,
      y: event.clientY,
    });
  }

  function removeProject(projectId) {
    const nextProjects = projects.filter((project) => project.id !== projectId);
    setProjects(nextProjects);
    setContextMenu(null);

    if (activeProjectId === projectId) {
      const nextProject = nextProjects[0] ?? null;
      setActiveProjectId(nextProject?.id ?? null);
      setActiveSessionId(nextProject?.sessions[0]?.id ?? null);
      setDraftSession(null);
      if (!nextProject) setMessages([]);
    }
  }

  function deleteProjectSessions(projectId) {
    setProjects((items) =>
      items.map((project) =>
        project.id === projectId ? { ...project, sessions: [] } : project,
      ),
    );
    setContextMenu(null);

    if (activeProjectId === projectId) {
      setActiveSessionId(null);
      setDraftSession(null);
      setMessages([]);
    }
  }

  function deleteSession(projectId, sessionId) {
    const project = projects.find((item) => item.id === projectId);
    const remainingSessions = project?.sessions.filter((session) => session.id !== sessionId) ?? [];

    setProjects((items) =>
      items.map((item) =>
        item.id === projectId ? { ...item, sessions: remainingSessions } : item,
      ),
    );
    setContextMenu(null);

    if (activeProjectId === projectId && activeSessionId === sessionId) {
      setActiveSessionId(remainingSessions[0]?.id ?? null);
      setDraftSession(null);
      if (remainingSessions.length === 0) setMessages([]);
    }
  }

  async function saveSettings(event) {
    event.preventDefault();
    setSettingsStatus("saving");
    try {
      const saved = await invoke("save_app_settings", { settings });
      setSettings({
        base_url: saved.base_url ?? settings.base_url,
        api_key: saved.api_key ?? settings.api_key,
      });
      setSettingsStatus("saved");
    } catch (error) {
      setSettingsStatus(`Save failed: ${error}`);
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
          <input placeholder="Search sessions or projects" />
        </div>

        <div
          className={`project-list ${draggedProjectId ? "is-dragging-project" : ""}`}
          ref={projectListRef}
        >
          {projects.map((project) => (
            <section
              className={`project-group ${draggedProjectId === project.id ? "is-dragging" : ""}`}
              key={project.id}
              data-project-id={project.id}
              onPointerDown={(event) => beginProjectDrag(event, project.id)}
              onContextMenu={(event) => openProjectContextMenu(event, project.id)}
            >
              <div
                className={`project-header ${activeProjectId === project.id ? "is-active" : ""}`}
                role="button"
                tabIndex={0}
                onClick={(event) => {
                  if (suppressProjectClickRef.current) {
                    event.preventDefault();
                    suppressProjectClickRef.current = false;
                    return;
                  }
                  setActiveProjectId(project.id);
                  setView("chat");
                }}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    setActiveProjectId(project.id);
                    setView("chat");
                  }
                }}
              >
                <span className="project-dot" style={{ backgroundColor: project.accent }} />
                <span className="project-name">{project.name}</span>
                <button
                  className="project-action"
                  type="button"
                  title="New chat"
                  onClick={(event) => {
                    event.stopPropagation();
                    createBlankConversation(project.id);
                  }}
                >
                  <Plus size={15} />
                </button>
              </div>
              <div className="session-list">
                {project.sessions.map((session) => (
                  <button
                    key={session.id}
                    className={`session-item ${activeSessionId === session.id ? "is-selected" : ""}`}
                    onContextMenu={(event) => openSessionContextMenu(event, project.id, session.id)}
                    onClick={() => {
                      setActiveProjectId(project.id);
                      setActiveSessionId(session.id);
                      setDraftSession(null);
                      setView("chat");
                    }}
                  >
                    <MessageSquare size={15} />
                    <span>{session.title}</span>
                    <small>{session.updated}</small>
                  </button>
                ))}
              </div>
            </section>
          ))}
        </div>

        {draggedProject && projectDragPreview ? (
          <div
            className="project-drag-preview"
            style={{
              top: `${projectDragPreview.top}px`,
              left: `${projectDragPreview.left}px`,
              width: `${projectDragPreview.width}px`,
            }}
          >
            <div className={`project-header ${activeProjectId === draggedProject.id ? "is-active" : ""}`}>
              <span className="project-dot" style={{ backgroundColor: draggedProject.accent }} />
              <span className="project-name">{draggedProject.name}</span>
              <button className="project-action" type="button" tabIndex={-1}>
                <Plus size={15} />
              </button>
            </div>
            <div className="session-list">
              {draggedProject.sessions.map((session) => (
                <div
                  key={session.id}
                  className={`session-item ${activeSessionId === session.id ? "is-selected" : ""}`}
                >
                  <MessageSquare size={15} />
                  <span>{session.title}</span>
                  <small>{session.updated}</small>
                </div>
              ))}
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
            {contextMenu.type === "project" ? (
              <>
                <button
                  className="context-menu-item"
                  type="button"
                  role="menuitem"
                  onClick={() => removeProject(contextMenu.projectId)}
                >
                  <XCircle size={15} />
                  <span>{"移除项目"}</span>
                </button>
                <button
                  className="context-menu-item is-danger"
                  type="button"
                  role="menuitem"
                  onClick={() => deleteProjectSessions(contextMenu.projectId)}
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
                onClick={() => deleteSession(contextMenu.projectId, contextMenu.sessionId)}
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
            title="Settings"
            onClick={() => setView("settings")}
          >
            <Settings2 size={17} />
            <span>Settings</span>
          </button>
        </div>
      </aside>
      ) : null}

      {view === "settings" ? (
        <SettingsView
          settings={settings}
          setSettings={setSettings}
          settingsStatus={settingsStatus}
          onSubmit={saveSettings}
          onBack={() => setView("chat")}
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

              {isRunning && !modelError ? (
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
                    <select value={model} onChange={(event) => setModel(event.target.value)}>
                      {MODELS.map((item) => (
                        <option key={item.id} value={item.id}>
                          {item.label}
                        </option>
                      ))}
                    </select>
                    <button className="send-button" type="submit" disabled={!prompt.trim() || isRunning}>
                      {isRunning ? <Loader2 size={17} /> : <Send size={17} />}
                      <span>{isRunning ? "Running" : "Send"}</span>
                    </button>
                  </div>
                </div>
              </form>
            </div>
          </main>

          <WorkspacePanel activeProject={activeProject} timeline={timeline} selectedEvent={selectedEvent} selectedEventId={selectedEventId} setSelectedEventId={setSelectedEventId} />
        </>
      )}
    </div>
  );
}

function WorkspacePanel({ activeProject, timeline, selectedEvent, selectedEventId, setSelectedEventId }) {
  return (
    <aside className="workspace">
      <header className="workspace-header">
        <div>
          <span>Workspace</span>
          <strong>{activeProject?.name}</strong>
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

function SettingsView({ settings, setSettings, settingsStatus, onSubmit, onBack }) {
  return (
    <main className="settings-page">
      <header className="settings-header">
        <div>
          <div className="crumb">
            <Settings2 size={15} />
            Application settings
          </div>
          <h1>Settings</h1>
        </div>
        <button className="secondary-button" onClick={onBack}>
          Back
        </button>
      </header>

      <div className="settings-layout">
        <nav className="settings-nav">
          <button className="settings-nav-item is-active" type="button">
            <Settings2 size={16} />
            <span>Configuration</span>
          </button>
        </nav>

        <form className="settings-form" onSubmit={onSubmit}>
          <section className="settings-panel">
            <div className="settings-panel-header">
              <div>
                <h2>Configuration</h2>
                <p>Configure the DeepSeek-compatible endpoint used by SeekCode.</p>
              </div>
            </div>

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

            <div className="settings-actions">
              <span className="settings-status">{settingsStatus}</span>
              <button className="save-button" type="submit">
                <Save size={16} />
                <span>Save</span>
              </button>
            </div>
          </section>
        </form>
      </div>
    </main>
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
  return (
    <article className={`message ${isUser ? "is-user" : "is-assistant"}`}>
      <div className="message-body">
        {message.reasoning ? (
          <details className="reasoning" open>
            <summary>
              <Sparkles size={14} />
              Reasoning
            </summary>
            <p>{message.reasoning}</p>
          </details>
        ) : null}
        <div className="message-content">
          {message.content}
        </div>
        {message.events?.length ? (
          <div className="tool-strip">
            {message.events.map((tool) => (
              <div className={`tool-chip is-${tool.status}`} key={tool.id}>
                <Hammer size={14} />
                <span>{tool.name}</span>
                {tool.status === "running" ? <Loader2 size={13} /> : null}
                {tool.status === "done" ? <CheckCircle2 size={13} /> : null}
                {tool.status === "failed" ? <XCircle size={13} /> : null}
              </div>
            ))}
          </div>
        ) : null}
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

function getSelectedDirectoryPath(file, relativePath, rootName) {
  if (!file?.path) return rootName;

  const relativeParts = relativePath.split(/[\\/]/).filter(Boolean);
  let directoryPath = file.path;
  for (let index = 1; index < relativeParts.length; index += 1) {
    directoryPath = directoryPath.replace(/[\\/][^\\/]*$/, "");
  }
  return directoryPath || rootName;
}

function normalizeProjectPath(path) {
  return String(path ?? "")
    .replace(/\\/g, "/")
    .replace(/\/+$/, "")
    .toLowerCase();
}

function getProjectName(path) {
  const normalized = String(path ?? "").replace(/\\/g, "/").replace(/\/+$/, "");
  return normalized.split("/").filter(Boolean).at(-1) || "Untitled Project";
}

function getProjectAccent(name) {
  const palette = ["#4f7cff", "#0f9f7a", "#b35c32", "#7c5cc4", "#ca3f5f", "#2b8ca3"];
  const total = Array.from(name).reduce((sum, char) => sum + char.charCodeAt(0), 0);
  return palette[total % palette.length];
}

function createId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

createRoot(document.getElementById("root")).render(<App />);
