"use client";
// The workspace chat: a panel on the right of the app. A model on OpenRouter answers from the
// workspace's records, files and runs, and cites numbered sources that open the record.
// The server side is lib/chat.ts; the stream format is documented in app/api/chat/route.ts.
import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  ArrowUp,
  FileText,
  History,
  LineChart,
  Plus,
  Search,
  Square,
  Trash2,
  X,
} from "lucide-react";
import type {
  ChatEvent,
  ChatMessageRow,
  ChatSource,
  ChatStep,
  ChatThread,
} from "@/lib/chat";
import type { ResearchRecord } from "@/lib/schema";
import { kindMeta, timeAgo, useWorkspaceApi } from "./ui";
import { Select } from "./select";

type Model = { id: string; label: string; kind: "llm" | "decision" };
/** The answer being written: not stored yet. */
type Draft = {
  content: string;
  sources: ChatSource[];
  steps: ChatStep[];
  error: string;
  stopped: boolean;
};
const emptyDraft: Draft = {
  content: "",
  sources: [],
  steps: [],
  error: "",
  stopped: false,
};
const suggestions = [
  "What did we find so far, and how well is it checked?",
  "Which experiments are running or planned?",
  "What is still unverified?",
];
const stepLabel: Record<string, string> = {
  search: "Searched",
  read_record: "Read",
  list_records: "Listed",
};
const MIN_WIDTH = 340;
const MAX_WIDTH = 760;

const stored = (key: string) => {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
};
const store = (key: string, value: string | null) => {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch {
    // private mode: the choice lasts for this page only
  }
};

/** `[3]` in an answer becomes a link the renderer turns into a source button. Code is left alone. */
const withCitations = (text: string) =>
  text
    .split(/(```[\s\S]*?```|`[^`\n]*`)/)
    .map((part, index) =>
      index % 2 ? part : part.replace(/\[(\d{1,3})\](?!\()/g, "[$1](#cite-$1)"),
    )
    .join("");

const citedNumbers = (text: string) =>
  new Set([...text.matchAll(/\[(\d{1,3})\]/g)].map((m) => Number(m[1])));

function SourceIcon({ source }: { source: ChatSource }) {
  if (source.source === "run") return <LineChart size={12} />;
  if (source.source === "file") return <FileText size={12} />;
  const Icon =
    kindMeta[source.kind as ResearchRecord["kind"]]?.icon ?? FileText;
  return <Icon size={12} />;
}

function Answer({
  content,
  sources,
  steps,
  footer,
  writing,
  onSource,
}: {
  content: string;
  sources: ChatSource[];
  steps: ChatStep[];
  footer?: ReactNode;
  writing?: boolean;
  onSource: (source: ChatSource) => void;
}) {
  const byNumber = new Map(sources.map((s) => [s.n, s]));
  const cited = [...citedNumbers(content)]
    .map((n) => byNumber.get(n))
    .filter((s): s is ChatSource => !!s)
    .sort((a, b) => a.n - b.n);
  return (
    <div className="chat-message assistant">
      {steps.length > 0 && (
        <details className="chat-steps" open={writing && !content}>
          <summary>
            <Search size={11} />
            {steps.length} lookup{steps.length === 1 ? "" : "s"}
            {writing && !content ? "…" : ""}
          </summary>
          <ul>
            {steps.map((step, index) => (
              <li key={index}>
                <span>{stepLabel[step.tool] ?? step.tool}</span> {step.input}
              </li>
            ))}
          </ul>
        </details>
      )}
      {content ? (
        <div className="markdown chat-markdown">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              a: ({ children, href }) => {
                const n = href?.match(/^#cite-(\d+)$/)?.[1];
                const source = n ? byNumber.get(Number(n)) : undefined;
                if (n && !source) return <>[{n}]</>;
                if (source)
                  return (
                    <button
                      type="button"
                      className="chat-cite"
                      title={`${source.kind}: ${source.title}`}
                      onClick={() => onSource(source)}
                    >
                      {source.n}
                    </button>
                  );
                return (
                  <a href={href} target="_blank" rel="noopener noreferrer">
                    {children}
                  </a>
                );
              },
              // An answer never loads a remote image.
              img: ({ alt }) => <span>[image: {alt}]</span>,
            }}
          >
            {withCitations(content)}
          </ReactMarkdown>
        </div>
      ) : (
        writing && (
          <div className="chat-thinking" aria-label="Writing the answer">
            <i />
            <i />
            <i />
          </div>
        )
      )}
      {cited.length > 0 && (
        <div className="chat-sources">
          {cited.map((source) => (
            <button
              type="button"
              key={source.n}
              title={`Open ${source.kind}: ${source.title}`}
              onClick={() => onSource(source)}
            >
              <b>{source.n}</b>
              <SourceIcon source={source} />
              <span>{source.title}</span>
            </button>
          ))}
        </div>
      )}
      {footer}
    </div>
  );
}

function usageLine(message: ChatMessageRow) {
  const usage = message.usage;
  return [
    message.model,
    usage ? `${usage.total_tokens.toLocaleString("en-US")} tokens` : "",
    usage?.cost_usd !== undefined
      ? `$${usage.cost_usd < 0.01 ? usage.cost_usd.toFixed(4) : usage.cost_usd.toFixed(2)}`
      : "",
  ]
    .filter(Boolean)
    .join(" · ");
}

export function ChatPanel({
  records,
  focus,
  onWidth,
  onClose,
  onOpenRecord,
  onOpenView,
  notify,
}: {
  records: ResearchRecord[];
  /** The record the user has open, offered as the subject of the next question. */
  focus: ResearchRecord | null;
  onWidth: (width: number) => void;
  onClose: () => void;
  onOpenRecord: (record: ResearchRecord) => void;
  onOpenView: (hash: string) => void;
  notify: (message: string) => void;
}) {
  const { id: workspaceId, api, url } = useWorkspaceApi();
  const threadKey = `parallax.chat.thread.${workspaceId}`;
  const [configured, setConfigured] = useState<boolean | null>(null);
  const [threads, setThreads] = useState<ChatThread[]>([]);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessageRow[]>([]);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [busy, setBusy] = useState(false);
  const [input, setInput] = useState("");
  const [history, setHistory] = useState(false);
  const [models, setModels] = useState<Model[]>([]);
  const [model, setModel] = useState("");
  const [useFocus, setUseFocus] = useState(true);
  const [loadError, setLoadError] = useState("");
  const abort = useRef<AbortController | null>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const textarea = useRef<HTMLTextAreaElement>(null);
  const pinned = useRef(true); // the list follows the answer until the user scrolls up

  const loadThreads = useCallback(async () => {
    const state = await api<{
      threads: ChatThread[];
      configured: boolean;
      defaultModel: string;
    }>("/api/chat");
    setThreads(state.threads);
    setConfigured(state.configured);
    return state;
  }, [api]);

  const openThread = useCallback(
    async (id: string | null) => {
      abort.current?.abort();
      setHistory(false);
      setDraft(null);
      setThreadId(id);
      store(threadKey, id);
      pinned.current = true;
      if (!id) return setMessages([]);
      try {
        const thread = await api<{ messages: ChatMessageRow[] }>(
          `/api/chat?thread=${id}`,
        );
        setMessages(thread.messages);
      } catch {
        // The conversation is gone (deleted in another tab): start a new one.
        setThreadId(null);
        store(threadKey, null);
        setMessages([]);
      }
    },
    [api, threadKey],
  );

  useEffect(() => {
    let live = true;
    (async () => {
      try {
        const [state, catalog] = await Promise.all([
          loadThreads(),
          api<{ models: Model[] }>("/api/evaluators/models"),
        ]);
        if (!live) return;
        const chatModels = catalog.models.filter((m) => m.kind === "llm");
        setModels(chatModels);
        const saved = stored("parallax.chat.model");
        setModel(
          saved && chatModels.some((m) => m.id === saved)
            ? saved
            : state.defaultModel,
        );
        const last = stored(threadKey);
        if (last && state.threads.some((t) => t.id === last))
          await openThread(last);
      } catch (error) {
        if (live) setLoadError((error as Error).message);
      }
    })();
    return () => {
      live = false;
      abort.current?.abort();
    };
  }, [api, loadThreads, openThread, threadKey]);

  useLayoutEffect(() => {
    const element = scroller.current;
    if (element && pinned.current) element.scrollTop = element.scrollHeight;
  }, [messages, draft]);
  useEffect(() => textarea.current?.focus(), []);
  useEffect(() => setUseFocus(true), [focus?.id]);

  const send = async (text: string) => {
    const message = text.trim();
    if (!message || busy) return;
    setInput("");
    setBusy(true);
    pinned.current = true;
    const question: ChatMessageRow = {
      id: `pending-${Date.now()}`,
      role: "user",
      content: message,
      sources: [],
      steps: [],
      model: "",
      usage: null,
      createdAt: new Date().toISOString(),
    };
    setMessages((list) => [...list, question]);
    let next: Draft = emptyDraft;
    const update = (change: Partial<Draft>) => {
      next = { ...next, ...change };
      setDraft(next);
    };
    update({});
    const controller = new AbortController();
    abort.current = controller;
    try {
      const response = await fetch(url("/api/chat"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message,
          ...(threadId ? { threadId } : {}),
          ...(model ? { model } : {}),
          ...(focus && useFocus ? { focusRecordId: focus.id } : {}),
        }),
        signal: controller.signal,
      });
      if (!response.ok || !response.body)
        throw new Error(
          (await response.json().catch(() => ({}))).error || "Request failed",
        );
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const frames = buffer.split("\n\n");
        buffer = frames.pop() ?? "";
        for (const frame of frames) {
          const data = frame.replace(/^data: /, "");
          if (!data || data === "[DONE]") continue;
          const event = JSON.parse(data) as
            ChatEvent | { type: "error"; detail: string };
          if (event.type === "thread") {
            setThreadId(event.thread.id);
            store(threadKey, event.thread.id);
          } else if (event.type === "text")
            update({ content: next.content + event.delta });
          else if (event.type === "step")
            update({ steps: [...next.steps, event.step] });
          else if (event.type === "sources") update({ sources: event.sources });
          else if (event.type === "message") {
            const answer = event.message;
            setMessages((list) => [...list, answer]);
            setDraft(null);
          } else if (event.type === "error") update({ error: event.detail });
        }
      }
    } catch (error) {
      if (controller.signal.aborted) update({ stopped: true });
      else update({ error: (error as Error).message });
    } finally {
      abort.current = null;
      setBusy(false);
      void loadThreads().catch(() => undefined);
      textarea.current?.focus();
    }
  };

  const openSource = (source: ChatSource) => {
    const record = records.find((r) => r.id === source.recordId);
    if (record) return onOpenRecord(record);
    if (source.source === "run")
      return onOpenView(`#runs/run/${encodeURIComponent(source.id)}`);
    if (source.source === "file") return onOpenView("#artifacts");
    notify("This record is no longer in the workspace");
  };

  const remove = async (thread: ChatThread) => {
    if (!window.confirm(`Delete the conversation "${thread.title}"?`)) return;
    try {
      await api(`/api/chat?thread=${thread.id}`, { method: "DELETE" });
      if (thread.id === threadId) await openThread(null);
      await loadThreads();
    } catch (error) {
      notify((error as Error).message);
    }
  };

  const resize = (event: React.PointerEvent) => {
    event.preventDefault();
    const move = (e: PointerEvent) =>
      onWidth(
        Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, window.innerWidth - e.clientX)),
      );
    const stop = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
      document.body.classList.remove("chat-resizing");
    };
    document.body.classList.add("chat-resizing");
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop);
  };

  const empty = !messages.length && !draft;
  return (
    <aside className="chat-panel" aria-label="Ask your research">
      <div
        className="chat-resize"
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize the chat"
        onPointerDown={resize}
      />
      <header className="chat-header">
        <div className="chat-title">
          <span className="eyebrow">ASK</span>
          <strong>
            {threads.find((t) => t.id === threadId)?.title || "Your research"}
          </strong>
        </div>
        <button
          className="icon-button"
          aria-label="New conversation"
          title="New conversation"
          onClick={() => void openThread(null)}
        >
          <Plus size={16} />
        </button>
        <button
          className={`icon-button ${history ? "active" : ""}`}
          aria-label="Conversations"
          title="Conversations"
          aria-expanded={history}
          onClick={() => setHistory(!history)}
        >
          <History size={16} />
        </button>
        <button
          className="icon-button"
          aria-label="Close the chat"
          title="Close (⌘J)"
          onClick={onClose}
        >
          <X size={16} />
        </button>
      </header>
      {history && (
        <div className="chat-history">
          {threads.length === 0 && <p>No conversation yet.</p>}
          {threads.map((thread) => (
            <div
              key={thread.id}
              className={thread.id === threadId ? "current" : ""}
            >
              <button onClick={() => void openThread(thread.id)}>
                <span>{thread.title}</span>
                <small>{timeAgo(thread.updatedAt)}</small>
              </button>
              <button
                className="icon-button"
                aria-label={`Delete the conversation ${thread.title}`}
                onClick={() => void remove(thread)}
              >
                <Trash2 size={13} />
              </button>
            </div>
          ))}
        </div>
      )}
      <div
        className="chat-scroll"
        ref={scroller}
        onScroll={(e) => {
          const el = e.currentTarget;
          pinned.current =
            el.scrollHeight - el.scrollTop - el.clientHeight < 40;
        }}
      >
        {loadError && <p className="chat-error">{loadError}</p>}
        {configured === false && (
          <div className="chat-notice">
            <strong>No OpenRouter key</strong>
            <p>
              The chat runs on your own OpenRouter key. Without one, Parallax
              contacts nobody.
            </p>
            <button
              className="button secondary compact"
              onClick={() => onOpenView("#traces/evaluators")}
            >
              Add a key
            </button>
          </div>
        )}
        {empty && configured !== false && (
          <div className="chat-empty">
            <h2>
              Ask your research<span className="heading-period">.</span>
            </h2>
            <p>
              Answers come from this workspace&apos;s records, files and runs,
              with the sources to open.
            </p>
            {suggestions.map((text) => (
              <button key={text} onClick={() => void send(text)}>
                {text}
              </button>
            ))}
          </div>
        )}
        {messages.map((message) =>
          message.role === "user" ? (
            <div className="chat-message user" key={message.id}>
              {message.content}
            </div>
          ) : (
            <Answer
              key={message.id}
              content={message.content}
              sources={message.sources}
              steps={message.steps}
              onSource={openSource}
              footer={<div className="chat-usage">{usageLine(message)}</div>}
            />
          ),
        )}
        {draft && (
          <Answer
            content={draft.content}
            sources={draft.sources}
            steps={draft.steps}
            writing={busy}
            onSource={openSource}
            footer={
              draft.error ? (
                <p className="chat-error">{draft.error}</p>
              ) : draft.stopped ? (
                <div className="chat-usage">
                  Stopped. This answer is not saved.
                </div>
              ) : undefined
            }
          />
        )}
      </div>
      <form
        className="chat-composer"
        onSubmit={(e) => {
          e.preventDefault();
          void send(input);
        }}
      >
        {focus && (
          <label className="chat-focus" title={focus.title}>
            <input
              type="checkbox"
              checked={useFocus}
              onChange={(e) => setUseFocus(e.target.checked)}
            />
            <span>About the open {focus.kind}:</span>
            <b>{focus.title}</b>
          </label>
        )}
        <div className="chat-input">
          <textarea
            ref={textarea}
            rows={1}
            value={input}
            placeholder="Ask about this workspace…"
            aria-label="Your question"
            disabled={configured === false}
            onChange={(e) => {
              setInput(e.target.value);
              e.target.style.height = "auto";
              e.target.style.height = `${Math.min(e.target.scrollHeight, 180)}px`;
            }}
            onKeyDown={(e) => {
              if (
                e.key === "Enter" &&
                !e.shiftKey &&
                !e.nativeEvent.isComposing
              ) {
                e.preventDefault();
                e.currentTarget.style.height = "auto";
                void send(input);
              }
            }}
          />
          {busy ? (
            <button
              type="button"
              className="chat-send"
              aria-label="Stop the answer"
              title="Stop"
              onClick={() => abort.current?.abort()}
            >
              <Square size={13} />
            </button>
          ) : (
            <button
              type="submit"
              className="chat-send"
              aria-label="Send"
              title="Send (Enter)"
              disabled={!input.trim() || configured === false}
            >
              <ArrowUp size={15} />
            </button>
          )}
        </div>
        <div className="chat-foot">
          {models.length > 0 && (
            <Select
              className="chat-model"
              label="Chat model"
              value={model}
              onValueChange={(value) => {
                setModel(value);
                store("parallax.chat.model", value);
              }}
              options={[
                ...models.map((m) => ({ value: m.id, label: m.label })),
                ...(model && !models.some((m) => m.id === model)
                  ? [{ value: model, label: model }]
                  : []),
              ]}
            />
          )}
          <span title="Your question, the earlier turns of this conversation and the passages the model reads go to OpenRouter, with your key. Records marked sensitive are included.">
            Sends what it reads to OpenRouter
          </span>
        </div>
      </form>
    </aside>
  );
}
