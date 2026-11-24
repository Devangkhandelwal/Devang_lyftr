import { useEffect, useMemo, useRef, useState } from "react";
import { sendChatMessage, API_URL } from "./api.js";

/**
 * A fresh session id for every page load.
 * Nothing is kept in localStorage, so reloading the page always starts a
 * brand new conversation.
 */
function createSessionId() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `s-${crypto.randomUUID()}`;
  }
  // crypto.randomUUID needs a secure context; this keeps the app working
  // when it is served over plain http from a remote host.
  return `s-${Math.random().toString(16).slice(2)}${Date.now().toString(16)}`;
}

function SourceBadge({ source }) {
  const isCache = source === "LLM_CACHE";
  return (
    <span
      className={`badge ${isCache ? "badge-cache" : "badge-live"}`}
      data-testid="meta-source"
    >
      {source}
    </span>
  );
}

/**
 * The metadata strip under an assistant reply.
 * Every meta-* element holds only the raw value. Labels live in their own
 * span outside the element carrying the test id.
 */
function MessageMetadata({ metadata }) {
  return (
    <div className="metadata">
      <SourceBadge source={metadata.resolution_source} />

      <span className="metadata-item">
        <span className="metadata-label">Model:</span>
        <span className="metadata-value" data-testid="meta-model">
          {metadata.model}
        </span>
      </span>

      <span className="metadata-item">
        <span className="metadata-label">Latency:</span>
        <span className="metadata-value" data-testid="meta-latency">
          {metadata.latency_ms} ms
        </span>
      </span>

      <span className="metadata-item">
        <span className="metadata-label">In:</span>
        <span className="metadata-value" data-testid="meta-input-tokens">
          {metadata.input_tokens}
        </span>
      </span>

      <span className="metadata-item">
        <span className="metadata-label">Out:</span>
        <span className="metadata-value" data-testid="meta-output-tokens">
          {metadata.output_tokens}
        </span>
      </span>

      <span className="metadata-item">
        <span className="metadata-label">Key:</span>
        <span className="metadata-value" data-testid="meta-key-id">
          {metadata.key_id === null || metadata.key_id === undefined
            ? "none"
            : metadata.key_id}
        </span>
      </span>

      <span className="metadata-item">
        <span className="metadata-label">Fallbacks:</span>
        <span className="metadata-value" data-testid="meta-fallback-attempts">
          {metadata.fallback_attempts}
        </span>
      </span>
    </div>
  );
}

export default function App() {
  // The session id is created once, when the component first mounts.
  const [sessionId] = useState(createSessionId);
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [isSending, setIsSending] = useState(false);

  const bottomRef = useRef(null);
  const inputRef = useRef(null);

  // Keep the newest message in view, including the loading indicator.
  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  }, [messages, isSending]);

  const canSend = useMemo(
    () => draft.trim().length > 0 && !isSending,
    [draft, isSending]
  );

  async function handleSend() {
    const message = draft.trim();
    if (!message || isSending) {
      return;
    }

    // Show the user message straight away and clear the input.
    setMessages((current) => [...current, { kind: "user", content: message }]);
    setDraft("");
    setIsSending(true);

    try {
      const data = await sendChatMessage({ sessionId, message, systemPrompt });
      setMessages((current) => [
        ...current,
        {
          kind: "assistant",
          content: data.reply,
          metadata: data.metadata,
        },
      ]);
    } catch (error) {
      // Validation errors, 503s and network failures all land here and are
      // shown in the conversation instead of blanking the screen.
      setMessages((current) => [
        ...current,
        {
          kind: "error",
          content: error.message,
          status: error.status,
        },
      ]);
    } finally {
      setIsSending(false);
      // The textarea is disabled while sending, which drops focus. Give it
      // back so the next message can be typed straight away.
      window.setTimeout(() => inputRef.current?.focus(), 0);
    }
  }

  function handleKeyDown(event) {
    // Enter sends, Shift+Enter inserts a newline.
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1 className="title">LLM Chatbot</h1>
          <p className="subtitle">
            Key failover and response caching &middot; <code>{API_URL}</code>
          </p>
        </div>
        <div className="session">
          <span className="session-label">Session</span>
          <code className="session-id" data-testid="session-id">
            {sessionId}
          </code>
        </div>
      </header>

      <section className="prompt-row">
        <label className="prompt-label" htmlFor="system-prompt">
          System prompt (optional)
        </label>
        <input
          id="system-prompt"
          className="prompt-input"
          data-testid="system-prompt-input"
          type="text"
          placeholder="e.g. You are a terse assistant that answers in one sentence."
          value={systemPrompt}
          onChange={(event) => setSystemPrompt(event.target.value)}
          disabled={isSending}
        />
      </section>

      <main className="messages">
        {messages.length === 0 && !isSending && (
          <p className="empty">
            Ask something to start. Send the same first question in a new
            session and the answer comes back from the cache.
          </p>
        )}

        {messages.map((message, index) => {
          if (message.kind === "user") {
            return (
              <div className="row row-user" key={index}>
                <div className="bubble bubble-user" data-testid="user-message">
                  {message.content}
                </div>
              </div>
            );
          }

          if (message.kind === "assistant") {
            return (
              <div className="row row-assistant" key={index}>
                <div
                  className="bubble bubble-assistant"
                  data-testid="assistant-message"
                >
                  <div className="reply">{message.content}</div>
                  {message.metadata && (
                    <MessageMetadata metadata={message.metadata} />
                  )}
                </div>
              </div>
            );
          }

          return (
            <div className="row row-assistant" key={index}>
              <div className="bubble bubble-error" data-testid="error-message">
                <span className="error-title">
                  {message.status ? `Error ${message.status}` : "Request failed"}
                </span>
                <div className="error-detail">{message.content}</div>
              </div>
            </div>
          );
        })}

        {isSending && (
          <div className="row row-assistant">
            <div className="bubble bubble-assistant loading" data-testid="loading">
              <span className="dot" />
              <span className="dot" />
              <span className="dot" />
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </main>

      <footer className="composer">
        <textarea
          ref={inputRef}
          className="composer-input"
          data-testid="chat-input"
          rows={2}
          autoFocus
          placeholder="Type a message, press Enter to send, Shift+Enter for a new line"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isSending}
        />
        <button
          className="composer-button"
          data-testid="send-button"
          type="button"
          onClick={handleSend}
          disabled={!canSend}
        >
          {isSending ? "Sending" : "Send"}
        </button>
      </footer>
    </div>
  );
}
