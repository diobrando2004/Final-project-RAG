import { useEffect, useRef, useState } from "react";

function DataTable({ rows }) {
  if (!rows || rows.length === 0) return null;
  const columns = Object.keys(rows[0]);
  return (
    <div className="data-table-wrapper">
      <table className="data-table">
        <thead>
          <tr>{columns.map((col) => <th key={col}>{col}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {columns.map((col) => (
                <td key={col}>
                  {row[col] === null || row[col] === undefined ? "—" : String(row[col])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ChunkItem({ chunk }) {
  const [expanded, setExpanded] = useState(false);
  const preview = chunk.slice(0, 200);
  const hasMore = chunk.length > 200;

  return (
    <div className="source-chunk">
      <p className="source-chunk-text">
        {expanded ? chunk : preview}
        {hasMore && !expanded && <span className="source-chunk-ellipsis">…</span>}
      </p>
      {hasMore && (
        <button
          className="source-chunk-toggle"
          onClick={() => setExpanded((v) => !v)}
        >
          {expanded ? "Show less" : "Show more"}
        </button>
      )}
    </div>
  );
}

function SourcesPanel({ sources }) {
  const [openDoc, setOpenDoc] = useState(null);

  if (!sources || sources.length === 0) return null;

  return (
    <div className="sources-panel">
      <div className="sources-label">Sources</div>
      <div className="sources-list">
        {sources.map((src) => {
          const isOpen = openDoc === src.name;
          return (
            <div key={src.name} className="source-doc">
              <button
                className={`source-doc-btn ${isOpen ? "open" : ""}`}
                onClick={() => setOpenDoc(isOpen ? null : src.name)}
              >
                <span className="source-doc-icon">📄</span>
                <span className="source-doc-name">{src.name}</span>
                <span className="source-doc-count">{src.chunks.length} chunk{src.chunks.length !== 1 ? "s" : ""}</span>
                <span className="source-doc-arrow">{isOpen ? "▲" : "▼"}</span>
              </button>
              {isOpen && (
                <div className="source-doc-chunks">
                  {src.chunks.map((chunk, i) => (
                    <ChunkItem key={i} chunk={chunk} />
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function ChatPanel({ messages, thinking, selectedSources, onSend }) {
  const bottomRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, thinking]);

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  function submit() {
    const val = textareaRef.current?.value.trim();
    if (!val) return;
    onSend(val);
    textareaRef.current.value = "";
    textareaRef.current.style.height = "42px";
  }

  function handleInput(e) {
    e.target.style.height = "42px";
    e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px";
  }

  const isAuto = !selectedSources || selectedSources.includes("Auto/All");
  const headerLabel = isAuto
    ? "Auto/All"
    : selectedSources.length === 1
    ? selectedSources[0]
    : `${selectedSources.length} sources`;

  return (
    <div className="chat-panel">
      <div className="chat-header">
        <div className="chat-header-title">Chat</div>
        <div className="chat-header-source">{headerLabel}</div>
      </div>

      <div className="chat-messages">
        {messages.length === 0 && !thinking && (
          <div className="chat-empty">
            <div className="chat-empty-icon">📚</div>
            <div className="chat-empty-text">
              Ask anything about your documents.<br />
              Select sources in the sidebar<br />
              or leave it on Auto/All.
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <div className="message-role">
              {msg.role === "user" ? "You" : "Assistant"}
            </div>
            <div className="message-bubble">{msg.content}</div>
            {msg.role === "assistant" && msg.table && (
              <DataTable rows={msg.table} />
            )}
            {msg.role === "assistant" && msg.sources?.length > 0 && (
              <SourcesPanel sources={msg.sources} />
            )}
          </div>
        ))}

        {thinking && (
          <div className="message assistant">
            <div className="message-role">Assistant</div>
            <div className="thinking">
              <div className="thinking-dot" />
              <div className="thinking-dot" />
              <div className="thinking-dot" />
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      <div className="chat-input-bar">
        <textarea
          ref={textareaRef}
          className="chat-input"
          placeholder="Ask a question… (Enter to send, Shift+Enter for new line)"
          onKeyDown={handleKeyDown}
          onInput={handleInput}
          rows={1}
          disabled={thinking}
        />
        <button className="send-btn" onClick={submit} disabled={thinking} title="Send">
          ↑
        </button>
      </div>
    </div>
  );
}