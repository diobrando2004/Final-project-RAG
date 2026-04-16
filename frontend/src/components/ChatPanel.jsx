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

function ThinkingTimer() {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => setElapsed((e) => e + 0.1), 100);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="thinking">
      <div className="thinking-dot" />
      <div className="thinking-dot" />
      <div className="thinking-dot" />
      <span className="thinking-timer">{elapsed.toFixed(1)}s</span>
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
            {msg.role === "assistant" && msg.elapsed != null && (
              <div className="message-elapsed">⏱ {msg.elapsed.toFixed(1)}s</div>
            )}
          </div>
        ))}

        {thinking && (
          <div className="message assistant">
            <div className="message-role">Assistant</div>
            <ThinkingTimer />
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