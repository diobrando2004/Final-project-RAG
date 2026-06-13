import { useRef, useState, useEffect } from "react";
import { uploadDocuments, deleteDocument, reindexDocument, getDatabases, connectDatabase, disconnectDatabase, reloadDatabase } from "../api";

const ALLOWED_EXTS = [".pdf", ".md", ".csv", ".xlsx", ".xls", ".docx", ".db", ".sqlite"];

export default function Sidebar({ docs, sources, selectedSources, onSourcesChange, selectedSqlDb, selectedSqlTables, onSqlDbChange, onSqlTableToggle, onDocsChanged }) {
  const fileInputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);
  const [pendingFiles, setPendingFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [reindexing, setReindexing] = useState(null);
  const [status, setStatus] = useState("");
  const [filterOpen, setFilterOpen] = useState(false);

  const [databases, setDatabases] = useState([]);
  const [dbName, setDbName] = useState("");
  const [dbConnStr, setDbConnStr] = useState("");
  const [dbConnecting, setDbConnecting] = useState(false);
  const [reloadingDb, setReloadingDb] = useState(null);
  const [expandedDbs, setExpandedDbs] = useState({});

  useEffect(() => {
    getDatabases().then(setDatabases).catch(() => {});
  }, []);

  async function handleConnectDb() {
    if (!dbName.trim() || !dbConnStr.trim()) return;
    setDbConnecting(true);
    setStatus("Connecting to database…");
    try {
      const msg = await connectDatabase(dbName.trim(), dbConnStr.trim());
      setStatus(msg);
      setDbName("");
      setDbConnStr("");
      const updated = await getDatabases();
      setDatabases(updated);
      onDocsChanged();
    } catch (err) {
      setStatus(`Connect failed: ${err.response?.data?.detail || err.message}`);
    } finally {
      setDbConnecting(false);
    }
  }

  async function handleDisconnectDb(name) {
    if (!confirm(`Disconnect "${name}"?`)) return;
    try {
      const msg = await disconnectDatabase(name);
      setStatus(msg);
      const updated = await getDatabases();
      setDatabases(updated);
      onDocsChanged();
    } catch (err) {
      setStatus(`Disconnect failed: ${err.response?.data?.detail || err.message}`);
    }
  }

  async function handleReloadDb(name) {
    setReloadingDb(name);
    setStatus(`Reloading "${name}"…`);
    try {
      const msg = await reloadDatabase(name);
      setStatus(msg);
      const updated = await getDatabases();
      setDatabases(updated);
      onDocsChanged();
    } catch (err) {
      setStatus(`Reload failed: ${err.response?.data?.detail || err.message}`);
    } finally {
      setReloadingDb(null);
    }
  }

  function toggleSource(source) {
    if (source === "Auto/All") { onSourcesChange(["Auto/All"]); return; }
    let next = selectedSources.filter((s) => s !== "Auto/All");
    next = next.includes(source) ? next.filter((s) => s !== source) : [...next, source];
    onSourcesChange(next.length > 0 ? next : ["Auto/All"]);
  }

  function handleFiles(files) {
    const valid = Array.from(files).filter(
      (f) => ALLOWED_EXTS.some((ext) => f.name.toLowerCase().endsWith(ext))
    );
    if (valid.length === 0) { setStatus("Unsupported file type."); return; }
    setPendingFiles(valid);
    setStatus(`${valid.length} file(s) ready to upload.`);
  }

  async function handleUpload() {
    if (pendingFiles.length === 0) return;
    setUploading(true);
    setStatus("Uploading and ingesting…");
    try {
      const result = await uploadDocuments(pendingFiles);
      setStatus(result.message);
      setPendingFiles([]);
      onDocsChanged();
    } catch (err) {
      setStatus(`Upload failed: ${err.response?.data?.detail || err.message}`);
    } finally {
      setUploading(false);
    }
  }

  async function handleReindex(docName) {
    setReindexing(docName);
    setStatus(`Reindexing "${docName}"…`);
    try {
      const msg = await reindexDocument(docName);
      setStatus(msg);
      onDocsChanged();
    } catch (err) {
      setStatus(`Reindex failed: ${err.response?.data?.detail || err.message}`);
    } finally {
      setReindexing(null);
    }
  }

  async function handleDelete(docName) {
    if (!confirm(`Delete "${docName}"? This cannot be undone.`)) return;
    try {
      const msg = await deleteDocument(docName);
      setStatus(msg);
      onDocsChanged();
    } catch (err) {
      setStatus(`Delete failed: ${err.response?.data?.detail || err.message}`);
    }
  }

  const isAuto = selectedSources.includes("Auto/All");
  const filterLabel = isAuto ? "Auto/All"
    : selectedSources.length === 1 ? selectedSources[0]
    : `${selectedSources.length} sources`;

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-logo">RAG Assistant</div>
        <div className="sidebar-tagline">local · private · offline</div>
      </div>

      <div className="sidebar-body">

        {/* Source filter — PDF/CSV only */}
        <div>
          <div className="sidebar-section-title">Source filter</div>
          <button className="filter-toggle" onClick={() => setFilterOpen((o) => !o)}>
            <span className="filter-toggle-label">{filterLabel}</span>
            <span className="filter-toggle-arrow">{filterOpen ? "▲" : "▼"}</span>
          </button>
          {filterOpen && (
            <div className="filter-dropdown">
              {sources.map((src) => (
                <label key={src} className="filter-option">
                  <input type="checkbox" checked={selectedSources.includes(src)} onChange={() => toggleSource(src)} />
                  <span className="filter-option-label">{src}</span>
                </label>
              ))}
            </div>
          )}
        </div>

        {/* Upload */}
        <div>
          <div className="sidebar-section-title">Add documents</div>
          <div
            className={`upload-zone ${dragOver ? "drag-over" : ""}`}
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFiles(e.dataTransfer.files); }}
          >
            <input ref={fileInputRef} type="file" accept=".pdf,.md,.csv,.xlsx,.xls,.docx,.db,.sqlite" multiple onChange={(e) => handleFiles(e.target.files)} />
            <div className="upload-icon">📄</div>
            <div className="upload-label">Drop files or <span>click to browse</span><br />PDF, Markdown, CSV, Excel, Word supported</div>
          </div>
          {pendingFiles.length > 0 && (
            <div style={{ marginTop: 8, fontSize: 11, color: "var(--text-secondary)" }}>
              {pendingFiles.map((f) => <div key={f.name} style={{ fontFamily: "var(--font-mono)" }}>• {f.name}</div>)}
            </div>
          )}
          <div style={{ marginTop: 8 }}>
            <button className="btn btn-primary" onClick={handleUpload} disabled={uploading || pendingFiles.length === 0}>
              {uploading ? "Ingesting…" : "⬆ Upload & Ingest"}
            </button>
          </div>
          {status && <div className="status-msg" style={{ marginTop: 8 }}>{status}</div>}
        </div>

        {/* Document list */}
        <div>
          <div className="sidebar-section-title">Ingested documents ({docs.length})</div>
          {docs.length === 0 ? (
            <div className="doc-empty">No documents yet.<br />Upload files above.</div>
          ) : (
            <div className="doc-list">
              {docs.map((doc) => (
                <div className="doc-item" key={doc.name} title={doc.summary}>
                  <span className="doc-item-type">{doc.file_type === "csv" ? "csv" : "pdf"}</span>
                  <span className="doc-item-name">{doc.name}</span>
                  <button className="doc-action-btn" onClick={() => handleReindex(doc.name)} disabled={reindexing === doc.name} title="Reindex">
                    {reindexing === doc.name ? "…" : "↺"}
                  </button>
                  <button className="doc-delete-btn" onClick={() => handleDelete(doc.name)} disabled={reindexing === doc.name} title="Delete">✕</button>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Live SQL Databases */}
        <div>
          <div className="sidebar-section-title">Live SQL Databases</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <input className="chat-input" style={{ fontSize: 11, padding: "4px 8px", height: "auto" }}
              placeholder="Database name (e.g. my_postgres)" value={dbName}
              onChange={(e) => setDbName(e.target.value)} disabled={dbConnecting} />
            <input className="chat-input" style={{ fontSize: 11, padding: "4px 8px", height: "auto" }}
              placeholder="postgresql://user:pass@host:5432/db" value={dbConnStr}
              onChange={(e) => setDbConnStr(e.target.value)} disabled={dbConnecting} />
            <button className="btn btn-primary" onClick={handleConnectDb}
              disabled={dbConnecting || !dbName.trim() || !dbConnStr.trim()}>
              {dbConnecting ? "Connecting…" : "⚡ Connect"}
            </button>
          </div>

          {databases.length > 0 && (
            <div style={{ marginTop: 8 }}>
              {databases.map((db) => (
                <div key={db.name} style={{ marginBottom: 6 }}>

                  {/* Database row — radio button, one at a time */}
                  <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                    <button
                      style={{ background: "none", border: "none", cursor: "pointer", color: "var(--text-secondary)", fontSize: 10, padding: "0 2px", flexShrink: 0 }}
                      onClick={() => setExpandedDbs(p => ({ ...p, [db.name]: !p[db.name] }))}
                    >
                      {expandedDbs[db.name] ? "▼" : "▶"}
                    </button>
                    <label className="filter-option" style={{ flex: 1, margin: 0 }}>
                      <input
                        type="radio"
                        name="sqlDb"
                        checked={selectedSqlDb === db.name}
                        onChange={() => onSqlDbChange(db.name)}
                      />
                      <span className="filter-option-label" style={{ fontWeight: 600 }}>{db.name}</span>
                      <span style={{ fontSize: 10, color: "var(--text-muted)", marginLeft: 4 }}>{db.dialect}</span>
                    </label>
                    <button className="doc-action-btn" onClick={() => handleReloadDb(db.name)} disabled={reloadingDb === db.name} title="Reload">
                      {reloadingDb === db.name ? "…" : "↺"}
                    </button>
                    <button className="doc-delete-btn" onClick={() => handleDisconnectDb(db.name)} disabled={reloadingDb === db.name} title="Disconnect">✕</button>
                  </div>

                  {/* Table rows — checkboxes, multiple allowed */}
                  {expandedDbs[db.name] && db.tables.map((tbl) => {
                    const key = `${db.name}::${tbl}`;
                    return (
                      <div key={key} style={{ display: "flex", alignItems: "center", paddingLeft: 24, marginTop: 2 }}>
                        <span style={{ color: "var(--text-muted)", marginRight: 4, fontSize: 11 }}>└</span>
                        <label className="filter-option" style={{ flex: 1, margin: 0 }}>
                          <input
                            type="checkbox"
                            checked={(selectedSqlTables || []).includes(key)}
                            onChange={() => onSqlTableToggle(key)}
                          />
                          <span className="filter-option-label">{tbl}</span>
                        </label>
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
          )}
        </div>

      </div>
    </aside>
  );
}