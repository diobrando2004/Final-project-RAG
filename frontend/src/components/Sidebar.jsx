import { useRef, useState } from "react";
import { uploadDocuments, deleteDocument } from "../api";

const ALLOWED_EXTS = [".pdf", ".md", ".csv", ".xlsx", ".xls"];

export default function Sidebar({ docs, sources, selectedSources, onSourcesChange, onDocsChanged }) {
  const fileInputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);
  const [pendingFiles, setPendingFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState("");
  const [filterOpen, setFilterOpen] = useState(false);

  // ── Source filter logic ─────────────────────────────────────────────
  function toggleSource(source) {
    if (source === "Auto/All") {
      onSourcesChange(["Auto/All"]);
      return;
    }
    // Toggling a specific source
    let next = selectedSources.filter((s) => s !== "Auto/All");
    if (next.includes(source)) {
      next = next.filter((s) => s !== source);
    } else {
      next = [...next, source];
    }
    // If nothing selected, fall back to Auto/All
    onSourcesChange(next.length > 0 ? next : ["Auto/All"]);
  }

  function isSelected(source) {
    return selectedSources.includes(source);
  }

  const isAuto = selectedSources.includes("Auto/All");
  const filterLabel = isAuto
    ? "Auto/All"
    : selectedSources.length === 1
    ? selectedSources[0]
    : `${selectedSources.length} sources`;

  // ── File selection ──────────────────────────────────────────────────
  function handleFiles(files) {
    const valid = Array.from(files).filter(
      (f) => ALLOWED_EXTS.some((ext) => f.name.toLowerCase().endsWith(ext))
    );
    if (valid.length === 0) {
      setStatus("Only PDF, Markdown, CSV and Excel files are supported.");
      return;
    }
    setPendingFiles(valid);
    setStatus(`${valid.length} file(s) ready to upload.`);
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragOver(false);
    handleFiles(e.dataTransfer.files);
  }

  // ── Upload ──────────────────────────────────────────────────────────
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

  // ── Delete ──────────────────────────────────────────────────────────
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

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-logo">RAG Assistant</div>
        <div className="sidebar-tagline">local · private · offline</div>
      </div>

      <div className="sidebar-body">

        {/* Source filter */}
        <div>
          <div className="sidebar-section-title">Source filter</div>
          <button
            className="filter-toggle"
            onClick={() => setFilterOpen((o) => !o)}
          >
            <span className="filter-toggle-label">{filterLabel}</span>
            <span className="filter-toggle-arrow">{filterOpen ? "▲" : "▼"}</span>
          </button>

          {filterOpen && (
            <div className="filter-dropdown">
              {sources.map((src) => (
                <label key={src} className="filter-option">
                  <input
                    type="checkbox"
                    checked={isSelected(src)}
                    onChange={() => toggleSource(src)}
                  />
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
            onDrop={handleDrop}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.md,.csv,.xlsx,.xls"
              multiple
              onChange={(e) => handleFiles(e.target.files)}
            />
            <div className="upload-icon">📄</div>
            <div className="upload-label">
              Drop files or <span>click to browse</span>
              <br />PDF, Markdown, CSV and Excel supported
            </div>
          </div>

          {pendingFiles.length > 0 && (
            <div style={{ marginTop: 8, fontSize: 11, color: "var(--text-secondary)" }}>
              {pendingFiles.map((f) => (
                <div key={f.name} style={{ fontFamily: "var(--font-mono)" }}>• {f.name}</div>
              ))}
            </div>
          )}

          <div style={{ marginTop: 8 }}>
            <button
              className="btn btn-primary"
              onClick={handleUpload}
              disabled={uploading || pendingFiles.length === 0}
            >
              {uploading ? "Ingesting…" : "⬆ Upload & Ingest"}
            </button>
          </div>

          {status && (
            <div className="status-msg" style={{ marginTop: 8 }}>{status}</div>
          )}
        </div>

        {/* Document list */}
        <div>
          <div className="sidebar-section-title">
            Ingested documents ({docs.length})
          </div>
          {docs.length === 0 ? (
            <div className="doc-empty">No documents yet.<br />Upload files above.</div>
          ) : (
            <div className="doc-list">
              {docs.map((doc) => (
                <div className="doc-item" key={doc.name} title={doc.summary}>
                  <span className="doc-item-type">{doc.file_type === "csv" ? "csv" : "pdf"}</span>
                  <span className="doc-item-name">{doc.name}</span>
                  <button
                    className="doc-delete-btn"
                    onClick={() => handleDelete(doc.name)}
                    title="Delete document"
                  >
                    ✕
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

      </div>
    </aside>
  );
}