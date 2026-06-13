import { useState, useEffect, useCallback } from "react";
import { sendChat, getDocuments, getSources } from "./api";
import Sidebar from "./components/Sidebar";
import ChatPanel from "./components/ChatPanel";

export default function App() {
  const [messages, setMessages] = useState([]);
  const [thinking, setThinking] = useState(false);
  const [docs, setDocs] = useState([]);
  const [sources, setSources] = useState(["Auto/All"]);
  const [selectedSources, setSelectedSources] = useState(["Auto/All"]);
  const [selectedSqlDb, setSelectedSqlDb] = useState(null);       // one DB at a time (radio)
  const [selectedSqlTables, setSelectedSqlTables] = useState([]); // multiple tables (checkboxes)

  const refreshDocs = useCallback(async () => {
    try {
      const [docList, srcList] = await Promise.all([getDocuments(), getSources()]);
      setDocs(docList);
      setSources(srcList);
      setSelectedSources((prev) => {
        const valid = prev.filter((s) => s === "Auto/All" || srcList.includes(s));
        return valid.length > 0 ? valid : ["Auto/All"];
      });
    } catch (err) {
      console.error("Failed to load documents/sources:", err);
    }
  }, []);

  useEffect(() => { refreshDocs(); }, []);

  function handleSqlDbChange(dbName) {
    if (selectedSqlDb === dbName) {
      setSelectedSqlDb(null);
      setSelectedSqlTables([]);
    } else {
      setSelectedSqlDb(dbName);
      setSelectedSqlTables([]);
      setSelectedSources(["Auto/All"]);
    }
  }

  function handleSqlTableToggle(tableKey) {
    setSelectedSqlTables(prev =>
      prev.includes(tableKey)
        ? prev.filter(t => t !== tableKey)
        : [...prev, tableKey]
    );
    setSelectedSources(["Auto/All"]);
  }

  const effectiveSources = selectedSqlTables.length > 0
    ? selectedSqlTables
    : selectedSqlDb
    ? [selectedSqlDb]
    : selectedSources;

  async function handleSend(query) {
    setMessages((prev) => [...prev, { role: "user", content: query, table: null }]);
    setThinking(true);
    const startTime = performance.now();
    try {
      const res = await sendChat(query, effectiveSources);
      const elapsed = (performance.now() - startTime) / 1000;
      setMessages((prev) => [...prev, {
        role: "assistant",
        content: res.answer,
        table: res.table || null,
        sources: res.sources || [],
        elapsed,
      }]);
    } catch (err) {
      const elapsed = (performance.now() - startTime) / 1000;
      setMessages((prev) => [...prev, {
        role: "assistant",
        content: `Error: ${err.response?.data?.detail || err.message}`,
        table: null,
        elapsed,
      }]);
    } finally {
      setThinking(false);
    }
  }

  return (
    <div className="app-shell">
      <Sidebar
        docs={docs}
        sources={sources}
        selectedSources={selectedSources}
        onSourcesChange={(s) => { setSelectedSources(s); setSelectedSqlDb(null); setSelectedSqlTables([]); }}
        selectedSqlDb={selectedSqlDb}
        selectedSqlTables={selectedSqlTables}
        onSqlDbChange={handleSqlDbChange}
        onSqlTableToggle={handleSqlTableToggle}
        onDocsChanged={refreshDocs}
      />
      <ChatPanel
        messages={messages}
        thinking={thinking}
        selectedSources={effectiveSources}
        onSend={handleSend}
      />
    </div>
  );
}