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
  const [selectedSqlSource, setSelectedSqlSource] = useState(null); // e.g. "mydb" or "mydb::table1"

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

  useEffect(() => {
    refreshDocs();
  }, []);

  async function handleSend(query) {
    // If a SQL source is selected, use it exclusively
    const effectiveSources = selectedSqlSource
      ? [selectedSqlSource]
      : selectedSources;

    setMessages((prev) => [...prev, { role: "user", content: query, table: null }]);
    setThinking(true);
    const startTime = performance.now();

    try {
      const res = await sendChat(query, effectiveSources);
      const elapsed = (performance.now() - startTime) / 1000;
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: res.answer,
          table: res.table || null,
          sources: res.sources || [],
          elapsed,
        },
      ]);
    } catch (err) {
      const elapsed = (performance.now() - startTime) / 1000;
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Error: ${err.response?.data?.detail || err.message}`,
          table: null,
          elapsed,
        },
      ]);
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
        onSourcesChange={(s) => { setSelectedSources(s); setSelectedSqlSource(null); }}
        selectedSqlSource={selectedSqlSource}
        onSqlSourceChange={(s) => { setSelectedSqlSource(s); setSelectedSources(["Auto/All"]); }}
        onDocsChanged={refreshDocs}
      />
      <ChatPanel
        messages={messages}
        thinking={thinking}
        selectedSources={selectedSqlSource ? [selectedSqlSource] : selectedSources}
        onSend={handleSend}
      />
    </div>
  );
}