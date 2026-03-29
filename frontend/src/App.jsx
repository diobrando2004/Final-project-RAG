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

  const refreshDocs = useCallback(async () => {
    try {
      const [docList, srcList] = await Promise.all([getDocuments(), getSources()]);
      setDocs(docList);
      setSources(srcList);
      // If none of the selected sources exist anymore, reset to Auto/All
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
    setMessages((prev) => [...prev, { role: "user", content: query, table: null }]);
    setThinking(true);

    try {
      const res = await sendChat(query, selectedSources);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: res.answer, table: res.table || null },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Error: ${err.response?.data?.detail || err.message}`,
          table: null,
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
        onSourcesChange={setSelectedSources}
        onDocsChanged={refreshDocs}
      />
      <ChatPanel
        messages={messages}
        thinking={thinking}
        selectedSources={selectedSources}
        onSend={handleSend}
      />
    </div>
  );
}