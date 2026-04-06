import axios from "axios";

// In Docker, requests go through Nginx which proxies /api/ to the backend.
// In local dev, point directly to the backend server.
const BASE = import.meta.env.VITE_API_URL || "/api";

export async function sendChat(query, sources = ["Auto/All"]) {
  const res = await axios.post(`${BASE}/chat`, { query, sources });
  return res.data; // { answer, table }
}

export async function getDocuments() {
  const res = await axios.get(`${BASE}/documents`);
  return res.data;
}

export async function getSources() {
  const res = await axios.get(`${BASE}/sources`);
  return res.data.sources;
}

export async function uploadDocuments(files) {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  const res = await axios.post(`${BASE}/documents/upload`, form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
}

export async function deleteDocument(docName) {
  const res = await axios.delete(`${BASE}/documents/${encodeURIComponent(docName)}`);
  return res.data.status;
}