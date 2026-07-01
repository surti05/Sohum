import { useState, useRef } from "react";
import axios from "axios";


function App() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([
    {
      role: "sohum",
      text: "Hello! I am Sohum, your enterprise AI assistant. Upload a document, or ask me anything about your business data, IT standards, or company policies.",
    },
  ]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef(null);

  const ask = async () => {
    if (!question.trim()) return;

    const userMessage = { role: "user", text: question };
    setMessages((prev) => [...prev, userMessage]);
    setQuestion("");
    setLoading(true);

    try {
      const response = await axios.post(`${process.env.REACT_APP_BACKEND_URL}/ask`, {
        question: question,
      });

      const sohumMessage = {
        role: "sohum",
        text: response.data.answer,
        confidence: response.data.confidence,
        model: response.data.model,
        sources: response.data.sources,
      };
      setMessages((prev) => [...prev, sohumMessage]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          role: "sohum",
          text: "Something went wrong. Please check the backend is running.",
        },
      ]);
    }

    setLoading(false);
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      ask();
    }
  };

  const handleFileSelect = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setMessages((prev) => [
      ...prev,
      { role: "user", text: `📄 Uploading: ${file.name}` },
    ]);
    setUploading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(
        `${process.env.REACT_APP_BACKEND_URL}/upload`,
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );

      setMessages((prev) => [
        ...prev,
        {
          role: "sohum",
          text: `Document indexed successfully: **${response.data.filename}**\n\nCreated ${response.data.chunks_created} searchable chunks. You can now ask me questions about this document.`,
        },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          role: "sohum",
          text: "Upload failed. Please check the backend is running and try again.",
        },
      ]);
    }

    setUploading(false);
    e.target.value = "";
  };

  const renderText = (text) => {
    return text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <div style={styles.headerLeft}>
          <div style={styles.logo}>S</div>
          <div>
            <div style={styles.headerTitle}>Sohum</div>
            <div style={styles.headerSub}>Enterprise AI Intelligence</div>
          </div>
        </div>
        <div style={styles.statusDot} title="Online" />
      </div>

      <div style={styles.chatArea}>
        {messages.map((msg, i) => (
          <div
            key={i}
            style={msg.role === "user" ? styles.userRow : styles.sohumRow}
          >
            {msg.role === "sohum" && (
              <div style={styles.avatarSmall}>S</div>
            )}
            <div
              style={
                msg.role === "user" ? styles.userBubble : styles.sohumBubble
              }
            >
              <div
                style={styles.bubbleText}
                dangerouslySetInnerHTML={{ __html: renderText(msg.text) }}
              />

              {msg.sources && msg.sources.length > 0 && (
                <div style={styles.sourcesBox}>
                  <div style={styles.sourcesLabel}>SOURCES</div>
                  {msg.sources.map((src, idx) => (
                    <div key={idx} style={styles.sourceItem}>
                      <span style={styles.sourceIcon}>📄</span>
                      <span style={styles.sourceName}>{src.source}</span>
                      <span style={styles.sourceRelevance}>
                        {Math.round(src.relevance * 100)}% relevant
                      </span>
                    </div>
                  ))}
                </div>
              )}

              {msg.confidence && (
                <div style={styles.meta}>
                  {msg.model} · {msg.confidence} confidence
                </div>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div style={styles.sohumRow}>
            <div style={styles.avatarSmall}>S</div>
            <div style={styles.sohumBubble}>
              <div style={styles.thinking}>Thinking...</div>
            </div>
          </div>
        )}
        {uploading && (
          <div style={styles.sohumRow}>
            <div style={styles.avatarSmall}>S</div>
            <div style={styles.sohumBubble}>
              <div style={styles.thinking}>
                Reading document and creating embeddings...
              </div>
            </div>
          </div>
        )}
      </div>

      <div style={styles.inputArea}>
        <input
          type="file"
          accept=".pdf"
          ref={fileInputRef}
          onChange={handleFileSelect}
          style={{ display: "none" }}
        />
        <button
          style={styles.uploadBtn}
          onClick={() => fileInputRef.current.click()}
          title="Upload a document"
          disabled={uploading}
        >
          📎
        </button>
        <textarea
          style={styles.input}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKey}
          placeholder="Ask anything about your enterprise data or IT standards..."
          rows={2}
        />
        <button
          style={loading ? styles.btnDisabled : styles.btn}
          onClick={ask}
          disabled={loading}
        >
          Ask
        </button>
      </div>
      <div style={styles.hint}>
        📎 Upload a PDF · Press Enter to send · Shift+Enter for new line
      </div>
    </div>
  );
}

const styles = {
  container: {
    maxWidth: "720px",
    margin: "0 auto",
    height: "100vh",
    display: "flex",
    flexDirection: "column",
    fontFamily: "-apple-system, BlinkMacSystemFont, sans-serif",
    background: "#ffffff",
  },
  header: {
    padding: "16px 20px",
    borderBottom: "1px solid #f0f0f0",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
  },
  headerLeft: {
    display: "flex",
    alignItems: "center",
    gap: "12px",
  },
  logo: {
    width: "36px",
    height: "36px",
    borderRadius: "10px",
    background: "#1D9E75",
    color: "#ffffff",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontWeight: "600",
    fontSize: "16px",
  },
  headerTitle: {
    fontWeight: "600",
    fontSize: "15px",
    color: "#111",
  },
  headerSub: {
    fontSize: "11px",
    color: "#888",
    marginTop: "1px",
  },
  statusDot: {
    width: "8px",
    height: "8px",
    borderRadius: "50%",
    background: "#1D9E75",
  },
  chatArea: {
    flex: 1,
    overflowY: "auto",
    padding: "20px",
    display: "flex",
    flexDirection: "column",
    gap: "16px",
  },
  userRow: {
    display: "flex",
    justifyContent: "flex-end",
  },
  sohumRow: {
    display: "flex",
    alignItems: "flex-start",
    gap: "10px",
  },
  avatarSmall: {
    width: "28px",
    height: "28px",
    borderRadius: "8px",
    background: "#1D9E75",
    color: "#ffffff",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontWeight: "600",
    fontSize: "12px",
    flexShrink: 0,
    marginTop: "2px",
  },
  userBubble: {
    background: "#f0f7ff",
    border: "1px solid #d0e8ff",
    borderRadius: "12px 12px 2px 12px",
    padding: "10px 14px",
    maxWidth: "75%",
  },
  sohumBubble: {
    background: "#f6faf8",
    border: "1px solid #d4ede5",
    borderRadius: "12px 12px 12px 2px",
    padding: "10px 14px",
    maxWidth: "80%",
  },
  bubbleText: {
    fontSize: "14px",
    lineHeight: "1.6",
    color: "#111",
    whiteSpace: "pre-wrap",
  },
  sourcesBox: {
    marginTop: "10px",
    paddingTop: "10px",
    borderTop: "1px solid #e0ede8",
  },
  sourcesLabel: {
    fontSize: "10px",
    fontWeight: "600",
    color: "#999",
    letterSpacing: "0.05em",
    marginBottom: "6px",
  },
  sourceItem: {
    display: "flex",
    alignItems: "center",
    gap: "6px",
    fontSize: "12px",
    color: "#555",
    padding: "4px 0",
  },
  sourceIcon: {
    fontSize: "12px",
  },
  sourceName: {
    flex: 1,
    color: "#1D9E75",
    fontWeight: "500",
  },
  sourceRelevance: {
    fontSize: "11px",
    color: "#999",
  },
  meta: {
    fontSize: "11px",
    color: "#888",
    marginTop: "6px",
  },
  thinking: {
    fontSize: "13px",
    color: "#888",
    fontStyle: "italic",
  },
  inputArea: {
    padding: "12px 20px 8px",
    borderTop: "1px solid #f0f0f0",
    display: "flex",
    gap: "10px",
    alignItems: "flex-end",
  },
  uploadBtn: {
    padding: "10px 12px",
    background: "#f6faf8",
    border: "1px solid #d4ede5",
    borderRadius: "10px",
    fontSize: "16px",
    cursor: "pointer",
    flexShrink: 0,
  },
  input: {
    flex: 1,
    padding: "10px 14px",
    border: "1px solid #e0e0e0",
    borderRadius: "10px",
    fontSize: "14px",
    resize: "none",
    outline: "none",
    fontFamily: "inherit",
    lineHeight: "1.5",
    color: "#111",
  },
  btn: {
    padding: "10px 20px",
    background: "#1D9E75",
    color: "#ffffff",
    border: "none",
    borderRadius: "10px",
    fontSize: "14px",
    fontWeight: "500",
    cursor: "pointer",
    whiteSpace: "nowrap",
  },
  btnDisabled: {
    padding: "10px 20px",
    background: "#ccc",
    color: "#ffffff",
    border: "none",
    borderRadius: "10px",
    fontSize: "14px",
    fontWeight: "500",
    cursor: "not-allowed",
    whiteSpace: "nowrap",
  },
  hint: {
    textAlign: "center",
    fontSize: "11px",
    color: "#bbb",
    paddingBottom: "10px",
  },
};

export default App;