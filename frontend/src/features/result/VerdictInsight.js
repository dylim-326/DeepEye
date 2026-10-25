import React, { useState, useEffect, useRef } from 'react';
import "../../css/result/VerdictInsight.css";

export default function VerdictInsight({ data }) {
  const [input, setInput] = useState("");
  const [chat, setChat] = useState([]);
  const [isTyping, setIsTyping] = useState(false);
  const [questionCount, setQuestionCount] = useState(0);
  const chatWindowRef = useRef(null);

  // 1. Initial setup upon data load
  useEffect(() => {
    if (data && data.llm_report) {
      setChat([{ type: 'ai', text: data.llm_report }]);
    } else {
      setChat([{ type: 'ai', text: "Ask any questions regarding manipulation probability based on the LIME analysis." }]);
    }
    setQuestionCount(0);
  }, [data]);

  // 2. Auto-scroll
  useEffect(() => {
    if (chatWindowRef.current) {
      chatWindowRef.current.scrollTop = chatWindowRef.current.scrollHeight;
    }
  }, [chat]);

  // 3. Message sending handler
  const sendMsg = async (e) => {
    e.preventDefault();
    
    if (!input || isTyping || questionCount >= 3) return;

    const userMessage = input;
    setInput("");
    
    const newChat = [...chat, { type: 'user', text: userMessage }];
    setChat(newChat);
    setIsTyping(true);

    try {
      const response = await fetch('http://127.0.0.1:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          message: userMessage,
          context: data,           
          lime_url: data.lime_url   
        }),
      });

      if (!response.ok) throw new Error("Server response error");

      const resData = await response.json();

      setChat((prev) => [...prev, { type: 'ai', text: resData.reply }]);
      setQuestionCount(prev => prev + 1);

    } catch (error) {
      console.error("Chat error:", error);
      setChat((prev) => [...prev, { type: 'ai', text: "An error occurred while retrieving the response." }]);
    } finally {
      setIsTyping(false);
    }
  };

  if (!data || !data.result) return null;

  const isFake = data.result === "FAKE";
  const verdictColor = isFake ? "#FF4D4D" : "#00FFC8";
  const rawScore = (data.final_score || 0) * 100;
  const displayScore = isFake ? rawScore : (100 - rawScore);

  const isLimitReached = questionCount >= 3;

  return (
    <section className="VerdictInsight">
      <div className="verdict-summary">
        <div className="main-verdict-area" style={{ display: 'flex', alignItems: 'baseline', gap: '12px' }}>
          <h2 style={{ color: verdictColor, fontWeight: "900", fontSize: "3rem", margin: 0 }}>
            {data.result}
          </h2>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '6px' }}>
            <span className="main-score" style={{ fontSize: '1.8rem', fontWeight: '700', color: '#fff' }}>
              {displayScore.toFixed(1)}%
            </span>
            <span style={{ fontSize: '0.85rem', color: 'rgba(255,255,255,0.5)', fontWeight: '400' }}>
              (Confidence)
            </span>
          </div>
        </div>

        <div className="ensemble-info">
          AI determined this media as <strong>{data.result}</strong> with {displayScore.toFixed(1)}% confidence.
          
          <div style={{ marginTop: '8px', fontSize: '0.8rem', lineHeight: '1.4' }}>
            <div style={{ opacity: 0.5 }}>
              • <strong>Decision Engine:</strong> Precision analysis via EfficientNet-B0
            </div>
            <div style={{ opacity: 0.5 }}>
              • <strong>Visual Verification:</strong> Cross-validation via MesoNet-based LIME
            </div>
            
            <div style={{ 
              marginTop: '6px', 
              color: isLimitReached ? '#FF4D4D' : '#00FFC8', 
              fontWeight: '600',
              letterSpacing: '0.5px'
            }}>
              ● Remaining Queries: {3 - questionCount} / 3 left
            </div>
          </div>
        </div>
      </div>

      <div className="llm-chat-section">
        <div className="chat-window" ref={chatWindowRef}>
          {chat.map((m, i) => (
            <div key={i} className={`bubble ${m.type}`}>
              {m.text}
            </div>
          ))}
          {isTyping && <div className="bubble ai typing">AI is generating response...</div>}
          {isLimitReached && <div className="bubble ai" style={{ opacity: 0.5 }}>🚫 You have reached the maximum limit of 3 questions.</div>}
        </div>
        
        <form onSubmit={sendMsg} className="input-group" style={{ opacity: isLimitReached ? 0.5 : 1 }}>
          <input 
            value={input} 
            onChange={(e) => setInput(e.target.value)} 
            placeholder={
              isLimitReached 
                ? "Query limit reached" 
                : isTyping ? "AI is responding..." : "Ask AI a question..."
            } 
            autoComplete="off"
            disabled={isTyping || isLimitReached}
          />
          <button type="submit" disabled={isTyping || isLimitReached}>Ask</button>
        </form>
      </div>
    </section>
  );
}