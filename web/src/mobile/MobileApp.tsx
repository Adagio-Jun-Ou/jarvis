import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { MobileTools } from './MobileTools';
import { MobileDashboard } from './MobileDashboard';
import { MobileSettings } from './MobileSettings';
import { MobileVisual } from './MobileVisual';
import { MobileInsights } from './MobileInsights';

type Tab = 'chat' | 'tools' | 'dashboard' | 'insights' | 'visual' | 'settings';

interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  thinking?: string;
  images?: string[];
  pending?: boolean;
}

interface ToolCall {
  id: string;
  name: string;
  status: 'running' | 'completed' | 'failed';
  result?: string;
  timestamp: number;
}

interface ThinkingState {
  collapsed: boolean;
  content: string;
  header: string;
}

interface ParsedThinking {
  header: string;
  content: string;
}

interface QueuedMessage {
  id: string;
  content: string;
  images?: string[];
  timestamp: number;
}

type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'reconnecting';

const RECONNECT_BASE_DELAY = 1000;
const RECONNECT_MAX_DELAY = 30000;
const MAX_STORED_MESSAGES = 100;

function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

function loadMessages(): Message[] {
  try {
    const saved = localStorage.getItem('jarvis_mobile_messages');
    if (saved) {
      const parsed = JSON.parse(saved);
      // Filter out pending messages and limit count
      return parsed.filter((m: Message) => !m.pending).slice(-MAX_STORED_MESSAGES);
    }
  } catch (e) {
    console.error('Failed to load messages:', e);
  }
  return [];
}

function saveMessages(messages: Message[]): void {
  try {
    // Only save non-pending messages, limit to max
    const toSave = messages.filter(m => !m.pending).slice(-MAX_STORED_MESSAGES);
    localStorage.setItem('jarvis_mobile_messages', JSON.stringify(toSave));
  } catch (e) {
    console.error('Failed to save messages:', e);
  }
}

function parseThinking(text: string): ParsedThinking | null {
  const match = text.match(/<think[^>]*>([\s\S]*?)<\/think>/i);
  if (!match) return null;

  const thinkContent = match[1];
  let header = 'Thought';
  let content = thinkContent;

  const headerMatch = text.match(/<think[^>]*header=["']([^"']+)["']/i);
  if (headerMatch) {
    header = headerMatch[1];
  } else {
    const lines = thinkContent.split('\n');
    const newLines: string[] = [];
    for (const line of lines) {
      if (line.startsWith('#')) {
        header = line.replace(/^#+\s*/, '').trim() || 'Thought';
        continue;
      }
      if (line.toLowerCase().startsWith('header:')) {
        header = line.split(':')[1]?.trim() || 'Thought';
        continue;
      }
      newLines.push(line);
    }
    content = newLines.join('\n').trim();
  }

  return { header, content: content.trim() };
}

function stripThinkingTags(text: string): string {
  return text.replace(/<think[^>]*>[\s\S]*?<\/think>/gi, '').trim();
}

function ThinkingBlock({ 
  content, 
  header, 
  collapsed, 
  onToggle 
}: { 
  content: string; 
  header: string; 
  collapsed: boolean; 
  onToggle: () => void;
}) {
  return (
    <div className="thinking-block">
      <div className="thinking-header" onClick={onToggle}>
        <span className="thinking-header-icon">💭</span>
        <span className="thinking-header-text">{header}</span>
        <button className="thinking-toggle" type="button">
          {collapsed ? '▶' : '▼'}
        </button>
      </div>
      {!collapsed && (
        <div className="thinking-content">
          <pre>{content}</pre>
        </div>
      )}
    </div>
  );
}

function ThinkingIndicator() {
  return (
    <div className="thinking-indicator">
      <div className="thinking-dots">
        <span></span>
        <span></span>
        <span></span>
      </div>
      <span className="thinking-text">Jarvis is thinking...</span>
    </div>
  );
}

function ConnectionIndicator({ status }: { status: ConnectionStatus }) {
  const statusMap = {
    connecting: { color: 'var(--accent-orange)', text: 'Connecting...' },
    connected: { color: 'var(--accent-green)', text: 'Connected' },
    disconnected: { color: 'var(--accent-pink)', text: 'Offline' },
    reconnecting: { color: 'var(--accent-orange)', text: 'Reconnecting...' },
  };
  
  const { color, text } = statusMap[status];
  
  return (
    <div className="connection-indicator" style={{ '--status-color': color } as React.CSSProperties}>
      <div className="status-dot" style={{ background: color }} />
      <span className="status-text">{text}</span>
    </div>
  );
}

function ToolCallPanel({ toolCalls }: { toolCalls: ToolCall[] }) {
  if (toolCalls.length === 0) return null;
  
  const running = toolCalls.filter(t => t.status === 'running');
  const completed = toolCalls.filter(t => t.status === 'completed');
  const failed = toolCalls.filter(t => t.status === 'failed');
  
  return (
    <div className="tool-calls-panel">
      <div className="tool-calls-header">
        <span className="tool-calls-title">🔧 Tool Execution</span>
        <span className="tool-calls-count">{toolCalls.length} tool{toolCalls.length !== 1 ? 's' : ''}</span>
      </div>
      <div className="tool-calls-list">
        {running.map(tool => (
          <div key={tool.id} className="tool-call running">
            <div className="tool-call-header">
              <span className="tool-icon">⚡</span>
              <span className="tool-name">{tool.name}</span>
              <span className="tool-status">running</span>
            </div>
          </div>
        ))}
        {completed.slice(-3).map(tool => (
          <div key={tool.id} className="tool-call completed">
            <div className="tool-call-header">
              <span className="tool-icon">✅</span>
              <span className="tool-name">{tool.name}</span>
              <span className="tool-status">done</span>
            </div>
          </div>
        ))}
        {failed.slice(-2).map(tool => (
          <div key={tool.id} className="tool-call failed">
            <div className="tool-call-header">
              <span className="tool-icon">❌</span>
              <span className="tool-name">{tool.name}</span>
              <span className="tool-status">failed</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function PendingMessageQueue({ messages, onRemove }: { messages: QueuedMessage[]; onRemove: (id: string) => void }) {
  if (messages.length === 0) return null;
  
  return (
    <div className="pending-queue">
      <div className="pending-queue-header">
        <span className="pending-queue-icon">📤</span>
        <span className="pending-queue-text">{messages.length} message{messages.length !== 1 ? 's' : ''} queued</span>
      </div>
      <div className="pending-queue-list">
        {messages.slice(0, 3).map(msg => (
          <div key={msg.id} className="pending-queue-item">
            <span className="pending-preview">{msg.content.slice(0, 30)}...</span>
            <button className="pending-remove" onClick={() => onRemove(msg.id)}>✕</button>
          </div>
        ))}
      </div>
    </div>
  );
}

interface Props {
  token: string;
  onLogout: () => void;
}

export const MobileApp: React.FC<Props> = ({ token, onLogout }) => {
  const [activeTab, setActiveTab] = useState<Tab>('chat');
  const [messages, setMessages] = useState<Message[]>(() => loadMessages());
  const [input, setInput] = useState('');
  const [isThinking, setIsThinking] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('connecting');
  const [toolCalls, setToolCalls] = useState<ToolCall[]>([]);
  const [showThinking, setShowThinking] = useState(true);
  const [thinkingStates, setThinkingStates] = useState<Record<string, ThinkingState>>({});
  const [pendingImages, setPendingImages] = useState<string[]>([]);
  const [showImageOptions, setShowImageOptions] = useState(false);
  const [fullscreenImage, setFullscreenImage] = useState<string | null>(null);
  const [messageQueue, setMessageQueue] = useState<QueuedMessage[]>([]);
  
  const wsRef = useRef<WebSocket | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);
  const reconnectAttempts = useRef(0);
  const reconnectTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isSendingRef = useRef(false);  // Synchronous guard against double-send
  const processedIdsRef = useRef<Set<string>>(new Set());  // Track processed message IDs

  const scrollToBottom = useCallback((smooth = true) => {
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTo({
        top: messagesContainerRef.current.scrollHeight,
        behavior: smooth ? 'smooth' : 'auto'
      });
    }
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, isThinking, scrollToBottom]);

  // Save messages to localStorage whenever they change
  useEffect(() => {
    if (messages.length > 0) {
      saveMessages(messages);
    }
  }, [messages]);

  useEffect(() => {
    const saved = localStorage.getItem('jarvis_mobile_settings');
    if (saved) {
      const parsed = JSON.parse(saved);
      setShowThinking(parsed.showThinking ?? true);
    }
  }, []);

  useEffect(() => {
    let isSubscribed = true;
    let wsInstance: WebSocket | null = null;
    
    const connect = () => {
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${wsProtocol}//${window.location.host}/api/mobile/ws`;
      
      console.log('[WS] Connecting to', wsUrl);
      
      if (reconnectTimeout.current) {
        clearTimeout(reconnectTimeout.current);
        reconnectTimeout.current = null;
      }
      
      if (!isSubscribed) return;
      
      setConnectionStatus('connecting');
      
      const socket = new WebSocket(wsUrl);
      wsInstance = socket;
      wsRef.current = socket;
      
      socket.onopen = () => {
        if (!isSubscribed) {
          socket.close();
          return;
        }
        console.log('[WS] Connection opened, sending auth');
        socket.send(JSON.stringify({ type: 'auth', token }));
        reconnectAttempts.current = 0;
      };
      
      socket.onmessage = (event) => {
        if (!isSubscribed) return;
        try {
          const data = JSON.parse(event.data);
          console.log('[WS] Received:', data.type);
          if (data.type === 'connected') {
            setConnectionStatus('connected');
            processMessageQueue();
          } else if (data.type === 'error') {
            console.error('[WS] Server error:', data.content);
          } else {
            handleMessage(data);
          }
        } catch (e) {
          console.error('Failed to parse message:', e);
        }
      };
      
      socket.onclose = (event) => {
        if (!isSubscribed) return;
        console.log('[WS] Closed:', event.code, event.reason, 'wasClean:', event.wasClean);
        
        // If closed immediately (code 1006) without clean close, likely WebSocket blocked
        // Fall back to HTTP polling
        if (event.code === 1006 || !event.wasClean) {
          console.log('[WS] WebSocket likely blocked, falling back to HTTP polling');
          setConnectionStatus('connected');
          startPolling();
          return;
        }
        
        if (reconnectAttempts.current === 0) {
          setConnectionStatus('connecting');
        } else {
          setConnectionStatus('disconnected');
        }
        
        if (!event.wasClean && reconnectAttempts.current < 10) {
          const delay = Math.min(
            RECONNECT_BASE_DELAY * Math.pow(2, reconnectAttempts.current),
            RECONNECT_MAX_DELAY
          );
          reconnectAttempts.current++;
          
          console.log('[WS] Reconnecting in', delay, 'ms (attempt', reconnectAttempts.current, ')');
          reconnectTimeout.current = setTimeout(() => {
            if (isSubscribed) connect();
          }, delay);
        }
      };
      
      socket.onerror = (error) => {
        console.error('[WS] Error:', error);
      };
    };
    
    // HTTP polling fallback
    let pollingInterval: ReturnType<typeof setInterval> | null = null;
    
    const startPolling = () => {
      if (pollingInterval) return;
      console.log('[Poll] Starting HTTP polling fallback');
      pollingInterval = setInterval(async () => {
        try {
          const resp = await fetch('/api/mobile/poll', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ token }),
          });
          if (resp.ok) {
            const data = await resp.json();
            if (data.messages && data.messages.length > 0) {
              data.messages.forEach((msg: any) => handleMessage(msg));
            }
          }
        } catch (e) {
          console.error('[Poll] Error:', e);
        }
      }, 1000);
    };
    
    connect();
    
    return () => {
      isSubscribed = false;
      if (reconnectTimeout.current) {
        clearTimeout(reconnectTimeout.current);
        reconnectTimeout.current = null;
      }
      if (wsInstance) {
        wsInstance.close();
      }
      if (pollingInterval) {
        clearInterval(pollingInterval);
      }
    };
  }, [token]);

  const processMessageQueue = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    if (messageQueue.length === 0) return;
    
    messageQueue.forEach(msg => {
      wsRef.current?.send(JSON.stringify({
        type: 'message',
        content: msg.content,
        images: msg.images,
        id: msg.id,
      }));
    });
    
    setMessageQueue([]);
  }, [messageQueue]);

  const handleImageSelect = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
      const base64 = e.target?.result as string;
      setPendingImages(prev => [...prev, base64]);
    };
    reader.readAsDataURL(file);
    event.target.value = '';
    setShowImageOptions(false);
  }, []);

  const removePendingImage = useCallback((index: number) => {
    setPendingImages(prev => prev.filter((_, i) => i !== index));
  }, []);

  const removeFromQueue = useCallback((id: string) => {
    setMessageQueue(prev => prev.filter(m => m.id !== id));
  }, []);
  
  const handleMessage = (data: any) => {
    const messageId = data.id;
    
    // Deduplicate done/error messages by ID
    if (messageId && (data.type === 'done' || data.type === 'error')) {
      if (processedIdsRef.current.has(messageId)) {
        console.log('Skipping duplicate message:', messageId);
        return;
      }
      processedIdsRef.current.add(messageId);
      if (processedIdsRef.current.size > 100) {
        const arr = Array.from(processedIdsRef.current);
        processedIdsRef.current = new Set(arr.slice(-50));
      }
    }
    
    switch (data.type) {
      case 'thinking':
        setIsThinking(true);
        break;
        
      case 'response':
        setIsThinking(false);
        setMessages(prev => {
          const lastMsg = prev[prev.length - 1];
          if (lastMsg && lastMsg.role === 'assistant' && lastMsg.content) {
            return [...prev.slice(0, -1), { ...lastMsg, content: lastMsg.content + data.content }];
          }
          return [...prev, {
            id: generateId(),
            role: 'assistant',
            content: data.content,
            timestamp: Date.now(),
          }];
        });
        break;
        
      case 'done':
        setIsThinking(false);
        break;
        
      case 'error':
        setIsThinking(false);
        setMessages(prev => [...prev, {
          id: generateId(),
          role: 'system',
          content: `Error: ${data.content}`,
          timestamp: Date.now(),
        }]);
        break;
    }
  };
  
  const sendMessage = useCallback(async () => {
    // Guard against double-send using synchronous ref
    if (isSendingRef.current) {
      console.log('[MobileApp] Blocked duplicate send');
      return;
    }
    if (!input.trim() && pendingImages.length === 0) return;
    if (isThinking) return;
    
    isSendingRef.current = true;  // Immediately block all subsequent sends
    
    const messageId = generateId();
    console.log('[MobileApp] Sending message:', messageId);
    
    const userMessage: Message = {
      id: messageId,
      role: 'user',
      content: input,
      timestamp: Date.now(),
      images: pendingImages.length > 0 ? [...pendingImages] : undefined,
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setPendingImages([]);
    setIsThinking(true);
    scrollToBottom(false);
    
    // Reset sending guard after a short delay
    setTimeout(() => {
      isSendingRef.current = false;
    }, 500);
    
    if (connectionStatus !== 'connected' || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setMessageQueue(prev => [...prev, {
        id: messageId,
        content: userMessage.content,
        images: userMessage.images,
        timestamp: Date.now(),
      }]);
      setIsThinking(false);
      return;
    }
    
    wsRef.current.send(JSON.stringify({
      type: 'message',
      content: userMessage.content,
      images: userMessage.images,
      id: messageId,
    }));
    
    inputRef.current?.focus();
  }, [input, pendingImages, isThinking, connectionStatus, scrollToBottom]);

  const downloadImage = useCallback(async (imageData: string, filename: string) => {
    try {
      const response = await fetch(imageData);
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (e) {
      console.error('Download failed:', e);
    }
  }, []);

  const toggleThinking = (msgId: string) => {
    setThinkingStates(prev => {
      const current = prev[msgId] || { collapsed: true, content: '', header: 'Thought' };
      return { ...prev, [msgId]: { ...current, collapsed: !current.collapsed } };
    });
  };

  const renderMessage = (msg: Message) => {
    const parsed = parseThinking(msg.content);
    const thinkingState = thinkingStates[msg.id] || { collapsed: true, content: '', header: 'Thought' };
    const displayContent = showThinking ? stripThinkingTags(msg.content) : msg.content;

    return (
      <div key={msg.id} className={`message ${msg.role}${msg.pending ? ' pending' : ''}`}>
        {msg.images && msg.images.length > 0 && (
          <div className="message-images">
            {msg.images.map((img, idx) => (
              <div key={idx} className="message-image-wrapper">
                <img 
                  src={img} 
                  alt={`Attachment ${idx + 1}`} 
                  className="message-image"
                  onClick={() => setFullscreenImage(img)}
                />
                <button 
                  className="image-download-btn"
                  onClick={() => downloadImage(img, `jarvis-image-${Date.now()}-${idx}.png`)}
                  title="Download"
                >
                  ⬇️
                </button>
              </div>
            ))}
          </div>
        )}
        
        {msg.role === 'user' && (
          <div className="message-content">{msg.content}</div>
        )}
        
        {msg.role === 'assistant' && (
          <>
            {parsed && showThinking && (
              <ThinkingBlock
                content={parsed.content}
                header={parsed.header}
                collapsed={thinkingState.collapsed}
                onToggle={() => toggleThinking(msg.id)}
              />
            )}
            {msg.thinking && showThinking && (
              <ThinkingBlock
                content={msg.thinking}
                header="Model Reasoning"
                collapsed={thinkingState.collapsed}
                onToggle={() => toggleThinking(msg.id)}
              />
            )}
            <div className="message-content">
              {renderMarkdown(displayContent)}
            </div>
          </>
        )}
        
        {msg.role === 'system' && (
          <div className="message-content">{msg.content}</div>
        )}
        
        {msg.pending && (
          <div className="message-pending-indicator">
            <span className="pending-dot"></span>
            <span>Waiting to send...</span>
          </div>
        )}
      </div>
    );
  };

  const handleSettingsChange = (settings: { showThinking: boolean }) => {
    setShowThinking(settings.showThinking);
    localStorage.setItem('jarvis_mobile_settings', JSON.stringify(settings));
  };

  const clearConversation = () => {
    setMessages([]);
    setToolCalls([]);
    setMessageQueue([]);
    localStorage.removeItem('jarvis_mobile_messages');
  };

  const recentToolCalls = useMemo(() => toolCalls.slice(-10), [toolCalls]);

  return (
    <div className="mobile-app">
      <header className="mobile-header">
        <div className="header-left">
          <div className="jarvis-indicator">
            <div className={`pulse ${connectionStatus === 'connected' ? 'connected' : ''}`} />
          </div>
          <h1>Jarvis</h1>
        </div>
        <div className="header-right">
          <ConnectionIndicator status={connectionStatus} />
          <button onClick={() => setActiveTab('settings')} className="icon-btn">⚙️</button>
        </div>
      </header>

      <main className="main-content">
        {activeTab === 'chat' && (
          <div className="chat-view">
            <div className="messages-container" ref={messagesContainerRef}>
              {messages.length === 0 && messageQueue.length === 0 && (
                <div className="welcome">
                  <div className="jarvis-logo">
                    <svg viewBox="0 0 100 100">
                      <circle cx="50" cy="50" r="45" fill="none" stroke="currentColor" strokeWidth="2" />
                      <circle cx="35" cy="40" r="5" fill="currentColor" />
                      <circle cx="65" cy="40" r="5" fill="currentColor" />
                      <path d="M35 65 Q50 75 65 65" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
                    </svg>
                  </div>
                  <h2>How can I help you today?</h2>
                  <div className="quick-actions">
                    <button onClick={() => setInput('What can you do?')} className="quick-action">✨ What can you do?</button>
                    <button onClick={() => setInput('Help me write some code')} className="quick-action">💻 Help with code</button>
                    <button onClick={() => setInput('Explain a concept to me')} className="quick-action">📚 Explain something</button>
                  </div>
                </div>
              )}
              
              {messageQueue.length > 0 && (
                <PendingMessageQueue messages={messageQueue} onRemove={removeFromQueue} />
              )}
              
              {messages.map(renderMessage)}
              
              {isThinking && <ThinkingIndicator />}
              
              <div ref={messagesEndRef} />
            </div>
            
            <ToolCallPanel toolCalls={recentToolCalls} />
            
            {pendingImages.length > 0 && (
              <div className="pending-images">
                {pendingImages.map((img, idx) => (
                  <div key={idx} className="pending-image-wrapper">
                    <img src={img} alt={`Pending ${idx + 1}`} className="pending-image" />
                    <button 
                      className="remove-image-btn"
                      onClick={() => removePendingImage(idx)}
                    >
                      ✕
                    </button>
                  </div>
                ))}
              </div>
            )}
            
            <footer className="input-container">
              <input
                type="file"
                ref={fileInputRef}
                accept="image/*"
                onChange={handleImageSelect}
                style={{ display: 'none' }}
              />
              <input
                type="file"
                ref={cameraInputRef}
                accept="image/*"
                capture="environment"
                onChange={handleImageSelect}
                style={{ display: 'none' }}
              />
              
              <button 
                onClick={() => setShowImageOptions(!showImageOptions)} 
                className="attach-btn"
                title="Attach image"
              >
                📎
              </button>
              
              {showImageOptions && (
                <div className="image-options-menu">
                  <button onClick={() => { fileInputRef.current?.click(); setShowImageOptions(false); }}>
                    🖼️ Gallery
                  </button>
                  <button onClick={() => { cameraInputRef.current?.click(); setShowImageOptions(false); }}>
                    📷 Camera
                  </button>
                </div>
              )}
              
              {messages.length > 0 && (
                <button onClick={clearConversation} className="clear-btn" title="Clear">🗑️</button>
              )}
              <div className="input-wrapper">
                <textarea
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }}}
                  placeholder={pendingImages.length > 0 ? "Add a message or send..." : "Ask Jarvis..."}
                  rows={1}
                />
                <button 
                  onClick={sendMessage}
                  disabled={(!input.trim() && pendingImages.length === 0) || isThinking}
                  className="send-btn"
                >
                  <svg viewBox="0 0 24 24" fill="currentColor">
                    <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
                  </svg>
                </button>
              </div>
            </footer>
          </div>
        )}

        {activeTab === 'tools' && <MobileTools toolCalls={toolCalls} />}
        {activeTab === 'dashboard' && <MobileDashboard messages={messages} toolCalls={toolCalls} isThinking={isThinking} />}
        {activeTab === 'insights' && <MobileInsights />}
        {activeTab === 'settings' && (
          <MobileSettings 
            showThinking={showThinking}
            onSettingsChange={handleSettingsChange}
            onLogout={onLogout}
            connectionStatus={connectionStatus}
          />
        )}
        {activeTab === 'visual' && <MobileVisual />}
      </main>

      <nav className="bottom-nav">
        <button className={`nav-item ${activeTab === 'chat' ? 'active' : ''}`} onClick={() => setActiveTab('chat')}>
          <span className="nav-icon">💬</span>
          <span className="nav-label">Chat</span>
        </button>
        <button className={`nav-item ${activeTab === 'tools' ? 'active' : ''}`} onClick={() => setActiveTab('tools')}>
          <span className="nav-icon">🔧</span>
          <span className="nav-label">Tools</span>
          {toolCalls.filter(t => t.status === 'running').length > 0 && (
            <span className="nav-badge">{toolCalls.filter(t => t.status === 'running').length}</span>
          )}
        </button>
        <button className={`nav-item ${activeTab === 'dashboard' ? 'active' : ''}`} onClick={() => setActiveTab('dashboard')}>
          <span className="nav-icon">📊</span>
          <span className="nav-label">Dash</span>
        </button>
        <button className={`nav-item ${activeTab === 'insights' ? 'active' : ''}`} onClick={() => setActiveTab('insights')}>
          <span className="nav-icon">💡</span>
          <span className="nav-label">Insights</span>
        </button>
        <button className={`nav-item ${activeTab === 'visual' ? 'active' : ''}`} onClick={() => setActiveTab('visual')}>
          <span className="nav-icon">🔮</span>
          <span className="nav-label">Visual</span>
        </button>
        <button className={`nav-item ${activeTab === 'settings' ? 'active' : ''}`} onClick={() => setActiveTab('settings')}>
          <span className="nav-icon">⚙️</span>
          <span className="nav-label">Settings</span>
        </button>
      </nav>
      
      {fullscreenImage && (
        <div className="fullscreen-image-overlay" onClick={() => setFullscreenImage(null)}>
          <img src={fullscreenImage} alt="Fullscreen" className="fullscreen-image" />
          <button className="fullscreen-close" onClick={() => setFullscreenImage(null)}>✕</button>
          <button 
            className="fullscreen-download"
            onClick={(e) => { e.stopPropagation(); downloadImage(fullscreenImage, `jarvis-${Date.now()}.png`); }}
          >
            ⬇️ Download
          </button>
        </div>
      )}
    </div>
  );
};

function renderMarkdown(text: string): React.ReactNode {
  if (!text) return null;
  
  const parts = text.split(/(```[\s\S]*?```|`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*|\n)/);
  
  return parts.map((part, index) => {
    if (part.startsWith('```')) {
      const match = part.match(/```(\w*)\n?([\s\S]*?)```/);
      if (match) {
        const [, lang, code] = match;
        return (
          <pre key={index} className={`code-block ${lang || ''}`}>
            {lang && <div className="code-lang">{lang}</div>}
            <code>{code.trim()}</code>
          </pre>
        );
      }
    }
    if (part.startsWith('`') && part.endsWith('`')) {
      return <code key={index} className="inline-code">{part.slice(1, -1)}</code>;
    }
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={index}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith('*') && part.endsWith('*') && part.length > 2) {
      return <em key={index}>{part.slice(1, -1)}</em>;
    }
    if (part === '\n') {
      return <br key={index} />;
    }
    return <span key={index}>{part}</span>;
  });
}

export default MobileApp;