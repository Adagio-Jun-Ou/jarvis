import { useState, useEffect, useCallback, useRef } from 'react';

export type Message = {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  thinking?: string;
  model?: 'primary' | 'backend' | 'collaborative';
  timestamp: number;
};

export type ToolCall = {
  id: string;
  name: string;
  status: 'running' | 'completed' | 'failed';
  result?: string;
  timestamp: number;
};

export type WSStatus = 'connecting' | 'connected' | 'disconnected' | 'reconnecting';

export type ConnectionState = {
  status: WSStatus;
  reconnectAttempt: number;
  lastReconnectTime: number | null;
  queueLength: number;
};

export type Conversation = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
};

const RECONNECT_BASE_DELAY = 1000;
const RECCONNECT_MAX_DELAY = 30000;
const PING_INTERVAL = 30000;
const PING_TIMEOUT = 10000;
const CONNECTION_TIMEOUT = 10000;
const MAX_MESSAGES = 100;
const STORAGE_KEY = 'jarvis_messages';
const CONVERSATION_KEY = 'jarvis_current_conversation';

function loadMessages(): Message[] {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored);
      if (Array.isArray(parsed)) {
        return parsed.slice(-MAX_MESSAGES);
      }
    }
  } catch {
    localStorage.removeItem(STORAGE_KEY);
  }
  return [];
}

function saveMessages(messages: Message[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(messages.slice(-MAX_MESSAGES)));
  } catch {
  }
}

interface QueuedMessage {
  id: string;
  type: string;
  content: string;
  timestamp: number;
  retries: number;
}

export type ModeQuestion = {
  message: string;
  estimatedSteps: number;
  toolsNeeded: string[];
  reason: string;
  userInput: string;
};

export type ModeInfo = {
  mode: string;
  estimatedSteps?: number;
  toolsDetected?: string[];
  message?: string;
};

export function useWebSocket(url: string) {
  const [status, setStatus] = useState<WSStatus>('connecting');
  const [connectionState, setConnectionState] = useState<ConnectionState>({
    status: 'connecting',
    reconnectAttempt: 0,
    lastReconnectTime: null,
    queueLength: 0,
  });
  const [messages, setMessages] = useState<Message[]>(() => loadMessages());
  const [isThinking, setIsThinking] = useState(false);
  const [toolCalls, setToolCalls] = useState<ToolCall[]>([]);
  const [modeQuestion, setModeQuestion] = useState<ModeQuestion | null>(null);
  const [modeInfo, setModeInfo] = useState<ModeInfo | null>(null);
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(() => {
    try {
      return localStorage.getItem(CONVERSATION_KEY) || null;
    } catch {
      return null;
    }
  });
  const wsRef = useRef<WebSocket | null>(null);
  const currentMessageRef = useRef<Message | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const pingTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const connectionTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(false);
  const reconnectAttemptsRef = useRef(0);
  const isConnectingRef = useRef(false);
  const messageIdCounterRef = useRef(0);
  const isSendingRef = useRef(false);
  const messageQueueRef = useRef<QueuedMessage[]>([]);

  // Calculate reconnect delay with exponential backoff
  const getReconnectDelay = useCallback(() => {
    const delay = Math.min(
      RECONNECT_BASE_DELAY * Math.pow(2, reconnectAttemptsRef.current),
      RECCONNECT_MAX_DELAY
    );
    return delay + Math.random() * 1000; // Add jitter
  }, []);

  const clearAllTimers = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = null;
    }
    if (pingTimeoutRef.current) {
      clearTimeout(pingTimeoutRef.current);
      pingTimeoutRef.current = null;
    }
    if (connectionTimeoutRef.current) {
      clearTimeout(connectionTimeoutRef.current);
      connectionTimeoutRef.current = null;
    }
  }, []);

  const addMessage = useCallback((message: Message) => {
    setMessages(prev => {
      const updated = [...prev, message];
      saveMessages(updated);
      return updated;
    });
  }, []);

  const clearMessages = useCallback(() => {
    setMessages([]);
    setToolCalls([]);
    currentMessageRef.current = null;
    localStorage.removeItem(STORAGE_KEY);
  }, []);

  const clearToolCalls = useCallback(() => {
    setToolCalls([]);
  }, []);

  const generateId = useCallback(() => {
    messageIdCounterRef.current += 1;
    return `msg-${Date.now()}-${messageIdCounterRef.current}`;
  }, []);

  // Process queued messages when connected
  const processQueue = useCallback(() => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) return;
    
    while (messageQueueRef.current.length > 0) {
      const msg = messageQueueRef.current.shift();
      if (msg) {
        try {
          wsRef.current.send(JSON.stringify({
            type: msg.type,
            content: msg.content,
          }));
        } catch (err) {
          console.error('[WS] Failed to send queued message:', err);
          // Re-queue if it fails and hasn't exceeded retries
          if (msg.retries < 3) {
            messageQueueRef.current.unshift({ ...msg, retries: msg.retries + 1 });
          }
        }
      }
    }
    
    setConnectionState(prev => ({ ...prev, queueLength: messageQueueRef.current.length }));
  }, []);

  const handleMessage = useCallback((data: { type: string; [key: string]: unknown }) => {
    switch (data.type) {
      case 'connected':
      case 'status':
        if (data.status === 'ready' || data.type === 'connected') {
          reconnectAttemptsRef.current = 0;
          setStatus('connected');
          setConnectionState({
            status: 'connected',
            reconnectAttempt: 0,
            lastReconnectTime: null,
            queueLength: messageQueueRef.current.length,
          });
          // Process any queued messages
          processQueue();
        }
        if (data.status === 'thinking') {
          setIsThinking(true);
        }
        break;

      case 'pong':
        // Clear ping timeout
        if (pingTimeoutRef.current) {
          clearTimeout(pingTimeoutRef.current);
          pingTimeoutRef.current = null;
        }
        break;

      case 'cleared':
        setMessages([]);
        setToolCalls([]);
        currentMessageRef.current = null;
        localStorage.removeItem(STORAGE_KEY);
        break;

      case 'token':
        setIsThinking(true);
        if (!currentMessageRef.current) {
          currentMessageRef.current = {
            id: generateId(),
            role: 'assistant',
            content: '',
            timestamp: Date.now(),
          };
          addMessage(currentMessageRef.current);
        }
        if (currentMessageRef.current && typeof data.content === 'string') {
          let token = data.content;
          
          // Parse model indicator from null marker
          const modelMatch = token.match(/\x00MODEL:(primary|backend|collaborative)/);
          if (modelMatch) {
            currentMessageRef.current.model = modelMatch[1] as 'primary' | 'backend' | 'collaborative';
            token = token.replace(/\x00MODEL:\w+\n?/g, '');
          }
          
          if (token) {
            currentMessageRef.current.content += token;
          }
          
          if (data.model && !currentMessageRef.current.model) {
            currentMessageRef.current.model = data.model as 'primary' | 'backend' | 'collaborative';
          }
          setMessages(prev => [...prev]);
        }
        break;

      case 'thinking':
        setIsThinking(true);
        if (!currentMessageRef.current) {
          currentMessageRef.current = {
            id: generateId(),
            role: 'assistant',
            content: '',
            thinking: '',
            timestamp: Date.now(),
          };
          addMessage(currentMessageRef.current);
        }
        if (currentMessageRef.current && typeof data.content === 'string') {
          if (!currentMessageRef.current.thinking) {
            currentMessageRef.current.thinking = '';
          }
          currentMessageRef.current.thinking += data.content;
          setMessages(prev => [...prev]);
        }
        break;

      case 'step_complete':
      case 'loop_complete':
        break;

      case 'mode_question':
        setIsThinking(false);
        setModeQuestion({
          message: data.message as string || 'Should I enter autonomous mode?',
          estimatedSteps: data.estimated_steps as number || 0,
          toolsNeeded: data.tools_needed as string[] || [],
          reason: data.reason as string || '',
          userInput: data.user_input as string || '',
        });
        break;

      case 'mode_info':
        setModeInfo({
          mode: data.mode as string || 'chat',
          estimatedSteps: data.estimated_steps as number | undefined,
          toolsDetected: data.tools_detected as string[] | undefined,
          message: data.message as string | undefined,
        });
        break;

      case 'done':
        setIsThinking(false);
        if (currentMessageRef.current) {
          setMessages(prev => {
            saveMessages(prev);
            return prev;
          });
        }
        currentMessageRef.current = null;
        setToolCalls(prev => prev.map(tc => 
          tc.status === 'running' ? { ...tc, status: 'completed' as const, result: 'completed' } : tc
        ));
        break;

      case 'error':
        setIsThinking(false);
        addMessage({
          id: generateId(),
          role: 'system',
          content: `Error: ${data.content}`,
          timestamp: Date.now(),
        });
        currentMessageRef.current = null;
        break;

      case 'tool_start':
        if (typeof data.name === 'string') {
          setToolCalls(prev => [...prev, {
            id: (data.id as string) || `tool-${Date.now()}`,
            name: data.name as string,
            status: 'running',
            timestamp: Date.now(),
          }]);
        }
        break;

      case 'tool_result':
        if (typeof data.name === 'string') {
          setToolCalls(prev => prev.map(tc => 
            tc.name === data.name 
              ? { 
                  ...tc, 
                  status: data.success ? 'completed' : 'failed',
                  result: typeof data.result === 'string' ? data.result : undefined,
                }
              : tc
          ));
        }
        break;

      case 'conversation_created':
        {
          const convData = data.conversation as { id?: string } | undefined;
          if (convData && typeof convData.id === 'string') {
            setCurrentConversationId(convData.id);
            localStorage.setItem(CONVERSATION_KEY, convData.id);
            setMessages([]);
            localStorage.removeItem(STORAGE_KEY);
          }
        }
        break;

      case 'conversation_loaded':
        {
          const loadedConv = data.conversation as { id?: string; messages?: Array<{ id?: string; role: string; content: string; thinking?: string; timestamp?: number }> } | undefined;
          if (loadedConv && typeof loadedConv.id === 'string' && Array.isArray(loadedConv.messages)) {
            setCurrentConversationId(loadedConv.id);
            localStorage.setItem(CONVERSATION_KEY, loadedConv.id);
            const loadedMessages = loadedConv.messages.map((m) => ({
              id: m.id || `loaded-${Date.now()}-${Math.random()}`,
              role: m.role as 'user' | 'assistant' | 'system',
              content: m.content,
              thinking: m.thinking,
              timestamp: m.timestamp || Date.now(),
            }));
            setMessages(loadedMessages);
            saveMessages(loadedMessages);
          }
        }
        break;
    }
  }, [addMessage, generateId, processQueue]);

  const startPingInterval = useCallback(() => {
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
    }
    pingIntervalRef.current = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
        
        // Set ping timeout
        pingTimeoutRef.current = setTimeout(() => {
          console.log('[WS] Ping timeout, reconnecting...');
          wsRef.current?.close();
        }, PING_TIMEOUT);
      }
    }, PING_INTERVAL);
  }, []);

  const connect = useCallback(() => {
    if (isConnectingRef.current) return;
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    if (wsRef.current?.readyState === WebSocket.CONNECTING) return;
    if (!mountedRef.current) return;

    isConnectingRef.current = true;
    clearAllTimers();
    
    if (reconnectAttemptsRef.current > 0) {
      setStatus('reconnecting');
      setConnectionState(prev => ({
        ...prev,
        status: 'reconnecting',
        reconnectAttempt: reconnectAttemptsRef.current,
        lastReconnectTime: Date.now(),
      }));
    } else {
      setStatus('connecting');
      setConnectionState(prev => ({
        ...prev,
        status: 'connecting',
        reconnectAttempt: 0,
      }));
    }

    const ws = new WebSocket(url);
    wsRef.current = ws;

    connectionTimeoutRef.current = setTimeout(() => {
      if (ws.readyState !== WebSocket.OPEN) {
        ws.close();
      }
    }, CONNECTION_TIMEOUT);

    ws.onopen = () => {
      if (connectionTimeoutRef.current) {
        clearTimeout(connectionTimeoutRef.current);
        connectionTimeoutRef.current = null;
      }
      isConnectingRef.current = false;
      reconnectAttemptsRef.current = 0;
      setStatus('connected');
      setConnectionState({
        status: 'connected',
        reconnectAttempt: 0,
        lastReconnectTime: null,
        queueLength: messageQueueRef.current.length,
      });
      startPingInterval();
      processQueue();
    };

    ws.onclose = (event) => {
      isConnectingRef.current = false;
      setIsThinking(false);
      clearAllTimers();
      
      if (wsRef.current === ws) {
        wsRef.current = null;
      }
      
      if (mountedRef.current && event.code !== 1000) {
        setStatus('disconnected');
        setConnectionState(prev => ({
          ...prev,
          status: 'disconnected',
        }));
        
        const delay = getReconnectDelay();
        reconnectAttemptsRef.current++;
        
        setConnectionState(prev => ({
          ...prev,
          reconnectAttempt: reconnectAttemptsRef.current,
        }));
        
        reconnectTimeoutRef.current = setTimeout(() => {
          if (mountedRef.current) {
            connect();
          }
        }, delay);
      }
    };

    ws.onerror = (error) => {
      isConnectingRef.current = false;
      console.error('WebSocket error:', error);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleMessage(data);
      } catch (e) {
        console.warn('Failed to parse WebSocket message:', e, event.data);
      }
    };

  }, [url, handleMessage, clearAllTimers, startPingInterval, processQueue, getReconnectDelay]);

  const disconnect = useCallback(() => {
    clearAllTimers();
    
    if (wsRef.current) {
      wsRef.current.close(1000, 'Manual disconnect');
      wsRef.current = null;
    }
    
    setStatus('disconnected');
    setConnectionState({
      status: 'disconnected',
      reconnectAttempt: 0,
      lastReconnectTime: null,
      queueLength: messageQueueRef.current.length,
    });
    reconnectAttemptsRef.current = 0;
  }, [clearAllTimers]);

  const sendMessage = useCallback((content: string) => {
    // Guard against double-send
    if (isSendingRef.current) {
      console.log('[useWebSocket] Blocked duplicate send');
      return;
    }
    if (!content.trim()) return;

    isSendingRef.current = true;
    setTimeout(() => { isSendingRef.current = false; }, 500);

    const userMessage: Message = {
      id: generateId(),
      role: 'user',
      content,
      timestamp: Date.now(),
    };

    addMessage(userMessage);
    currentMessageRef.current = null;
    setIsThinking(true);
    setToolCalls([]);

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'chat', content }));
    } else {
      // Queue message if not connected
      messageQueueRef.current.push({
        id: generateId(),
        type: 'chat',
        content,
        timestamp: Date.now(),
        retries: 0,
      });
      setConnectionState(prev => ({
        ...prev,
        queueLength: messageQueueRef.current.length,
      }));
      
      sendViaREST(content, handleMessage);
      setStatus('disconnected');
    }
  }, [addMessage, handleMessage, generateId]);

  const sendAgentLoop = useCallback((content: string, useBackend: boolean = false, maxSteps: number = 20) => {
    // Guard against double-send
    if (isSendingRef.current) {
      console.log('[useWebSocket] Blocked duplicate agent_loop');
      return;
    }
    if (!content.trim()) return;

    isSendingRef.current = true;
    setTimeout(() => { isSendingRef.current = false; }, 500);

    const userMessage: Message = {
      id: generateId(),
      role: 'user',
      content,
      timestamp: Date.now(),
    };

    addMessage(userMessage);
    currentMessageRef.current = null;
    setIsThinking(true);
    setToolCalls([]);

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ 
        type: 'agent_loop', 
        content,
        use_backend: useBackend,
        max_steps: maxSteps,
      }));
    } else {
      // Queue message if not connected
      messageQueueRef.current.push({
        id: generateId(),
        type: 'agent_loop',
        content,
        timestamp: Date.now(),
        retries: 0,
      });
      setConnectionState(prev => ({
        ...prev,
        queueLength: messageQueueRef.current.length,
      }));
      
      console.warn('WebSocket not connected, falling back to regular chat');
      wsRef.current?.send(JSON.stringify({ type: 'chat', content }));
    }
  }, [addMessage, generateId]);

  const sendClear = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'clear' }));
    }
    clearMessages();
    fetch('/api/clear', { method: 'POST' }).catch(() => {});
  }, [clearMessages]);

  const newConversation = useCallback(async () => {
    try {
      const response = await fetch('/api/conversations/new', { method: 'POST' });
      const data = await response.json();
      if (data.success && (data.conversation_id || data.conversation?.id)) {
        const convId = data.conversation_id || data.conversation.id;
        setCurrentConversationId(convId);
        localStorage.setItem(CONVERSATION_KEY, convId);
        setMessages([]);
        localStorage.removeItem(STORAGE_KEY);
        setToolCalls([]);
        return convId;
      }
    } catch (error) {
      console.error('Failed to create conversation:', error);
    }
    return null;
  }, []);

  const loadConversation = useCallback(async (conversationId: string) => {
    try {
      const response = await fetch(`/api/conversations/${conversationId}`);
      const data = await response.json();
      if (data.success && data.conversation) {
        setCurrentConversationId(conversationId);
        localStorage.setItem(CONVERSATION_KEY, conversationId);
        const loadedMessages = (data.conversation.messages || []).map((m: { id?: string; role: string; content: string; thinking?: string; timestamp?: number }) => ({
          id: m.id || `loaded-${Date.now()}-${Math.random()}`,
          role: m.role as 'user' | 'assistant' | 'system',
          content: m.content,
          thinking: m.thinking,
          timestamp: m.timestamp || Date.now(),
        }));
        setMessages(loadedMessages);
        saveMessages(loadedMessages);
        setToolCalls([]);
      }
    } catch (error) {
      console.error('Failed to load conversation:', error);
    }
  }, []);

  const deleteConversation = useCallback(async (conversationId: string) => {
    try {
      await fetch(`/api/conversations/${conversationId}`, { method: 'DELETE' });
      if (currentConversationId === conversationId) {
        setCurrentConversationId(null);
        localStorage.removeItem(CONVERSATION_KEY);
        setMessages([]);
        localStorage.removeItem(STORAGE_KEY);
      }
    } catch (error) {
      console.error('Failed to delete conversation:', error);
    }
  }, [currentConversationId]);

  const saveCurrentConversation = useCallback(async () => {
    if (!currentConversationId || messages.length === 0) return;
    try {
      await fetch(`/api/conversations/${currentConversationId}/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages }),
      });
    } catch (error) {
      console.error('Failed to save conversation:', error);
    }
  }, [currentConversationId, messages]);

  const respondToModeQuestion = useCallback((approved: boolean, userInput: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'mode_response',
        content: userInput,
        approved,
      }));
      setModeQuestion(null);
      setIsThinking(true);
    }
  }, []);

  const dismissModeQuestion = useCallback(() => {
    setModeQuestion(null);
  }, []);

  // Handle online/offline events
  useEffect(() => {
    const handleOnline = () => {
      console.log('[WS] Network online, reconnecting...');
      if (status === 'disconnected') {
        reconnectAttemptsRef.current = 0;
        connect();
      }
    };

    const handleOffline = () => {
      console.log('[WS] Network offline');
      disconnect();
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, [connect, disconnect, status]);

  useEffect(() => {
    mountedRef.current = true;
    connect();
    
    return () => {
      mountedRef.current = false;
      clearAllTimers();
      if (wsRef.current) {
        wsRef.current.close(1000, 'Component unmounted');
        wsRef.current = null;
      }
    };
  }, [connect, clearAllTimers]);

  return { 
    status, 
    connectionState,
    messages, 
    sendMessage, 
    sendClear, 
    isThinking, 
    toolCalls, 
    clearToolCalls, 
    clearMessages,
    currentConversationId,
    newConversation,
    loadConversation,
    deleteConversation,
    saveCurrentConversation,
    sendAgentLoop,
    modeQuestion,
    modeInfo,
    respondToModeQuestion,
    dismissModeQuestion,
    connect,
    disconnect,
  };
}

async function sendViaREST(content: string, handleMessage: (data: { type: string; [key: string]: unknown }) => void) {
  const formData = new FormData();
  formData.append('message', content);

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      handleMessage({ type: 'error', content: `Server error: ${response.status}` });
      return;
    }
    
    const reader = response.body?.getReader();
    if (!reader) {
      handleMessage({ type: 'error', content: 'No response stream' });
      return;
    }

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            handleMessage(JSON.parse(line.slice(6)));
          } catch {
          }
        }
      }
    }

    if (buffer.startsWith('data: ')) {
      try {
        handleMessage(JSON.parse(buffer.slice(6)));
      } catch {
      }
    }
    
    handleMessage({ type: 'done' });
  } catch {
    handleMessage({ type: 'error', content: 'Connection error' });
  }
}
