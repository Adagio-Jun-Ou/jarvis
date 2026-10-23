import { useState, useRef, useEffect, useCallback } from 'react';
import { marked } from 'marked';
import type { Message, ToolCall, ModeQuestion, ModeInfo } from '../hooks/useWebSocket';
import { MediaInput } from './MediaInput';
import { useSpeech } from '../hooks/useSpeech';
import type { WSStatus } from '../hooks/useWebSocket';
import './ChatTabNew.css';

interface ChatTabProps {
  messages: Message[];
  onSend: (message: string) => void;
  onClear?: () => void;
  status: WSStatus;
  showThinking?: boolean;
  isThinking?: boolean;
  toolCalls?: ToolCall[];
  onImageUpload?: (imageData: string) => void;
  onImageGenerate?: (prompt: string) => void;
  modeQuestion?: ModeQuestion | null;
  modeInfo?: ModeInfo | null;
  onModeResponse?: (approved: boolean, userInput: string) => void;
  onDismissModeQuestion?: () => void;
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

  return {
    header,
    content: content.trim(),
  };
}

function stripThinkingTags(text: string): string {
  return text.replace(/<think[^>]*>[\s\S]*?<\/think>/gi, '').trim();
}

function renderMarkdown(text: string): string {
  if (!text) return '';
  try {
    return marked.parse(text) as string;
  } catch {
    return text;
  }
}

function ThinkingBlockMini({ 
  content, 
  header, 
  expanded,
  onToggle 
}: { 
  content: string; 
  header: string; 
  expanded: boolean;
  onToggle: () => void;
}) {
  return (
    <div className="thinking-block-mini" onClick={onToggle}>
      <div className="thinking-mini-header">
        <span className="thinking-mini-icon">💭</span>
        <span className="thinking-mini-title">{header}</span>
        <span className="thinking-mini-expand">{expanded ? '▼' : '▶'}</span>
      </div>
      {expanded && (
        <div className="thinking-mini-content">
          <pre>{content.slice(0, 300)}{content.length > 300 ? '...' : ''}</pre>
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

function ModeBanner({ 
  info,
  question,
  onApprove,
}: { 
  info?: ModeInfo | null;
  question?: ModeQuestion | null;
  onApprove?: (approved: boolean) => void;
}) {
  if (question && onApprove) {
    return (
      <div className="mode-banner question">
        <div className="mode-banner-content">
          <span className="mode-banner-icon">🤖</span>
          <div className="mode-banner-text">
            <div className="mode-banner-title">Autonomous Mode Request</div>
            <div className="mode-banner-message">{question.message}</div>
            {question.estimatedSteps > 0 && (
              <div className="mode-banner-details">
                <span>{question.estimatedSteps} steps</span>
                {question.toolsNeeded.length > 0 && (
                  <span> • {question.toolsNeeded.slice(0, 3).join(', ')}{question.toolsNeeded.length > 3 ? '...' : ''}</span>
                )}
              </div>
            )}
          </div>
          <div className="mode-banner-actions">
            <button className="mode-btn approve" onClick={() => onApprove(true)}>✓ Approve</button>
            <button className="mode-btn decline" onClick={() => onApprove(false)}>✗ Decline</button>
          </div>
        </div>
      </div>
    );
  }

  if (info) {
    return (
      <div className={`mode-banner info ${info.mode === 'agentic' ? 'agentic' : ''}`}>
        <span className="mode-banner-icon">{info.mode === 'agentic' ? '🤖' : '💬'}</span>
        <div className="mode-banner-text">
          <span className="mode-banner-mode">{info.mode === 'agentic' ? 'Autonomous' : 'Chat'} mode</span>
          {info.estimatedSteps && <span className="mode-banner-steps"> ({info.estimatedSteps} steps)</span>}
        </div>
        {info.toolsDetected && info.toolsDetected.length > 0 && (
          <span className="mode-banner-tools">🔧 {info.toolsDetected.length}</span>
        )}
      </div>
    );
  }

  return null;
}

function ToolsSidebar({ 
  toolCalls, 
  collapsed, 
  onToggle 
}: { 
  toolCalls: ToolCall[];
  collapsed: boolean;
  onToggle: () => void;
}) {
  const runningTools = toolCalls.filter(t => t.status === 'running');
  const completedTools = toolCalls.filter(t => t.status === 'completed').slice(-3);
  const failedTools = toolCalls.filter(t => t.status === 'failed').slice(-2);

  if (collapsed) {
    return (
      <div className="tools-sidebar collapsed" onClick={onToggle}>
        <div className="tools-sidebar-header">
          <span className="tools-sidebar-icon">🔧</span>
          {runningTools.length > 0 && <span className="tools-running-badge">{runningTools.length}</span>}
        </div>
      </div>
    );
  }

  return (
    <div className="tools-sidebar">
      <div className="tools-sidebar-header" onClick={onToggle}>
        <span className="tools-sidebar-icon">🔧</span>
        <span className="tools-sidebar-title">Tools</span>
        {runningTools.length > 0 && <span className="tools-running-badge">{runningTools.length} running</span>}
        <button className="tools-collapse-btn">◀</button>
      </div>
      
      <div className="tools-sidebar-content">
        {runningTools.length > 0 && (
          <div className="tools-section">
            <div className="tools-section-title">Running</div>
            {runningTools.map(tool => (
              <div key={tool.id} className="tool-item running">
                <div className="tool-item-header">
                  <span className="tool-pulse">⚡</span>
                  <span className="tool-name">{tool.name}</span>
                </div>
                <div className="tool-progress">
                  <div className="tool-progress-bar"></div>
                </div>
              </div>
            ))}
          </div>
        )}

        {completedTools.length > 0 && (
          <div className="tools-section">
            <div className="tools-section-title">Completed</div>
            {completedTools.map(tool => (
              <div key={tool.id} className="tool-item completed">
                <span className="tool-check">✓</span>
                <span className="tool-name">{tool.name}</span>
              </div>
            ))}
          </div>
        )}

        {failedTools.length > 0 && (
          <div className="tools-section">
            <div className="tools-section-title">Failed</div>
            {failedTools.map(tool => (
              <div key={tool.id} className="tool-item failed">
                <span className="tool-x">✗</span>
                <span className="tool-name">{tool.name}</span>
              </div>
            ))}
          </div>
        )}

        {toolCalls.length === 0 && (
          <div className="tools-empty">
            <span className="tools-empty-icon">🔧</span>
            <span className="tools-empty-text">No active tools</span>
          </div>
        )}
      </div>
    </div>
  );
}

function VoiceOutput({ text, autoSpeak }: { text: string; autoSpeak: boolean }) {
  const { speak, stop, isSpeaking } = useSpeech({ maxLength: 500 });
  const lastSpokenRef = useRef<string>('');

  const handleSpeak = useCallback(() => {
    if (!text || isSpeaking) return;
    lastSpokenRef.current = text;
    speak(text);
  }, [text, isSpeaking, speak]);

  useEffect(() => {
    if (autoSpeak && text && text !== lastSpokenRef.current && !isSpeaking) {
      const cleanText = text.replace(/<[^>]*>/g, '').slice(0, 500);
      if (cleanText.length > 10) {
        handleSpeak();
      }
    }
  }, [text, autoSpeak, isSpeaking, handleSpeak]);

  if (!text) return null;

  return (
    <div className="voice-output">
      {isSpeaking ? (
        <button onClick={stop} className="voice-btn speaking">🔇 Stop</button>
      ) : (
        <button onClick={handleSpeak} className="voice-btn">🔊 Speak</button>
      )}
    </div>
  );
}

interface ImagePreview {
  id: string;
  url: string;
  prompt?: string;
}

export function ChatTabNew({ 
  messages, 
  onSend,
  onClear,
  status, 
  showThinking = true,
  isThinking = false,
  toolCalls = [],
  onImageUpload,
  onImageGenerate,
  modeQuestion,
  modeInfo,
  onModeResponse,
  onDismissModeQuestion: _onDismissModeQuestion,
}: ChatTabProps) {
  const [input, setInput] = useState('');
  const [thinkingStates, setThinkingStates] = useState<Record<string, ThinkingState>>({});
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessingMedia, setIsProcessingMedia] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(false);
  const [imagePreviews, setImagePreviews] = useState<ImagePreview[]>([]);
  const [toolsCollapsed, setToolsCollapsed] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    textareaRef.current?.focus();
  }, []);

  useEffect(() => {
    if (toolCalls.some(t => t.status === 'running')) {
      setToolsCollapsed(false);
    }
  }, [toolCalls]);

  const handleSubmit = () => {
    if (input.trim()) {
      onSend(input.trim());
      setInput('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const toggleThinking = (msgId: string) => {
    setThinkingStates(prev => {
      const current = prev[msgId] || { collapsed: true, content: '', header: 'Thought' };
      return {
        ...prev,
        [msgId]: { ...current, collapsed: !current.collapsed },
      };
    });
  };

  const handleVoiceRecord = async (audioBlob: Blob) => {
    setIsRecording(false);
    setIsProcessingMedia(true);
    
    try {
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.webm');
      
      const response = await fetch('/api/transcribe', {
        method: 'POST',
        body: formData,
      });
      
      const data = await response.json();
      
      if (data.text) {
        setInput(prev => prev ? `${prev} ${data.text}` : data.text);
      } else if (data.error) {
        console.error('Transcription error:', data.error);
      }
    } catch (error) {
      console.error('Voice recording error:', error);
    } finally {
      setIsProcessingMedia(false);
    }
  };

  const handleImageUploadLocal = async (imageData: string) => {
    setIsProcessingMedia(true);
    
    try {
      if (onImageUpload) {
        onImageUpload(imageData);
      } else {
        const response = await fetch('/api/analyze-image', {
          method: 'POST',
          body: JSON.stringify({ image: imageData }),
          headers: { 'Content-Type': 'application/json' },
        });
        
        const data = await response.json();
        
        if (data.success && data.description) {
          setInput(prev => prev + `\n[Image analysis: ${data.description}]`);
        } else {
          console.error('Image analysis failed:', data.error);
        }
      }
    } catch (error) {
      console.error('Image upload error:', error);
    } finally {
      setIsProcessingMedia(false);
    }
  };

  const handleImageGenerateLocal = async (prompt: string) => {
    setIsProcessingMedia(true);
    
    try {
      if (onImageGenerate) {
        onImageGenerate(prompt);
      } else {
        const response = await fetch('/api/generate-image', {
          method: 'POST',
          body: JSON.stringify({ prompt }),
          headers: { 'Content-Type': 'application/json' },
        });
        
        const data = await response.json();
        
        if (data.success) {
          const imageUrl = data.image_url || (data.image_base64 ? `data:image/png;base64,${data.image_base64}` : null);
          if (imageUrl) {
            const preview: ImagePreview = {
              id: `img-${Date.now()}`,
              url: imageUrl,
              prompt: data.prompt || prompt,
            };
            setImagePreviews(prev => [...prev, preview]);
            setInput(prev => prev + `\n![Generated Image](${imageUrl})`);
          }
        } else {
          console.error('Image generation failed:', data.error);
        }
      }
    } catch (error) {
      console.error('Image generation error:', error);
    } finally {
      setIsProcessingMedia(false);
    }
  };

  const lastAssistantMessage = [...messages].reverse().find(m => m.role === 'assistant');

  const renderMessage = (msg: Message) => {
    const parsed = parseThinking(msg.content);
    const thinkingState = thinkingStates[msg.id] || {
      collapsed: true,
      content: '',
      header: 'Thought',
    };

    const displayContent = showThinking ? stripThinkingTags(msg.content) : msg.content;
    const msgThinking = msg.thinking;

    return (
      <div className={`message ${msg.role}`} key={msg.id}>
        {msg.role === 'user' && (
          <div className="message-content">{msg.content}</div>
        )}
        {msg.role === 'assistant' && (
          <>
            {parsed && showThinking && (
              <ThinkingBlockMini
                content={parsed.content}
                header={parsed.header}
                expanded={!thinkingState.collapsed}
                onToggle={() => toggleThinking(msg.id)}
              />
            )}
            {msgThinking && showThinking && (
              <ThinkingBlockMini
                content={msgThinking}
                header="Reasoning"
                expanded={!thinkingState.collapsed}
                onToggle={() => toggleThinking(msg.id)}
              />
            )}
            {msg.model && (
              <div className={`model-indicator model-${msg.model}`}>
                {msg.model === 'primary' && '🌸 Kimi K2.5'}
                {msg.model === 'backend' && '🧠 GLM-5'}
                {msg.model === 'collaborative' && '🔗 Kimi + GLM-5'}
              </div>
            )}
            <div 
              className="message-content"
              dangerouslySetInnerHTML={{ __html: renderMarkdown(displayContent) }}
            />
            {voiceEnabled && msg === lastAssistantMessage && (
              <VoiceOutput text={displayContent} autoSpeak={false} />
            )}
          </>
        )}
        {msg.role === 'system' && (
          <div className="message-content system">{msg.content}</div>
        )}
      </div>
    );
  };

  const handleModeApprove = (approved: boolean) => {
    if (onModeResponse && modeQuestion) {
      onModeResponse(approved, modeQuestion.userInput || '');
    }
  };

  return (
    <div className="chat-tab-new">
      <ModeBanner 
        info={modeInfo}
        question={modeQuestion}
        onApprove={handleModeApprove}
      />
      
      <div className="chat-main-area">
        <div className="chat-messages">
          {messages.length === 0 && imagePreviews.length === 0 && (
            <div className="welcome-message">
              <div className="welcome-title">Jarvis</div>
              <div className="welcome-subtitle">How can I help you today?</div>
              <div className="welcome-features">
                <div className="welcome-feature">🎤 Voice input</div>
                <div className="welcome-feature">📷 Image analysis</div>
                <div className="welcome-feature">🎨 Image generation</div>
              </div>
            </div>
          )}
          
          {imagePreviews.length > 0 && (
            <div className="image-previews">
              {imagePreviews.map(preview => (
                <div key={preview.id} className="image-preview-item">
                  <img src={preview.url} alt={preview.prompt || 'Generated'} />
                  {preview.prompt && <div className="image-preview-prompt">{preview.prompt}</div>}
                  <button className="image-preview-remove" onClick={() => setImagePreviews(prev => prev.filter(p => p.id !== preview.id))}>×</button>
                </div>
              ))}
            </div>
          )}
          
          {messages.map(renderMessage)}
          {isThinking && <ThinkingIndicator />}
          <div ref={messagesEndRef} />
        </div>

        {toolCalls.length > 0 && (
          <ToolsSidebar 
            toolCalls={toolCalls} 
            collapsed={toolsCollapsed}
            onToggle={() => setToolsCollapsed(!toolsCollapsed)}
          />
        )}
      </div>

      <div className="chat-input-area">
        <div className="input-controls">
          <MediaInput
            onVoiceRecord={handleVoiceRecord}
            onImageUpload={handleImageUploadLocal}
            onImageGenerate={handleImageGenerateLocal}
            isRecording={isRecording}
            isProcessing={isProcessingMedia}
            disabled={status === 'connecting' || isThinking}
          />
          <button
            className={`voice-toggle ${voiceEnabled ? 'active' : ''}`}
            onClick={() => setVoiceEnabled(!voiceEnabled)}
            title={voiceEnabled ? 'Disable voice output' : 'Enable voice output'}
          >
            {voiceEnabled ? '🔊' : '🔈'}
          </button>
          {onClear && messages.length > 0 && (
            <button
              className="clear-btn"
              onClick={onClear}
              title="Clear conversation"
              disabled={isThinking}
            >
              🗑️
            </button>
          )}
        </div>
        <div className="input-wrapper">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={status === 'connected' ? 'Send a message...' : 'Working in REST mode...'}
            disabled={status === 'connecting' || isThinking}
            rows={1}
          />
          <button
            className="send-btn"
            onClick={handleSubmit}
            disabled={!input.trim() || status === 'connecting' || isThinking}
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}