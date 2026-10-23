import { useState, useRef, useEffect, useCallback } from 'react';
import { marked } from 'marked';
import type { Message, ToolCall, ModeQuestion, ModeInfo } from '../hooks/useWebSocket';
import { MediaInput } from './MediaInput';
import { useSpeech } from '../hooks/useSpeech';
import './ChatTab.css';

interface ChatTabProps {
  messages: Message[];
  onSend: (message: string) => void;
  onClear?: () => void;
  status: 'connecting' | 'connected' | 'disconnected';
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

function ModeQuestionDialog({ 
  question, 
  onApprove, 
  onDismiss 
}: { 
  question: ModeQuestion; 
  onApprove: (approved: boolean) => void;
  onDismiss: () => void;
}) {
  return (
    <div className="mode-question-overlay">
      <div className="mode-question-dialog">
        <div className="mode-question-icon">🤖</div>
        <h3>Autonomous Mode</h3>
        <p className="mode-question-message">{question.message}</p>
        
        {question.estimatedSteps > 0 && (
          <div className="mode-question-details">
            <div className="mode-detail">
              <span className="mode-detail-label">Estimated steps:</span>
              <span className="mode-detail-value">{question.estimatedSteps}</span>
            </div>
            {question.toolsNeeded.length > 0 && (
              <div className="mode-detail">
                <span className="mode-detail-label">Tools needed:</span>
                <span className="mode-detail-value">{question.toolsNeeded.join(', ')}</span>
              </div>
            )}
          </div>
        )}
        
        <div className="mode-question-actions">
          <button 
            className="mode-btn approve"
            onClick={() => onApprove(true)}
          >
            ✅ Yes, proceed autonomously
          </button>
          <button 
            className="mode-btn decline"
            onClick={() => onApprove(false)}
          >
            💬 No, stay in chat mode
          </button>
          <button 
            className="mode-btn dismiss"
            onClick={onDismiss}
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}

function ModeInfoBanner({ info }: { info: ModeInfo }) {
  if (!info) return null;
  
  return (
    <div className="mode-info-banner">
      <span className="mode-info-icon">
        {info.mode === 'agentic' ? '🤖' : '💬'}
      </span>
      <span className="mode-info-text">
        {info.mode === 'agentic' 
          ? `Autonomous mode${info.estimatedSteps ? ` (${info.estimatedSteps} steps)` : ''}`
          : 'Chat mode'}
      </span>
      {info.toolsDetected && info.toolsDetected.length > 0 && (
        <span className="mode-info-tools">
          🔧 {info.toolsDetected.length} tool(s)
        </span>
      )}
    </div>
  );
}

function ToolCallItem({ tool }: { tool: ToolCall }) {
  return (
    <div className={`tool-call ${tool.status}`}>
      <div className="tool-call-header">
        <span className="tool-icon">
          {tool.status === 'running' ? '⚡' : tool.status === 'completed' ? '✅' : '❌'}
        </span>
        <span className="tool-name">{tool.name}</span>
        <span className="tool-status">{tool.status}</span>
      </div>
      {tool.result && (
        <div className="tool-result">
          <pre>{tool.result.slice(0, 200)}{tool.result.length > 200 ? '...' : ''}</pre>
        </div>
      )}
    </div>
  );
}

function ToolCallPanel({ toolCalls }: { toolCalls: ToolCall[] }) {
  if (toolCalls.length === 0) return null;
  
  return (
    <div className="tool-calls-panel">
      <div className="tool-calls-header">
        <span className="tool-calls-title">🔧 Tool Execution</span>
        <span className="tool-calls-count">{toolCalls.length} tool{toolCalls.length !== 1 ? 's' : ''}</span>
      </div>
      <div className="tool-calls-list">
        {toolCalls.map(tool => (
          <ToolCallItem key={tool.id} tool={tool} />
        ))}
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

export function ChatTab({ 
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
  onDismissModeQuestion,
}: ChatTabProps) {
  const [input, setInput] = useState('');
  const [thinkingStates, setThinkingStates] = useState<Record<string, ThinkingState>>({});
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessingMedia, setIsProcessingMedia] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(false);
  const [imagePreviews, setImagePreviews] = useState<ImagePreview[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    textareaRef.current?.focus();
  }, []);

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

    const hasThinking = parsed !== null && showThinking;
    const displayContent = showThinking ? stripThinkingTags(msg.content) : msg.content;
    
    const msgThinking = msg.thinking;
    
    if (hasThinking && parsed) {
      const currentState: ThinkingState = {
        collapsed: thinkingState.collapsed,
        content: parsed.content,
        header: parsed.header,
      };

      return (
        <div className={`message ${msg.role}`} key={msg.id}>
          {msg.role === 'user' && (
            <div className="message-content">{msg.content}</div>
          )}
          {msg.role === 'assistant' && (
            <>
              <ThinkingBlock
                content={currentState.content}
                header={currentState.header}
                collapsed={currentState.collapsed}
                onToggle={() => toggleThinking(msg.id)}
              />
              {msgThinking && (
                <ThinkingBlock
                  content={msgThinking}
                  header="Model Reasoning"
                  collapsed={thinkingState.collapsed}
                  onToggle={() => toggleThinking(msg.id)}
                />
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
        </div>
      );
    }

    return (
      <div className={`message ${msg.role}`} key={msg.id}>
        {msg.role === 'user' && (
          <div className="message-content">{msg.content}</div>
        )}
        {msg.role === 'assistant' && (
          <>
            {msgThinking && showThinking && (
              <ThinkingBlock
                content={msgThinking}
                header="Model Reasoning"
                collapsed={thinkingState.collapsed}
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
          <div className="message-content">{msg.content}</div>
        )}
      </div>
    );
  };

  return (
    <div className="chat-tab">
      {modeQuestion && onModeResponse && (
        <ModeQuestionDialog 
          question={modeQuestion}
          onApprove={(approved) => onModeResponse(approved, modeQuestion.userInput)}
          onDismiss={onDismissModeQuestion || (() => {})}
        />
      )}
      
      <div className="chat-messages">
        {modeInfo && <ModeInfoBanner info={modeInfo} />}
        {messages.length === 0 && imagePreviews.length === 0 && (
          <div className="welcome-message">
            <h2>Jarvis</h2>
            <p>How can I help you today?</p>
            <div className="welcome-features">
              <span>🎤 Voice input</span>
              <span>📷 Image analysis</span>
              <span>🎨 Image generation</span>
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
      
      <ToolCallPanel toolCalls={toolCalls} />

      <div className="input-area">
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
            {isThinking ? '...' : 'Send'}
          </button>
        </div>
      </div>
    </div>
  );
}