import { useState, useMemo, useEffect } from 'react';
import { ChatTabNew } from './components/ChatTabNew';
import { CommandTab } from './components/CommandTab';
import { ToolsTab } from './components/ToolsTab';
import { DashboardTab } from './components/DashboardTab';
import { VisualTab } from './components/VisualTab';
import { MusicTab } from './components/MusicTab';
import { ConversationSidebar } from './components/ConversationSidebar';
import { useWebSocket } from './hooks/useWebSocket';
import { DashboardProvider } from './contexts/DashboardContext';
import type { WSStatus } from './hooks/useWebSocket';
import './App.css';

type Tab = 'chat' | 'command' | 'tools' | 'dashboard' | 'visual' | 'music';

interface Settings {
  showThinking: boolean;
  streamMode: boolean;
}

function StatusIndicator({ status, isThinking }: { status: WSStatus; isThinking?: boolean }) {
  const statusConfig: Record<WSStatus, { text: string; className: string }> = {
    connecting: { text: 'Connecting...', className: '' },
    connected: { text: 'Ready', className: 'connected' },
    disconnected: { text: 'REST mode', className: 'connected' },
    reconnecting: { text: 'Reconnecting...', className: '' },
  };

  const config = statusConfig[status];

  return (
    <div className="status">
      <div className={`status-dot ${config.className} ${isThinking ? 'thinking' : ''}`} />
      <span className="status-text">{isThinking ? 'Thinking...' : config.text}</span>
    </div>
  );
}

interface Model {
  id: string;
  name: string;
  context: number;
  type: string;
  vision: boolean;
}

function SettingsPanel({ 
  isOpen, 
  onClose, 
  settings, 
  onSettingsChange 
}: { 
  isOpen: boolean; 
  onClose: () => void;
  settings: Settings;
  onSettingsChange: (settings: Settings) => void;
}) {
  const [models, setModels] = useState<Model[]>([]);
  const [currentModel, setCurrentModel] = useState<string>('');
  const [switching, setSwitching] = useState(false);

  useEffect(() => {
    if (isOpen) {
      fetch('/api/models/available')
        .then(r => r.json())
        .then(data => {
          if (data.success) {
            setModels(data.models);
            setCurrentModel(data.current);
          }
        })
        .catch(console.error);
    }
  }, [isOpen]);

  const updateSetting = (key: string, value: boolean) => {
    const newSettings = { ...settings, [key]: value };
    onSettingsChange(newSettings);
    localStorage.setItem('jarvis_settings', JSON.stringify(newSettings));
  };

  const switchModel = async (modelId: string) => {
    if (modelId === currentModel || switching) return;
    setSwitching(true);
    try {
      const resp = await fetch('/api/models/switch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: modelId }),
      });
      const data = await resp.json();
      if (data.success) {
        setCurrentModel(modelId);
      } else {
        console.error('Failed to switch model:', data.error);
      }
    } catch (e) {
      console.error('Failed to switch model:', e);
    } finally {
      setSwitching(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="settings-overlay" onClick={onClose}>
      <div className="settings-panel" onClick={e => e.stopPropagation()}>
        <div className="settings-header">
          <h3>Settings</h3>
          <button className="settings-close" onClick={onClose}>×</button>
        </div>
        <div className="settings-content">
          <div className="settings-section">
            <h4>Model Selection</h4>
            <div className="model-selector">
              {models.map(model => (
                <button
                  key={model.id}
                  className={`model-btn ${currentModel === model.id ? 'active' : ''}`}
                  onClick={() => switchModel(model.id)}
                  disabled={switching}
                >
                  <span className="model-name">{model.name}</span>
                  <span className="model-meta">
                    {model.vision && '👁️ '} 
                    {model.context >= 100000 ? '128K' : `${Math.round(model.context/1024)}K`} ctx
                  </span>
                </button>
              ))}
            </div>
            {switching && <div className="model-switching">Switching model...</div>}
          </div>
          <div className="setting-row">
            <span>Show thinking blocks</span>
            <div className={`toggle ${settings.showThinking ? 'on' : ''}`} onClick={() => updateSetting('showThinking', !settings.showThinking)}>
              <div className="toggle-slider" />
            </div>
          </div>
          <div className="setting-row">
            <span>Stream responses</span>
            <div className={`toggle ${settings.streamMode ? 'on' : ''}`} onClick={() => updateSetting('streamMode', !settings.streamMode)}>
              <div className="toggle-slider" />
            </div>
          </div>
          <div className="settings-section">
            <h4>Keyboard Shortcuts</h4>
            <div className="shortcut"><kbd>Enter</kbd> Send message</div>
            <div className="shortcut"><kbd>Shift+Enter</kbd> New line</div>
          </div>
        </div>
      </div>
    </div>
  );
}

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('chat');
  const [showSettings, setShowSettings] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [settings, setSettings] = useState<Settings>(() => {
    const saved = localStorage.getItem('jarvis_settings');
    return saved ? JSON.parse(saved) : { showThinking: true, streamMode: true };
  });
  
  const wsUrl = useMemo(() => {
    return `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws`;
  }, []);
  
  const { 
    status, 
    messages, 
    sendMessage, 
    sendClear, 
    isThinking, 
    toolCalls,
    currentConversationId,
    newConversation,
    loadConversation,
    deleteConversation,
    saveCurrentConversation,
    modeQuestion,
    modeInfo,
    respondToModeQuestion,
    dismissModeQuestion,
  } = useWebSocket(wsUrl);

  const handleClear = () => {
    if (confirm('Clear all conversation history?')) {
      sendClear();
    }
  };

  const handleNewConversation = async () => {
    await saveCurrentConversation();
    await newConversation();
    setSidebarOpen(false);
  };

  const handleLoadConversation = async (id: string) => {
    await saveCurrentConversation();
    await loadConversation(id);
    setSidebarOpen(false);
  };

  const handleDeleteConversation = async (id: string) => {
    await deleteConversation(id);
  };

  return (
    <DashboardProvider>
    <div className="app">
      <header className="header">
        <div className="header-left">
          <div className="logo">Jarvis</div>
          <nav className="tabs">
            <button
              className={`tab ${activeTab === 'chat' ? 'active' : ''}`}
              onClick={() => setActiveTab('chat')}
            >
              Chat
            </button>
            <button
              className={`tab ${activeTab === 'command' ? 'active' : ''}`}
              onClick={() => setActiveTab('command')}
            >
              Command
            </button>
            <button
              className={`tab ${activeTab === 'tools' ? 'active' : ''}`}
              onClick={() => setActiveTab('tools')}
            >
              Tools {toolCalls.length > 0 && <span className="tab-badge">{toolCalls.length}</span>}
            </button>
            <button
              className={`tab ${activeTab === 'dashboard' ? 'active' : ''}`}
              onClick={() => setActiveTab('dashboard')}
            >
              Dashboard
            </button>
            <button
              className={`tab ${activeTab === 'visual' ? 'active' : ''}`}
              onClick={() => setActiveTab('visual')}
            >
              👁️ Visual
            </button>
            <button
              className={`tab ${activeTab === 'music' ? 'active' : ''}`}
              onClick={() => setActiveTab('music')}
            >
              🎵 Music
            </button>
          </nav>
        </div>
        <div className="header-right">
          <button 
            className="history-btn" 
            title="Conversation History" 
            onClick={() => setSidebarOpen(!sidebarOpen)}
          >
            📋 History
          </button>
          <StatusIndicator status={status} isThinking={isThinking} />
          <button className="icon-btn" title="Settings" onClick={() => setShowSettings(true)}>
            ⚙️
          </button>
        </div>
      </header>

      <main className="main">
        {activeTab === 'chat' && (
          <ConversationSidebar
            currentConversationId={currentConversationId}
            onSelectConversation={handleLoadConversation}
            onNewConversation={handleNewConversation}
            onDeleteConversation={handleDeleteConversation}
            isOpen={sidebarOpen}
            onToggle={() => setSidebarOpen(!sidebarOpen)}
          />
        )}
        <div className={`tab-content ${activeTab === 'chat' ? 'active' : ''}`}>
          <ChatTabNew 
            messages={messages} 
            onSend={sendMessage} 
            onClear={handleClear}
            status={status} 
            showThinking={settings.showThinking}
            isThinking={isThinking}
            toolCalls={toolCalls}
            modeQuestion={modeQuestion}
            modeInfo={modeInfo}
            onModeResponse={respondToModeQuestion}
            onDismissModeQuestion={dismissModeQuestion}
          />
        </div>
        <div className={`tab-content ${activeTab === 'command' ? 'active' : ''}`}>
          <CommandTab />
        </div>
        <div className={`tab-content ${activeTab === 'tools' ? 'active' : ''}`}>
          <ToolsTab toolCalls={toolCalls} />
        </div>
        <div className={`tab-content ${activeTab === 'dashboard' ? 'active' : ''}`}>
          <DashboardTab messages={messages} isThinking={isThinking} />
        </div>
        <div className={`tab-content ${activeTab === 'visual' ? 'active' : ''}`}>
          <VisualTab />
        </div>
        <div className={`tab-content ${activeTab === 'music' ? 'active' : ''}`}>
          <MusicTab />
        </div>
      </main>

      {showSettings && (
        <SettingsPanel
          isOpen={showSettings}
          onClose={() => setShowSettings(false)}
          settings={settings}
          onSettingsChange={setSettings}
        />
      )}
    </div>
    </DashboardProvider>
  );
}

export default App;