import React, { useState, useEffect } from 'react';

interface Model {
  id: string;
  name: string;
  context: number;
  type: string;
  vision: boolean;
}

type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'reconnecting';

interface Props {
  showThinking: boolean;
  onSettingsChange: (settings: { showThinking: boolean }) => void;
  onLogout: () => void;
  connectionStatus: ConnectionStatus;
}

export const MobileSettings: React.FC<Props> = ({ 
  showThinking, 
  onSettingsChange, 
  onLogout,
  connectionStatus 
}) => {
  const [models, setModels] = useState<Model[]>([]);
  const [currentModel, setCurrentModel] = useState<string>('');
  const [switching, setSwitching] = useState(false);
  const [about, setAbout] = useState({ version: '3.0', build: '' });

  useEffect(() => {
    fetch('/api/models/available')
      .then(r => r.json())
      .then(data => {
        if (data.success) {
          setModels(data.models);
          setCurrentModel(data.current);
        }
      })
      .catch(console.error);
    
    fetch('/api/status')
      .then(r => r.json())
      .then(data => {
        if (data.success && data.version) {
          setAbout({ version: data.version, build: data.build || '' });
        }
      })
      .catch(console.error);
  }, []);

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
      }
    } catch (e) {
      console.error('Failed to switch model:', e);
    } finally {
      setSwitching(false);
    }
  };

  const clearCache = async () => {
    try {
      await fetch('/api/cache/clear', { method: 'POST' });
    } catch (e) {
      console.error('Failed to clear cache:', e);
    }
  };

  return (
    <div className="settings-view">
      <div className="settings-header">
        <h2>⚙️ Settings</h2>
      </div>

      <div className="settings-content">
        <div className="settings-section">
          <h3>Connection</h3>
          <div className="connection-status">
            <div className={`status-indicator ${connectionStatus}`} />
            <span className="status-text">
              {connectionStatus === 'connected' ? 'Connected' : 
               connectionStatus === 'connecting' ? 'Connecting...' : 'Disconnected'}
            </span>
          </div>
        </div>

        <div className="settings-section">
          <h3>Model Selection</h3>
          <div className="model-list">
            {models.map(model => (
              <button
                key={model.id}
                className={`model-option ${currentModel === model.id ? 'active' : ''}`}
                onClick={() => switchModel(model.id)}
                disabled={switching}
              >
                <div className="model-info">
                  <span className="model-name">{model.name}</span>
                  <span className="model-meta">
                    {model.vision && '👁️ '}
                    {model.context >= 100000 ? '128K' : `${Math.round(model.context/1024)}K`} context
                  </span>
                </div>
                {currentModel === model.id && <span className="check-mark">✓</span>}
              </button>
            ))}
          </div>
          {switching && <div className="switching-msg">Switching model...</div>}
        </div>

        <div className="settings-section">
          <h3>Display</h3>
          <div className="setting-row">
            <div className="setting-info">
              <span className="setting-label">Show Thinking Blocks</span>
              <span className="setting-desc">Display AI reasoning process</span>
            </div>
            <button
              className={`toggle-btn ${showThinking ? 'on' : 'off'}`}
              onClick={() => onSettingsChange({ showThinking: !showThinking })}
            >
              <span className="toggle-slider" />
            </button>
          </div>
        </div>

        <div className="settings-section">
          <h3>Data</h3>
          <button className="danger-btn" onClick={clearCache}>
            🗑️ Clear Cache
          </button>
        </div>

        <div className="settings-section">
          <h3>Account</h3>
          <button className="logout-btn" onClick={onLogout}>
            🚪 Sign Out
          </button>
        </div>

        <div className="settings-section about-section">
          <h3>About</h3>
          <div className="about-content">
            <div className="about-logo">
              <svg viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="50" cy="50" r="45" stroke="currentColor" strokeWidth="2" />
                <circle cx="35" cy="40" r="5" fill="currentColor" />
                <circle cx="65" cy="40" r="5" fill="currentColor" />
                <path d="M35 65 Q50 75 65 65" stroke="currentColor" strokeWidth="3" fill="none" strokeLinecap="round" />
              </svg>
            </div>
            <div className="about-info">
              <span className="about-name">Jarvis AI Assistant</span>
              <span className="about-version">Version {about.version}</span>
              {about.build && <span className="about-build">{about.build}</span>}
            </div>
          </div>
        </div>

        <div className="settings-section">
          <h3>Security</h3>
          <div className="security-info">
            <span className="security-item">🔐 WebAuthn / FaceID Enabled</span>
            <span className="security-item">🛡️ JWT Token Authentication</span>
            <span className="security-item">🔒 HTTPS Encrypted Connection</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MobileSettings;