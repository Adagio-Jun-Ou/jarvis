import React, { useState, useEffect } from 'react';

interface ToolCall {
  id: string;
  name: string;
  status: 'running' | 'completed' | 'failed';
  result?: string;
  timestamp: number;
}

interface AvailableTool {
  name: string;
  description: string;
  requires_permission: boolean;
}

interface Props {
  toolCalls: ToolCall[];
}

export const MobileTools: React.FC<Props> = ({ toolCalls }) => {
  const [availableTools, setAvailableTools] = useState<AvailableTool[]>([]);
  const [loading, setLoading] = useState(true);
  const [showAvailable, setShowAvailable] = useState(false);

  useEffect(() => {
    const fetchTools = async () => {
      try {
        const resp = await fetch('/api/tools');
        const data = await resp.json();
        setAvailableTools(data.tools || []);
      } catch (e) {
        console.error('Failed to fetch tools:', e);
      } finally {
        setLoading(false);
      }
    };
    fetchTools();
  }, []);

  const runningTools = toolCalls.filter(t => t.status === 'running');
  const completedTools = toolCalls.filter(t => t.status === 'completed');
  const failedTools = toolCalls.filter(t => t.status === 'failed');

  return (
    <div className="tools-view">
      <div className="tools-header">
        <h2>🔧 Tools</h2>
        <span className="tools-count">{availableTools.length} available</span>
      </div>

      <div className="tools-content">
        {runningTools.length > 0 && (
          <div className="tools-section">
            <h3 className="section-title">
              <span className="status-dot running"></span>
              Running ({runningTools.length})
            </h3>
            <div className="tools-list">
              {runningTools.map(tool => (
                <div key={tool.id} className="tool-card running">
                  <div className="tool-header">
                    <span className="tool-name">⚡ {tool.name}</span>
                    <span className="tool-status running">RUNNING</span>
                  </div>
                  <div className="progress-bar">
                    <div className="progress-fill"></div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {(completedTools.length > 0 || failedTools.length > 0) && (
          <div className="tools-section">
            <h3 className="section-title">Recent Activity</h3>
            <div className="tools-list">
              {completedTools.slice(-5).reverse().map(tool => (
                <div key={tool.id} className="tool-card completed">
                  <div className="tool-header">
                    <span className="tool-name">✅ {tool.name}</span>
                    <span className="tool-status completed">DONE</span>
                  </div>
                  {tool.result && (
                    <div className="tool-result">
                      <pre>{tool.result.slice(0, 150)}{tool.result.length > 150 ? '...' : ''}</pre>
                    </div>
                  )}
                  <div className="tool-time">
                    {new Date(tool.timestamp).toLocaleTimeString()}
                  </div>
                </div>
              ))}
              
              {failedTools.slice(-3).reverse().map(tool => (
                <div key={tool.id} className="tool-card failed">
                  <div className="tool-header">
                    <span className="tool-name">❌ {tool.name}</span>
                    <span className="tool-status failed">FAILED</span>
                  </div>
                  {tool.result && (
                    <div className="tool-result error">
                      <pre>{tool.result}</pre>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {toolCalls.length === 0 && (
          <div className="empty-state">
            <div className="empty-icon">🔧</div>
            <h3>No Tools Running</h3>
            <p>Tools will appear here when Jarvis uses them</p>
          </div>
        )}

        <div className="tools-section">
          <button 
            className="toggle-section-btn"
            onClick={() => setShowAvailable(!showAvailable)}
          >
            <span>📦 Available Tools</span>
            <span className="toggle-icon">{showAvailable ? '▼' : '▶'}</span>
          </button>
          
          {showAvailable && (
            <div className="available-tools">
              {loading ? (
                <div className="loading">Loading tools...</div>
              ) : (
                availableTools.map(tool => (
                  <div key={tool.name} className="available-tool-card">
                    <div className="available-tool-header">
                      <span className="available-tool-name">{tool.name}</span>
                      {tool.requires_permission && <span className="perm-badge">⚠️</span>}
                    </div>
                    <p className="available-tool-desc">{tool.description}</p>
                  </div>
                ))
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default MobileTools;