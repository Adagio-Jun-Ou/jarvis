import { useState, useEffect } from 'react';
import type { ToolCall } from '../hooks/useWebSocket';
import { useDashboard } from '../contexts/DashboardContext';

interface ToolsTabProps {
  toolCalls: ToolCall[];
}

interface AvailableTool {
  name: string;
  description: string;
  requires_permission: boolean;
}

export function ToolsTab({ toolCalls }: ToolsTabProps) {
  const [availableTools, setAvailableTools] = useState<AvailableTool[]>([]);
  const [loading, setLoading] = useState(true);
  const [permissions, setPermissions] = useState<Record<string, string>>({});
  const [savingTool, setSavingTool] = useState<string | null>(null);
  const dashboardData = useDashboard();

  useEffect(() => {
    const fetchPermissions = async () => {
      try {
        const permResp = await fetch('/api/permissions');
        if (permResp.ok) {
          const permData = await permResp.json();
          setPermissions(permData.permissions || {});
        }
      } catch (e) {
        console.error('Failed to fetch permissions:', e);
      } finally {
        setLoading(false);
      }
    };
    fetchPermissions();
  }, []);

  useEffect(() => {
    if (dashboardData.data?.tools?.tools) {
      setAvailableTools(dashboardData.data.tools.tools);
    }
  }, [dashboardData.data]);

  const setPermission = async (toolName: string, permission: string) => {
    setSavingTool(toolName);
    try {
      const resp = await fetch('/api/permissions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tool: toolName, permission }),
      });
      if (resp.ok) {
        setPermissions(prev => ({ ...prev, [toolName]: permission }));
      }
    } catch (e) {
      console.error('Failed to set permission:', e);
    } finally {
      setSavingTool(null);
    }
  };

  const runningTools = toolCalls.filter(t => t.status === 'running');
  const completedTools = toolCalls.filter(t => t.status === 'completed');
  const failedTools = toolCalls.filter(t => t.status === 'failed');

  return (
    <div className="tools-tab">
      <div className="tools-header">
        <h2>Tools</h2>
        <span className="tools-count">{availableTools.length} available</span>
      </div>
      
      {loading ? (
        <div className="tools-loading">Loading tools...</div>
      ) : (
        <div className="tools-sections">
          {toolCalls.length > 0 && (
            <div className="tools-section tools-execution-section">
              <h3>
                🔧 Tool Execution
                {runningTools.length > 0 && (
                  <span className="running-count">{runningTools.length} running</span>
                )}
              </h3>
              
              {runningTools.length > 0 && (
                <div className="tools-running">
                  {runningTools.map(tool => (
                    <div key={tool.id} className="tool-item running">
                      <div className="tool-item-header">
                        <span className="tool-item-icon pulse">⚡</span>
                        <span className="tool-item-name">{tool.name}</span>
                        <span className="tool-item-status running">RUNNING</span>
                      </div>
                      <div className="tool-item-progress">
                        <div className="progress-bar">
                          <div className="progress-bar-fill"></div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {completedTools.length > 0 && (
                <div className="tools-completed">
                  <h4>✅ Completed ({completedTools.length})</h4>
                  {completedTools.slice(-5).reverse().map(tool => (
                    <div key={tool.id} className="tool-item completed">
                      <div className="tool-item-header">
                        <span className="tool-item-icon">✅</span>
                        <span className="tool-item-name">{tool.name}</span>
                        <span className="tool-item-status completed">DONE</span>
                      </div>
                      {tool.result && (
                        <div className="tool-item-result">
                          <pre>{tool.result.slice(0, 300)}{tool.result.length > 300 ? '...' : ''}</pre>
                        </div>
                      )}
                      <div className="tool-item-time">
                        {new Date(tool.timestamp).toLocaleTimeString()}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {failedTools.length > 0 && (
                <div className="tools-failed">
                  <h4>❌ Failed ({failedTools.length})</h4>
                  {failedTools.slice(-3).reverse().map(tool => (
                    <div key={tool.id} className="tool-item failed">
                      <div className="tool-item-header">
                        <span className="tool-item-icon">❌</span>
                        <span className="tool-item-name">{tool.name}</span>
                        <span className="tool-item-status failed">FAILED</span>
                      </div>
                      {tool.result && (
                        <div className="tool-item-result error">
                          <pre>{tool.result}</pre>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {toolCalls.length === 0 && (
            <div className="tools-section">
              <h3>Tool Execution</h3>
              <div className="tools-empty">
                <div className="tools-empty-icon">🔧</div>
                <p>No tools currently executing</p>
                <p className="tools-empty-hint">Tools will appear here when Jarvis uses them</p>
              </div>
            </div>
          )}

          <div className="tools-section">
            <h3>Available Tools ({availableTools.length})</h3>
            <div className="available-tools">
              {availableTools.map(tool => (
                <div key={tool.name} className="available-tool">
                  <div className="available-tool-header">
                    <span className="available-tool-name">🔧 {tool.name}</span>
                    {tool.requires_permission && <span className="permission-badge" title="Requires permission">⚠️</span>}
                    <span className={`permission-status ${(permissions[tool.name] || 'ask')}`}>
                      {permissions[tool.name] === 'allow' ? '✓ Allowed' : 
                       permissions[tool.name] === 'deny' ? '✗ Denied' : '? Ask'}
                    </span>
                  </div>
                  <div className="available-tool-desc">{tool.description}</div>
                  <div className="tool-permission-controls">
                    <span className="permission-label">Permission:</span>
                    <select 
                      value={permissions[tool.name] || 'ask'}
                      onChange={(e) => setPermission(tool.name, e.target.value)}
                      disabled={savingTool === tool.name}
                      className="permission-select"
                    >
                      <option value="ask">Ask Every Time</option>
                      <option value="allow">Always Allow</option>
                      <option value="deny">Always Deny</option>
                    </select>
                    {savingTool === tool.name && <span className="saving-indicator">💾</span>}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}