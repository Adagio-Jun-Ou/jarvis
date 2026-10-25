import { useState, useEffect, useRef } from 'react';
import './CommandTab.css';

interface LogEntry {
  timestamp: string;
  level: string;
  subsystem: string;
  message: string;
}

interface CommandStats {
  total: number;
  primary: number;
  backend: number;
  cached: number;
  tokens: number;
  latency: number;
}

type ViewMode = 'shell' | 'logs' | 'history';

const LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR'];
const LOG_LEVEL_COLORS: Record<string, string> = {
  DEBUG: '#6b7280',
  INFO: '#22c55e',
  WARNING: '#f59e0b',
  ERROR: '#ef4444',
};

export function CommandTab() {
  const [viewMode, setViewMode] = useState<ViewMode>('shell');
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [stats, setStats] = useState<CommandStats>({
    total: 0,
    primary: 0,
    backend: 0,
    cached: 0,
    tokens: 0,
    latency: 0,
  });
  const [input, setInput] = useState('');
  const [commandHistory, setCommandHistory] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [logFilter, setLogFilter] = useState<string>('all');
  const [logSearch, setLogSearch] = useState('');
  const [loading, setLoading] = useState(false);
  const [pendingCommand, setPendingCommand] = useState<string | null>(null);
  const terminalRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 2000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [logs]);

  const fetchData = async () => {
    try {
      const [logsResp, statsResp] = await Promise.all([
        fetch('/api/subsystem_logs').catch(() => null),
        fetch('/api/cli-stats').catch(() => null),
      ]);
      
      if (logsResp) {
        const logsData = await logsResp.json();
        const newLogs: LogEntry[] = [];
        for (const [subsystem, entries] of Object.entries(logsData)) {
          if (Array.isArray(entries)) {
            for (const entry of entries) {
              newLogs.push({
                timestamp: entry.time || new Date().toISOString().split('T')[1].split('.')[0],
                level: entry.level || 'INFO',
                subsystem: subsystem.toUpperCase(),
                message: entry.message || '',
              });
            }
          }
        }
        if (newLogs.length > 0) {
          setLogs(prev => {
            const combined = [...prev, ...newLogs];
            return combined.slice(-200);
          });
        }
      }

      if (statsResp) {
        const statsData = await statsResp.json();
        setStats({
          total: statsData.total_calls || 0,
          primary: statsData.primary_calls || 0,
          backend: statsData.backend_calls || 0,
          cached: statsData.cache_hits || 0,
          tokens: statsData.total_tokens || 0,
          latency: statsData.avg_latency || 0,
        });
      }
    } catch (e) {
      console.error('Failed to fetch data:', e);
    }
  };

  const handleCommand = async (cmd: string) => {
    const trimmed = cmd.trim();
    if (!trimmed) return;

    addLog('INPUT', 'USER', `> ${trimmed}`);
    setCommandHistory(prev => [trimmed, ...prev.slice(0, 49)]);
    setHistoryIndex(-1);
    setInput('');
    setLoading(true);

    try {
      const resp = await fetch('/api/command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command: trimmed }),
      });
      
      if (resp.ok) {
        const data = await resp.json();
        addLog('OUTPUT', 'JARVIS', data.response || JSON.stringify(data, null, 2));
      } else {
        addLog('ERROR', 'SYSTEM', `Command failed: ${resp.status}`);
      }
    } catch (e) {
      addLog('ERROR', 'SYSTEM', 'Connection error');
    } finally {
      setLoading(false);
    }
  };

  const addLog = (level: string, subsystem: string, message: string) => {
    setLogs(prev => [...prev, {
      timestamp: new Date().toISOString().split('T')[1].split('.')[0],
      level,
      subsystem,
      message,
    }].slice(-200));
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !loading) {
      e.preventDefault();
      if (pendingCommand) {
        handleCommand(`approve ${pendingCommand}`);
        setPendingCommand(null);
      } else {
        handleCommand(input);
      }
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (historyIndex < commandHistory.length - 1) {
        const newIndex = historyIndex + 1;
        setHistoryIndex(newIndex);
        setInput(commandHistory[newIndex] || '');
      }
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (historyIndex > 0) {
        const newIndex = historyIndex - 1;
        setHistoryIndex(newIndex);
        setInput(commandHistory[newIndex] || '');
      } else {
        setHistoryIndex(-1);
        setInput('');
      }
    } else if (e.key === 'Escape') {
      setPendingCommand(null);
    } else if (e.key === 'l' && e.ctrlKey) {
      e.preventDefault();
      setLogs([]);
    }
  };

  const filteredLogs = logs.filter(log => {
    if (logFilter !== 'all' && log.level !== logFilter) return false;
    if (logSearch && !log.message.toLowerCase().includes(logSearch.toLowerCase())) return false;
    return true;
  });

  const quickCommands = [
    { label: 'status', cmd: 'status' },
    { label: 'models', cmd: 'models' },
    { label: 'tools', cmd: 'tools' },
    { label: 'clear', cmd: 'clear' },
  ];

  return (
    <div className="command-tab">
      <div className="command-header">
        <div className="command-tabs">
          <button 
            className={`cmd-tab ${viewMode === 'shell' ? 'active' : ''}`}
            onClick={() => setViewMode('shell')}
          >
            ⚡ Shell
          </button>
          <button 
            className={`cmd-tab ${viewMode === 'logs' ? 'active' : ''}`}
            onClick={() => setViewMode('logs')}
          >
            📋 Logs
          </button>
          <button 
            className={`cmd-tab ${viewMode === 'history' ? 'active' : ''}`}
            onClick={() => setViewMode('history')}
          >
            🕐 History
          </button>
        </div>
        <div className="command-stats">
          <div className="stat-item">
            <span className="stat-label">API</span>
            <span className="stat-value">{stats.total}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Primary</span>
            <span className="stat-value primary">{stats.primary}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Backend</span>
            <span className="stat-value backend">{stats.backend}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Latency</span>
            <span className="stat-value">{stats.latency}ms</span>
          </div>
        </div>
      </div>

      <div className="command-content">
        {viewMode === 'shell' && (
          <div className="shell-view">
            <div className="quick-commands">
              {quickCommands.map(qc => (
                <button
                  key={qc.label}
                  className="quick-cmd"
                  onClick={() => handleCommand(qc.cmd)}
                  disabled={loading}
                >
                  {qc.label}
                </button>
              ))}
            </div>
            <div className="terminal-output" ref={terminalRef} onClick={() => inputRef.current?.focus()}>
              {logs.length === 0 && (
                <div className="terminal-welcome">
                  <div className="welcome-title">⚡ Command Interface</div>
                  <div className="welcome-info">
                    <p>Execute commands to interact with Jarvis.</p>
                    <p>Type <code>help</code> for available commands.</p>
                    <p>Use ↑/↓ for history, Ctrl+L to clear.</p>
                  </div>
                </div>
              )}
              {filteredLogs.slice(-50).map((log, i) => (
                <div key={i} className={`log-line ${log.level.toLowerCase()}`}>
                  <span className="log-time">{log.timestamp}</span>
                  <span className="log-level" style={{ color: LOG_LEVEL_COLORS[log.level] || '#888' }}>
                    {log.level}
                  </span>
                  <span className="log-subsystem">[{log.subsystem}]</span>
                  <span className="log-message">{log.message}</span>
                </div>
              ))}
              {loading && (
                <div className="log-line system">
                  <span className="log-message">⏳ Executing...</span>
                </div>
              )}
              {pendingCommand && (
                <div className="command-prompt">
                  <span>Approve command: <code>{pendingCommand}</code>?</span>
                  <button onClick={() => { handleCommand(`approve ${pendingCommand}`); setPendingCommand(null); }}>✓ Approve</button>
                  <button onClick={() => setPendingCommand(null)}>✗ Cancel</button>
                </div>
              )}
            </div>
          </div>
        )}

        {viewMode === 'logs' && (
          <div className="logs-view">
            <div className="logs-controls">
              <div className="log-filters">
                <button 
                  className={`filter-btn ${logFilter === 'all' ? 'active' : ''}`}
                  onClick={() => setLogFilter('all')}
                >
                  All
                </button>
                {LOG_LEVELS.map(level => (
                  <button
                    key={level}
                    className={`filter-btn ${logFilter === level ? 'active' : ''}`}
                    onClick={() => setLogFilter(level)}
                    style={logFilter === level ? { borderColor: LOG_LEVEL_COLORS[level] } : {}}
                  >
                    {level}
                  </button>
                ))}
              </div>
              <input
                type="text"
                className="log-search"
                placeholder="Search logs..."
                value={logSearch}
                onChange={e => setLogSearch(e.target.value)}
              />
            </div>
            <div className="logs-list" ref={terminalRef}>
              {filteredLogs.slice(-100).map((log, i) => (
                <div key={i} className={`log-line ${log.level.toLowerCase()}`}>
                  <span className="log-time">{log.timestamp}</span>
                  <span className="log-level" style={{ color: LOG_LEVEL_COLORS[log.level] || '#888' }}>
                    {log.level}
                  </span>
                  <span className="log-subsystem">[{log.subsystem}]</span>
                  <span className="log-message">{log.message}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {viewMode === 'history' && (
          <div className="history-view">
            <div className="history-list">
              {commandHistory.length === 0 ? (
                <div className="history-empty">
                  <span className="history-empty-icon">🕐</span>
                  <span>No command history yet</span>
                </div>
              ) : (
                commandHistory.map((cmd, i) => (
                  <div 
                    key={i} 
                    className="history-item"
                    onClick={() => handleCommand(cmd)}
                  >
                    <span className="history-index">{i + 1}</span>
                    <span className="history-cmd">{cmd}</span>
                  </div>
                ))
              )}
            </div>
          </div>
        )}
      </div>

      <div className="command-input-area">
        <span className="command-prompt">$</span>
        <input
          ref={inputRef}
          type="text"
          className="command-input"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
          placeholder={pendingCommand ? 'Press Enter to approve, Esc to cancel' : 'Enter command...'}
          autoFocus
        />
      </div>
    </div>
  );
}