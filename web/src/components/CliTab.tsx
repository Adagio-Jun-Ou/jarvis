import { useState, useEffect, useRef } from 'react';

interface LogEntry {
  timestamp: string;
  level: string;
  subsystem: string;
  message: string;
}

export function CliTab() {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [stats, setStats] = useState({
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
  const terminalRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const fetchLogs = async () => {
      try {
        const [logsResp, statsResp] = await Promise.all([
          fetch('/api/subsystem_logs'),
          fetch('/api/cli-stats'),
        ]);
        
        const logsData = await logsResp.json();
        const statsData = await statsResp.json();
        
        // Convert subsystem logs to terminal format
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
          setLogs(prev => [...prev, ...newLogs].slice(-200));
        }
        
        setStats({
          total: statsData.total_calls || 0,
          primary: statsData.primary_calls || 0,
          backend: statsData.backend_calls || 0,
          cached: statsData.cache_hits || 0,
          tokens: statsData.total_tokens || 0,
          latency: statsData.avg_latency || 0,
        });
      } catch (e) {
        // Add error log
        setLogs(prev => [...prev, {
          timestamp: new Date().toISOString().split('T')[1].split('.')[0],
          level: 'ERROR',
          subsystem: 'WEB',
          message: 'Failed to fetch logs',
        }].slice(-200));
      }
    };

    fetchLogs();
    const interval = setInterval(fetchLogs, 2000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [logs]);

  const handleCommand = async (cmd: string) => {
    const trimmed = cmd.trim();
    if (!trimmed) return;

    // Add command to log
    setLogs(prev => [...prev, {
      timestamp: new Date().toISOString().split('T')[1].split('.')[0],
      level: 'INPUT',
      subsystem: 'USER',
      message: `> ${trimmed}`,
    }]);

    setCommandHistory(prev => [trimmed, ...prev.slice(0, 49)]);
    setHistoryIndex(-1);
    setInput('');

    // Process command
    try {
      const resp = await fetch('/api/command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command: trimmed }),
      });
      
      if (resp.ok) {
        const data = await resp.json();
        setLogs(prev => [...prev, {
          timestamp: new Date().toISOString().split('T')[1].split('.')[0],
          level: 'OUTPUT',
          subsystem: 'JARVIS',
          message: data.response || JSON.stringify(data, null, 2),
        }]);
      } else {
        setLogs(prev => [...prev, {
          timestamp: new Date().toISOString().split('T')[1].split('.')[0],
          level: 'ERROR',
          subsystem: 'SYSTEM',
          message: `Command failed: ${resp.status}`,
        }]);
      }
    } catch (e) {
      setLogs(prev => [...prev, {
        timestamp: new Date().toISOString().split('T')[1].split('.')[0],
        level: 'ERROR',
        subsystem: 'SYSTEM',
        message: 'Connection error',
      }]);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleCommand(input);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (historyIndex < commandHistory.length - 1) {
        const newIndex = historyIndex + 1;
        setHistoryIndex(newIndex);
        setInput(commandHistory[newIndex]);
      }
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (historyIndex > 0) {
        const newIndex = historyIndex - 1;
        setHistoryIndex(newIndex);
        setInput(commandHistory[newIndex]);
      } else if (historyIndex === 0) {
        setHistoryIndex(-1);
        setInput('');
      }
    }
  };

  const getLevelClass = (level: string) => {
    switch (level) {
      case 'ERROR': return 'log-error';
      case 'WARNING': return 'log-warning';
      case 'INPUT': return 'log-input';
      case 'OUTPUT': return 'log-output';
      default: return 'log-info';
    }
  };

  return (
    <div className="cli-tab">
      <div className="cli-stats-bar">
        <span className="stat">Calls: {stats.total}</span>
        <span className="stat">Primary: {stats.primary}</span>
        <span className="stat">Backend: {stats.backend}</span>
        <span className="stat">Cached: {stats.cached}</span>
        <span className="stat">Tokens: {stats.tokens}</span>
        <span className="stat">Latency: {stats.latency}ms</span>
      </div>
      <div className="cli-terminal" ref={terminalRef} onClick={() => inputRef.current?.focus()}>
        {logs.length === 0 ? (
          <div className="cli-welcome">
            <div>Jarvis CLI v3.0</div>
            <div>Type 'help' for available commands</div>
          </div>
        ) : (
          logs.map((log, i) => (
            <div key={i} className={`cli-line ${getLevelClass(log.level)}`}>
              <span className="log-time">[{log.timestamp}]</span>
              <span className="log-subsystem">[{log.subsystem}]</span>
              <span className="log-message">{log.message}</span>
            </div>
          ))
        )}
      </div>
      <div className="cli-input-line">
        <span className="cli-prompt">$</span>
        <input
          ref={inputRef}
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Enter command..."
          className="cli-input"
          autoFocus
        />
      </div>
    </div>
  );
}