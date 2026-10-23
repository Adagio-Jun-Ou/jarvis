import { useState, useEffect, useRef } from 'react';
import './TerminalTab.css';

interface TerminalLine {
  id: string;
  type: 'input' | 'output' | 'error' | 'system';
  content: string;
  timestamp: Date;
}

interface TerminalStatus {
  success: boolean;
  workspace: string;
  mode: string;
  cwd: string;
  sessions: number;
  system: {
    os: string;
    hostname: string;
    python: string;
  };
  resources: {
    cpu: number;
    memory: number;
    disk: number;
  };
}

export function TerminalTab() {
  const [lines, setLines] = useState<TerminalLine[]>([]);
  const [input, setInput] = useState('');
  const [cwd, setCwd] = useState('~');
  const [status, setStatus] = useState<TerminalStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const inputRef = useRef<HTMLInputElement>(null);
  const terminalRef = useRef<HTMLDivElement>(null);
  const pendingCommandRef = useRef<string | null>(null);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [lines]);

  const fetchStatus = async () => {
    try {
      const resp = await fetch('/api/terminal/status');
      const data = await resp.json();
      if (data.success) {
        setStatus(data);
        setCwd(data.cwd || '~');
      }
    } catch (e) {
      console.error('Failed to fetch terminal status:', e);
    }
  };

  const addLine = (type: TerminalLine['type'], content: string) => {
    const id = `line-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    setLines(prev => [...prev, { id, type, content, timestamp: new Date() }]);
  };

  const executeCommand = async (command: string, approved = false) => {
    if (!command.trim()) return;

    setLoading(true);
    addLine('input', `$ ${command}`);
    setInput('');
    setHistory(prev => [...prev, command]);
    setHistoryIndex(-1);

    try {
      const resp = await fetch('/api/terminal/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command, approved }),
      });
      const data = await resp.json();

      if (data.needs_permission) {
        addLine('system', `⚠️ Permission required: ${data.reason}`);
        addLine('system', `Risk level: ${data.risk}`);
        pendingCommandRef.current = command;
      } else if (data.success) {
        if (data.stdout) addLine('output', data.stdout);
        if (data.stderr) addLine('error', data.stderr);
        if (!data.stdout && !data.stderr) addLine('output', '(no output)');
        setCwd(data.cwd || cwd);
      } else {
        addLine('error', data.error || 'Command failed');
      }
    } catch (e) {
      addLine('error', `Failed to execute: ${e}`);
    } finally {
      setLoading(false);
    }
  };

  const approveCommand = () => {
    if (pendingCommandRef.current) {
      const cmd = pendingCommandRef.current;
      pendingCommandRef.current = null;
      executeCommand(cmd, true);
    }
  };

  const cancelCommand = () => {
    pendingCommandRef.current = null;
    addLine('system', 'Command cancelled');
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !loading) {
      e.preventDefault();
      if (pendingCommandRef.current) {
        approveCommand();
      } else {
        executeCommand(input);
      }
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (history.length > 0) {
        const newIndex = historyIndex < history.length - 1 ? historyIndex + 1 : historyIndex;
        setHistoryIndex(newIndex);
        setInput(history[history.length - 1 - newIndex] || '');
      }
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (historyIndex > 0) {
        const newIndex = historyIndex - 1;
        setHistoryIndex(newIndex);
        setInput(history[history.length - 1 - newIndex] || '');
      } else {
        setHistoryIndex(-1);
        setInput('');
      }
    } else if (e.key === 'Escape') {
      if (pendingCommandRef.current) {
        cancelCommand();
      }
    } else if (e.key === 'l' && e.ctrlKey) {
      e.preventDefault();
      setLines([]);
    }
  };

  const handleClear = () => {
    setLines([]);
    addLine('system', 'Terminal cleared');
  };

  const quickCommands = [
    { label: 'ls', cmd: 'ls -la' },
    { label: 'pwd', cmd: 'pwd' },
    { label: 'status', cmd: 'echo "System: $(uname -s)" && echo "Host: $(hostname)"' },
    { label: 'disk', cmd: 'df -h / | tail -1' },
  ];

  return (
    <div className="terminal-tab">
      <div className="terminal-header">
        <div className="terminal-info">
          <span className="terminal-cwd">📁 {cwd}</span>
          {status && (
            <span className="terminal-resources">
              CPU: {status.resources.cpu}% | Mem: {status.resources.memory}% | Disk: {status.resources.disk}%
            </span>
          )}
        </div>
        <div className="terminal-actions">
          <button className="terminal-btn" onClick={handleClear} title="Clear terminal">
            🗑️ Clear
          </button>
          <button className="terminal-btn" onClick={fetchStatus} title="Refresh status">
            🔄 Refresh
          </button>
        </div>
      </div>

      <div className="terminal-quick-cmds">
        {quickCommands.map(qc => (
          <button
            key={qc.label}
            className="quick-cmd-btn"
            onClick={() => executeCommand(qc.cmd)}
            disabled={loading}
          >
            {qc.label}
          </button>
        ))}
      </div>

      <div className="terminal-output" ref={terminalRef} onClick={() => inputRef.current?.focus()}>
        {lines.length === 0 && (
          <div className="terminal-welcome">
            <div className="welcome-title">🖥️ Jarvis Terminal</div>
            <div className="welcome-info">
              <p>Execute shell commands in a sandboxed environment.</p>
              <p>Type commands and press Enter to execute.</p>
              <p>Use ↑/↓ for command history, Ctrl+L to clear.</p>
            </div>
          </div>
        )}
        {lines.map(line => (
          <div key={line.id} className={`terminal-line ${line.type}`}>
            <span className="line-content">{line.content}</span>
          </div>
        ))}
        {loading && (
          <div className="terminal-line system">
            <span className="line-content">⏳ Executing...</span>
          </div>
        )}
        {pendingCommandRef.current && (
          <div className="terminal-prompt-box">
            <span>Approve this command?</span>
            <button className="prompt-btn approve" onClick={approveCommand}>✓ Approve</button>
            <button className="prompt-btn cancel" onClick={cancelCommand}>✗ Cancel</button>
          </div>
        )}
      </div>

      <div className="terminal-input-row">
        <span className="terminal-prompt">$</span>
        <input
          ref={inputRef}
          type="text"
          className="terminal-input"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
          placeholder="Enter command..."
          autoFocus
        />
      </div>

      {status && (
        <div className="terminal-footer">
          <span className="footer-item">🖥️ {status.system.os}</span>
          <span className="footer-item">🐍 Python {status.system.python}</span>
          <span className="footer-item">🔐 Mode: {status.mode}</span>
        </div>
      )}
    </div>
  );
}