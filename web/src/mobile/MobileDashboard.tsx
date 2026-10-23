import React, { useState, useEffect } from 'react';

interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
}

interface ToolCall {
  id: string;
  name: string;
  status: 'running' | 'completed' | 'failed';
  result?: string;
  timestamp: number;
}

interface SubsystemStatus {
  name: string;
  status: 'running' | 'idle' | 'stopped';
  message: string;
  icon: string;
}

interface Props {
  messages: Message[];
  toolCalls: ToolCall[];
  isThinking: boolean;
}

export const MobileDashboard: React.FC<Props> = ({ messages, toolCalls, isThinking }) => {
  const [subsystems, setSubsystems] = useState<SubsystemStatus[]>([]);
  const [modelInfo, setModelInfo] = useState<{ primary: string; backend: string }>({ primary: 'Loading...', backend: '' });
  const [stats, setStats] = useState({
    erdos_problems: 0,
    evolution_genes: 0,
    insights_processed: 0,
    memory_percent: 0,
    daydream_count: 0,
    is_idle: true,
    last_activity: '',
    primary_requests: 0,
    backend_requests: 0,
    parallel_requests: 0,
    sequential_requests: 0,
    collaborative_requests: 0,
    embedding_count: 0,
    shared_context_keys: 0,
  });

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const resp = await fetch('/api/status');
        const data = await resp.json();
        
        const subs: SubsystemStatus[] = [];
        
        if (data.daydream) {
          subs.push({
            name: 'Daydream Engine',
            status: data.daydream.running ? 'running' : 'idle',
            message: data.daydream.running 
              ? `Exploring (${data.daydream.daydream_count || 0} sessions)` 
              : `Idle (${data.daydream.daydream_count || 0} sessions)`,
            icon: '💭',
          });
        }
        
        if (data.erdos_problems !== undefined) {
          subs.push({
            name: 'Erdős Engine',
            status: 'running',
            message: `${data.erdos_problems.toLocaleString()} problems explored`,
            icon: '🔢',
          });
        }
        
        if (data.evolution_genes !== undefined) {
          subs.push({
            name: 'Evolution',
            status: data.evolution_genes > 0 ? 'running' : 'idle',
            message: `${data.evolution_genes} genes, ${data.evolution_proposals || 0} proposals`,
            icon: '🧬',
          });
        }
        
        if (data.memory) {
          subs.push({
            name: 'Memory',
            status: data.memory.pressure === 'normal' ? 'running' : 'idle',
            message: `${data.memory.process_mb?.toFixed(0)}MB (${data.memory.ram_percent}% RAM)`,
            icon: '💾',
          });
        }
        
        subs.push({
          name: 'Tools Available',
          status: 'running',
          message: `${data.tools || 0} tools registered`,
          icon: '🔧',
        });
        
        if (data.insights_processed !== undefined) {
          subs.push({
            name: 'Insights',
            status: 'running',
            message: `${data.insights_processed} processed`,
            icon: '💡',
          });
        }
        
        setSubsystems(subs);
        
        setModelInfo({
          primary: data.model || data.models?.primary?.model || 'Unknown',
          backend: data.backend_model || data.models?.backend?.model || '',
        });
        
        setStats({
          erdos_problems: data.erdos_problems || 0,
          evolution_genes: data.evolution_genes || 0,
          insights_processed: data.insights_processed || 0,
          memory_percent: data.memory?.ram_percent || 0,
          daydream_count: data.daydream?.daydream_count || 0,
          is_idle: data.is_idle ?? true,
          last_activity: data.last_activity || '',
          primary_requests: data.primary_requests || 0,
          backend_requests: data.backend_requests || 0,
          parallel_requests: data.parallel_requests || 0,
          sequential_requests: data.sequential_requests || 0,
          collaborative_requests: data.collaborative_requests || 0,
          embedding_count: data.embedding_count || 0,
          shared_context_keys: data.shared_context_keys || 0,
        });
        
        // Fetch dual-model stats
        try {
          const dualResp = await fetch('/api/dual-model/status');
          if (dualResp.ok) {
            const dualData = await dualResp.json();
            if (dualData.success) {
              setStats(prev => ({
                ...prev,
                primary_requests: dualData.primary_requests || 0,
                backend_requests: dualData.backend_requests || 0,
                parallel_requests: dualData.parallel_requests || 0,
                sequential_requests: dualData.sequential_requests || 0,
                collaborative_requests: dualData.collaborative_requests || 0,
              }));
            }
          }
        } catch {}
        
        // Fetch embedding stats
        try {
          const embResp = await fetch('/api/dual-model/embeddings');
          if (embResp.ok) {
            const embData = await embResp.json();
            if (embData.success) {
              setStats(prev => ({
                ...prev,
                embedding_count: embData.stored_embeddings || 0,
                shared_context_keys: embData.shared_context?.total_keys || 0,
              }));
            }
          }
        } catch {}
        
      } catch (e) {
        console.error('Failed to fetch status:', e);
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 3000);
    return () => clearInterval(interval);
  }, []);

  const userMessages = messages.filter(m => m.role === 'user').length;
  const assistantMessages = messages.filter(m => m.role === 'assistant').length;
  const totalTokens = messages.reduce((sum, m) => sum + Math.ceil(m.content.length / 4), 0);
  const toolSuccess = toolCalls.filter(t => t.status === 'completed').length;

  return (
    <div className="dashboard-view">
      <div className="dashboard-header">
        <h2>📊 Dashboard</h2>
        <div className="header-status">
          {isThinking && <span className="thinking-indicator">💭 Thinking...</span>}
          <span className={`idle-status ${stats.is_idle ? 'idle' : 'active'}`}>
            {stats.is_idle ? '🟢 Idle' : '🔵 Active'}
          </span>
        </div>
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-icon">💬</div>
          <div className="stat-content">
            <span className="stat-value">{userMessages + assistantMessages}</span>
            <span className="stat-label">Messages</span>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon">🔧</div>
          <div className="stat-content">
            <span className="stat-value">{toolSuccess}</span>
            <span className="stat-label">Tools Used</span>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon">🧠</div>
          <div className="stat-content">
            <span className="stat-value">{totalTokens}</span>
            <span className="stat-label">Est. Tokens</span>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon">💡</div>
          <div className="stat-content">
            <span className="stat-value">{stats.insights_processed}</span>
            <span className="stat-label">Insights</span>
          </div>
        </div>
      </div>

      <div className="dashboard-section">
        <h3 className="section-title">🤖 Current Models</h3>
        <div className="models-grid">
          <div className="model-card primary">
            <span className="model-tier">Primary</span>
            <span className="model-name">{modelInfo.primary}</span>
            <span className="model-role">Chat & Interactive</span>
            <span className="model-requests">{stats.primary_requests || 0} calls</span>
          </div>
          {modelInfo.backend && (
            <div className="model-card backend">
              <span className="model-tier">Backend</span>
              <span className="model-name">{modelInfo.backend}</span>
              <span className="model-role">Code & Analysis</span>
              <span className="model-requests">{stats.backend_requests || 0} calls</span>
            </div>
          )}
        </div>
      </div>

      {/* Collaboration Mode Stats */}
      <div className="dashboard-section collaboration">
        <h3 className="section-title">🤝 Collaboration</h3>
        <div className="collab-stats-grid">
          <div className="collab-stat">
            <span className="collab-icon">⚡</span>
            <span className="collab-value">{stats.parallel_requests || 0}</span>
            <span className="collab-label">Parallel</span>
          </div>
          <div className="collab-stat">
            <span className="collab-icon">🔗</span>
            <span className="collab-value">{stats.sequential_requests || 0}</span>
            <span className="collab-label">Sequential</span>
          </div>
          <div className="collab-stat">
            <span className="collab-icon">🤝</span>
            <span className="collab-value">{stats.collaborative_requests || 0}</span>
            <span className="collab-label">Collab</span>
          </div>
        </div>
      </div>

      {/* Embedding Interconnect */}
      <div className="dashboard-section embedding">
        <h3 className="section-title">🔗 Interconnect</h3>
        <div className="embedding-stats-grid">
          <div className="embedding-stat">
            <span className="embedding-icon">📊</span>
            <span className="embedding-value">{stats.embedding_count || 0}</span>
            <span className="embedding-label">Embeddings</span>
          </div>
          <div className="embedding-stat">
            <span className="embedding-icon">🔄</span>
            <span className="embedding-value">{stats.shared_context_keys || 0}</span>
            <span className="embedding-label">Context Keys</span>
          </div>
        </div>
      </div>

      <div className="dashboard-section">
        <h3 className="section-title">⚙️ Subsystems</h3>
        <div className="subsystems-list">
          {subsystems.map((sub, index) => (
            <div key={index} className={`subsystem-row ${sub.status}`}>
              <div className="subsystem-icon">{sub.icon}</div>
              <div className="subsystem-info">
                <span className="subsystem-name">{sub.name}</span>
                <span className="subsystem-message">{sub.message}</span>
              </div>
              <span className={`subsystem-status ${sub.status}`}>
                {sub.status === 'running' ? '●' : '○'}
              </span>
            </div>
          ))}
        </div>
      </div>

      <div className="dashboard-section quick-stats">
        <h3 className="section-title">📈 Quick Stats</h3>
        <div className="quick-stats-grid">
          <div className="quick-stat">
            <span className="quick-stat-value">{stats.erdos_problems.toLocaleString()}</span>
            <span className="quick-stat-label">Problems</span>
          </div>
          <div className="quick-stat">
            <span className="quick-stat-value">{stats.evolution_genes}</span>
            <span className="quick-stat-label">Genes</span>
          </div>
          <div className="quick-stat">
            <span className="quick-stat-value">{stats.daydream_count}</span>
            <span className="quick-stat-label">Daydreams</span>
          </div>
          <div className="quick-stat">
            <span className="quick-stat-value">{stats.memory_percent}%</span>
            <span className="quick-stat-label">RAM</span>
          </div>
        </div>
      </div>

      {toolCalls.length > 0 && (
        <div className="dashboard-section">
          <h3 className="section-title">📝 Recent Tool Activity</h3>
          <div className="activity-list">
            {toolCalls.slice(-5).reverse().map(tool => (
              <div key={tool.id} className="activity-row">
                <span className="activity-icon">
                  {tool.status === 'completed' ? '✅' : tool.status === 'running' ? '⚡' : '❌'}
                </span>
                <span className="activity-name">{tool.name}</span>
                <span className="activity-time">
                  {new Date(tool.timestamp).toLocaleTimeString()}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {stats.last_activity && (
        <div className="last-activity">
          Last activity: {new Date(stats.last_activity).toLocaleTimeString()}
        </div>
      )}
    </div>
  );
};

export default MobileDashboard;