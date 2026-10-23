import { useState, useEffect, useMemo, useCallback } from 'react';
import { JarvisAvatar, JarvisMinimalAvatar } from './JarvisAvatar';
import { ActivityFeed } from './ActivityFeed';
import { InsightsCard } from './InsightsCard';
import { MusicCard } from './MusicCard';
import { ButlerPanel } from './ButlerPanel';
import { useDashboard } from '../contexts/DashboardContext';

interface ToolInfo {
  name: string;
  description: string;
  requires_permission: boolean;
}

interface SystemStats {
  tools: number;
  memoryMessages: number;
  model: string;
  backendModel?: string;
  turns: number;
  totalCalls: number;
  totalRequests?: number;
  cacheRate: number;
  avgLatency: number;
  callsPerMinute: number;
  memoryUsedMb: number;
  memoryPercent: number;
  erdosProblems?: number;
  evolutionGenes?: number;
  evolutionProposals?: number;
  learningSessions?: number;
  insightsProcessed?: number;
  hasActivity: boolean;
  dualModel?: {
    enabled: boolean;
    primary: { model: string; connected: boolean; tasks: string[] };
    backend: { model: string; connected: boolean; tasks: string[] };
    crossModelCalls: number;
    totalHandoffs: number;
  };
  freeTimeGranted?: boolean;
  primaryRequests?: number;
  backendRequests?: number;
  parallelRequests?: number;
  collaborativeRequests?: number;
  sequentialRequests?: number;
  primaryRequestsPerMin?: number;
  backendRequestsPerMin?: number;
  totalRequestsPerMin?: number;
  embeddingInterconnect?: {
    storedEmbeddings: number;
    primarySuccesses: number;
    backendSuccesses: number;
    synthesisSuccesses?: number;
    sharedContextKeys: number;
    lastUpdated?: string;
  };
  crossModelCalls?: number;
  errors?: number;
}

interface RefineChange {
  id: string;
  file: string;
  line: number;
  severity: string;
  confidence: number;
  type: string;
  risk: string;
  explanation: string;
  original?: string;
  proposed?: string;
  status: string;
  created?: string;
}

interface RefineStatus {
  active?: boolean;
  running?: boolean;
  state?: string;
  statistics: {
    total: number;
    pending: number;
    applied: number;
    failed: number;
  };
  pending_changes: RefineChange[];
  pending_count: number;
}

interface SubsystemStatus {
  name: string;
  status: 'running' | 'idle' | 'stopped';
  message: string;
  value?: string | number;
  icon: string;
}

interface HardwareStatus {
  cpu: { model: string; cores: number; memory_gb: number };
  gpu: { available: boolean; name: string | null; memory_gb: number };
  npu: { available: boolean; name: string | null };
  primary_accelerator: string;
}

interface MemoryStats {
  conversation: { messages: number; turns: number };
  vector?: { available: boolean; vector_count?: number; memory_mb?: number };
  knowledge_graph?: { available: boolean; nodes?: number; edges?: number; total_nodes?: number; total_edges?: number; communities?: number };
}

interface DashboardTabProps {
  messages?: Array<{ id: string; role: string; content: string; thinking?: string; timestamp: number }>;
  isThinking?: boolean;
}

const SECTIONS = {
  OVERVIEW: 'overview',
  LEARNING: 'learning',
  SELF_REFINE: 'refine',
  MEMORY: 'memory',
  HARDWARE: 'hardware',
} as const;

type SectionKey = typeof SECTIONS[keyof typeof SECTIONS];

export function DashboardTab({ messages = [], isThinking = false }: DashboardTabProps) {
  // Used for activity display in other components
  void messages;
  void isThinking;
  
  const dashboardData = useDashboard();
  
  const [activeSection, setActiveSection] = useState<SectionKey>(SECTIONS.OVERVIEW);
  const [tools, setTools] = useState<ToolInfo[]>([]);
  const [refineStatus, setRefineStatus] = useState<RefineStatus | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [hardware, setHardware] = useState<HardwareStatus | null>(null);
  const [memoryStats, setMemoryStats] = useState<MemoryStats | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [autoRefresh, setAutoRefresh] = useState(true);
  
  const [stats, setStats] = useState<SystemStats>({
    tools: 0,
    memoryMessages: 0,
    model: '-',
    turns: 0,
    totalCalls: 0,
    cacheRate: 0,
    avgLatency: 0,
    callsPerMinute: 0,
    memoryUsedMb: 0,
    memoryPercent: 0,
    hasActivity: false,
  });

  const [subsystems, setSubsystems] = useState<SubsystemStatus[]>([
    { name: 'Erdős Engine', status: 'idle', message: 'Ready', icon: '🧮' },
    { name: 'Learning Pipeline', status: 'idle', message: 'Ready', icon: '📚' },
    { name: 'Evolution', status: 'idle', message: 'Ready', icon: '🧬' },
    { name: 'Daydream', status: 'idle', message: 'Waiting', icon: '💭' },
    { name: 'Self-Refine', status: 'idle', message: 'Ready', icon: '🔧' },
    { name: 'Defense', status: 'idle', message: 'Monitoring', icon: '🛡️' },
    { name: 'Intuition', status: 'idle', message: 'Ready', icon: '💡' },
  ]);

  useEffect(() => {
    if (!dashboardData.data) return;
    
    const statusData = dashboardData.data.status || {};
    const toolsData = dashboardData.data.tools || {};
    
    setTools(toolsData.tools || []);
    setStats({
      tools: statusData.tools || 0,
      memoryMessages: statusData.memory_messages || 0,
      model: statusData.model || '-',
      turns: statusData.turns || 0,
      totalCalls: statusData.total_calls || 0,
      totalRequests: statusData.total_requests || 0,
      cacheRate: statusData.cache_rate || 0,
      avgLatency: statusData.avg_latency || 0,
      callsPerMinute: statusData.calls_per_minute || 0,
      memoryUsedMb: statusData.memory_used_mb || 0,
      memoryPercent: statusData.memory_percent || 0,
      erdosProblems: statusData.erdos_problems || 0,
      evolutionGenes: statusData.evolution_genes || 0,
      evolutionProposals: statusData.evolution_proposals || 0,
      learningSessions: statusData.learning_sessions || 0,
      insightsProcessed: statusData.insights_processed || 0,
      hasActivity: (statusData.turns || 0) > 0,
    });
    
    const dm = dashboardData.data.dual_model || {};
    if (dm.primary_requests !== undefined) {
      setStats(prev => ({
        ...prev,
        primaryRequests: dm.primary_requests || 0,
        backendRequests: dm.backend_requests || 0,
        parallelRequests: dm.parallel_requests || 0,
        collaborativeRequests: dm.collaborative_requests || 0,
        sequentialRequests: dm.sequential_requests || 0,
        primaryRequestsPerMin: dm.primary_requests_per_min || 0,
        backendRequestsPerMin: dm.backend_requests_per_min || 0,
        totalRequestsPerMin: dm.total_requests_per_min || 0,
      }));
    }
    
    const emb = dashboardData.data.embeddings;
    if (emb) {
      setStats(prev => ({
        ...prev,
        embeddingInterconnect: {
          storedEmbeddings: emb.stored_embeddings || 0,
          primarySuccesses: emb.successful_patterns?.primary?.length || 0,
          backendSuccesses: emb.successful_patterns?.backend?.length || 0,
          synthesisSuccesses: emb.successful_patterns?.synthesis?.length || 0,
          sharedContextKeys: emb.shared_context?.total_keys || 0,
        },
      }));
    }
  }, [dashboardData.data]);

  const fetchData = useCallback(async () => {
    try {
      const [refineResp, hardwareResp, memoryResp] = await Promise.all([
        fetch('/api/refine/status').catch(() => null),
        fetch('/api/hardware/status').catch(() => null),
        fetch('/api/memory/stats').catch(() => null),
      ]);

      if (refineResp) {
        const refineData = await refineResp.json();
        if (refineData.success || refineData.active) {
          setRefineStatus(refineData);
          setSubsystems(prev => prev.map(s => 
            s.name === 'Self-Refine' 
              ? { ...s, status: refineData.running ? 'running' : 'idle', message: `${refineData.pending_count || 0} tasks` }
              : s
          ));
        }
      }

      if (hardwareResp) {
        const hardwareData = await hardwareResp.json();
        if (hardwareData.success) {
          setHardware(hardwareData);
        }
      }

      if (memoryResp) {
        const memoryData = await memoryResp.json();
        if (memoryData.success) {
          setMemoryStats(memoryData.stats);
        }
      }
    } catch (e) {
      console.error('Dashboard fetch error:', e);
    }
  }, []);

  const runScan = async () => {
    setIsScanning(true);
    try {
      const resp = await fetch('/api/refine/scan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ include_ai: false }),
      });
      const data = await resp.json();
      if (data.success) {
        fetchData();
      }
    } catch (e) {
      console.error('Scan failed:', e);
    } finally {
      setIsScanning(false);
    }
  };

  const applyChange = async (changeId: string) => {
    try {
      await fetch('/api/refine/apply', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ change_id: changeId }),
      });
      fetchData();
    } catch (e) {
      console.error('Apply failed:', e);
    }
  };

  const rejectChange = async (changeId: string) => {
    try {
      await fetch('/api/refine/reject', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ change_id: changeId }),
      });
      fetchData();
    } catch (e) {
      console.error('Reject failed:', e);
    }
  };

  const filteredTools = useMemo(() => {
    if (!searchQuery) return tools;
    const q = searchQuery.toLowerCase();
    return tools.filter(t => 
      t.name.toLowerCase().includes(q) || 
      t.description.toLowerCase().includes(q)
    );
  }, [tools, searchQuery]);
  
  // Fetch refine, hardware, memory stats on mount
  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000); // Refresh every 10s
    return () => clearInterval(interval);
  }, [fetchData]);

  const getRiskColor = (risk: string) => {
    const colors: Record<string, string> = {
      low: '#22c55e',
      medium: '#f59e0b',
      high: '#ef4444',
      critical: '#dc2626',
    };
    return colors[risk] || '#6b6b70';
  };

  const getSeverityColor = (severity: string) => {
    const colors: Record<string, string> = {
      critical: '#dc2626',
      high: '#ef4444',
      medium: '#f59e0b',
      low: '#22c55e',
      suggestion: '#6b7280',
    };
    return colors[severity] || '#6b7280';
  };

  const formatTime = (iso: string | undefined) => {
    if (!iso) return '';
    const d = new Date(iso);
    return d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  };

  const StatCard = ({ icon, value, label, color }: { icon: string; value: string | number; label: string; color?: string }) => (
    <div className="stat-card">
      <div className="stat-icon" style={{ color: color || 'inherit' }}>{icon}</div>
      <div className="stat-content">
        <div className="stat-value">{value}</div>
        <div className="stat-label">{label}</div>
      </div>
    </div>
  );

  return (
    <div className="dashboard-tab">
      <div className="dashboard-header">
        <div className="header-left">
          <JarvisMinimalAvatar />
          <h2>Dashboard</h2>
        </div>
        <div className="header-controls">
          <span className="model-badge">{stats.model}</span>
          <label className="auto-refresh-toggle">
            <input type="checkbox" checked={autoRefresh} onChange={e => setAutoRefresh(e.target.checked)} />
            Auto-refresh
          </label>
        </div>
      </div>

      {/* Main Hero: Avatar + Live Activity Feed */}
      <div className="jarvis-main-hero">
        {/* Avatar Card - Prominent */}
        <div className="avatar-main-card">
          <div className="avatar-hero-section">
            <div className="avatar-glow-ring-large" />
            <div className="avatar-container-main">
              <JarvisAvatar />
            </div>
            <div className="avatar-status-pill">
              <span className="status-indicator online" />
              <span className="status-text">Online</span>
              <span className="model-name-display">{stats.model}</span>
            </div>
          </div>
          
          <div className="avatar-main-footer">
            {/* Dual Model Status - Prominent */}
            <div className="dual-model-section">
              <div className="dual-model-header">
                <span className="section-label">Bi-Model Architecture</span>
              </div>
              <div className="dual-model-bar">
                <div className="model-segment primary" style={{ width: (stats.primaryRequests || 0) + (stats.backendRequests || 0) > 0 
                  ? `${((stats.primaryRequests || 0) / ((stats.primaryRequests || 0) + (stats.backendRequests || 0))) * 100}%` : '50%' }}>
                  <span className="segment-label">Kimi K2.5</span>
                </div>
                <div className="model-segment backend" style={{ width: (stats.primaryRequests || 0) + (stats.backendRequests || 0) > 0 
                  ? `${((stats.backendRequests || 0) / ((stats.primaryRequests || 0) + (stats.backendRequests || 0))) * 100}%` : '50%' }}>
                  <span className="segment-label">GLM-5</span>
                </div>
              </div>
              <div className="model-cards-row">
                <div className="model-card primary">
                  <div className="model-card-header">
                    <span className="model-dot" style={{ background: '#ec4899' }} />
                    <span className="model-name">Kimi K2.5</span>
                  </div>
                  <div className="model-card-stats">
                    <span className="model-requests">{stats.primaryRequests || 0}</span>
                    <span className="model-rate">{stats.primaryRequestsPerMin || 0}/min</span>
                    <span className="model-tasks">chat • creative</span>
                  </div>
                </div>
                <div className="model-card backend">
                  <div className="model-card-header">
                    <span className="model-dot" style={{ background: '#8b5cf6' }} />
                    <span className="model-name">GLM-5</span>
                  </div>
                  <div className="model-card-stats">
                    <span className="model-requests">{stats.backendRequests || 0}</span>
                    <span className="model-rate">{stats.backendRequestsPerMin || 0}/min</span>
                    <span className="model-tasks">code • math • analysis</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Quick Stats Grid */}
            <div className="quick-stats-grid">
              <div className="quick-stat">
                <span className="quick-stat-icon">⚡</span>
                <div className="quick-stat-info">
                  <span className="quick-stat-value">{stats.callsPerMinute || 0}</span>
                  <span className="quick-stat-label">API/min</span>
                </div>
              </div>
              <div className="quick-stat">
                <span className="quick-stat-icon">💬</span>
                <div className="quick-stat-info">
                  <span className="quick-stat-value">{stats.turns || 0}</span>
                  <span className="quick-stat-label">Turns</span>
                </div>
              </div>
              <div className="quick-stat">
                <span className="quick-stat-icon">🧠</span>
                <div className="quick-stat-info">
                  <span className="quick-stat-value">{stats.memoryUsedMb > 1000 ? `${(stats.memoryUsedMb/1024).toFixed(1)}G` : `${Math.round(stats.memoryUsedMb)}M`}</span>
                  <span className="quick-stat-label">Memory</span>
                </div>
              </div>
              <div className="quick-stat">
                <span className="quick-stat-icon">⏱️</span>
                <div className="quick-stat-info">
                  <span className="quick-stat-value">{stats.avgLatency || 0}<span className="unit">ms</span></span>
                  <span className="quick-stat-label">Latency</span>
                </div>
              </div>
            </div>

            {/* Collaboration Stats */}
            <div className="collab-section">
              <div className="collab-stats-row">
                <div className="collab-stat">
                  <span className="collab-icon">⚡</span>
                  <span className="collab-value">{stats.parallelRequests || 0}</span>
                  <span className="collab-label">Parallel</span>
                </div>
                <div className="collab-stat">
                  <span className="collab-icon">🔗</span>
                  <span className="collab-value">{stats.sequentialRequests || 0}</span>
                  <span className="collab-label">Sequential</span>
                </div>
                <div className="collab-stat">
                  <span className="collab-icon">🤝</span>
                  <span className="collab-value">{stats.collaborativeRequests || 0}</span>
                  <span className="collab-label">Collab</span>
                </div>
              </div>
              
              {/* Embedding Interconnect Stats */}
              <div className="embedding-section">
                <div className="embedding-header">
                  <span className="section-label">🔗 Embedding Interconnect</span>
                  <span className="embedding-dim">768-dim</span>
                </div>
                <div className="embedding-stats-row">
                  <div className="embedding-stat">
                    <span className="embedding-icon">📊</span>
                    <span className="embedding-value">{stats.embeddingInterconnect?.storedEmbeddings ?? 0}</span>
                    <span className="embedding-label">Stored</span>
                  </div>
                  <div className="embedding-stat">
                    <span className="embedding-icon" style={{ color: '#ec4899' }}>🎭</span>
                    <span className="embedding-value">{stats.embeddingInterconnect?.primarySuccesses ?? 0}</span>
                    <span className="embedding-label">Primary</span>
                  </div>
                  <div className="embedding-stat">
                    <span className="embedding-icon" style={{ color: '#8b5cf6' }}>🔮</span>
                    <span className="embedding-value">{stats.embeddingInterconnect?.backendSuccesses ?? 0}</span>
                    <span className="embedding-label">Backend</span>
                  </div>
                  <div className="embedding-stat">
                    <span className="embedding-icon" style={{ color: '#22c55e' }}>🤝</span>
                    <span className="embedding-value">{stats.embeddingInterconnect?.synthesisSuccesses ?? 0}</span>
                    <span className="embedding-label">Synthesis</span>
                  </div>
                  <div className="embedding-stat">
                    <span className="embedding-icon">🔄</span>
                    <span className="embedding-value">{stats.embeddingInterconnect?.sharedContextKeys ?? 0}</span>
                    <span className="embedding-label">Context</span>
                  </div>
                </div>
                {stats.embeddingInterconnect?.storedEmbeddings === 0 && (
                  <div className="embedding-hint">
                    Embeddings accumulate as models process requests
                  </div>
                )}
              </div>
              
              {/* Free Time Toggle */}
              <div className="freetime-toggle-row">
                <label className="freetime-toggle">
                  <input 
                    type="checkbox" 
                    checked={stats.freeTimeGranted || false}
                    onChange={async (e) => {
                      try {
                        await fetch('/api/freetime/permission', {
                          method: 'POST',
                          headers: { 'Content-Type': 'application/json' },
                          body: JSON.stringify({ granted: e.target.checked }),
                        });
                        fetchData();
                      } catch (err) {
                        console.error('Failed to toggle free time:', err);
                      }
                    }}
                  />
                  <span className="toggle-slider"></span>
                  <span className="toggle-label">🧪 Sandbox</span>
                </label>
              </div>
            </div>
          </div>
        </div>
        
        {/* Live Activity Feed - Prominent Position */}
        <div className="live-activity-card">
          <ActivityFeed />
        </div>
      </div>

      {/* Quick Stats Row */}
      <div className="quick-stats-row">
        <StatCard icon="🔧" value={stats.tools} label="Tools" />
        <StatCard icon="💬" value={stats.memoryMessages} label="Messages" />
        <StatCard icon="🧠" value={stats.erdosProblems || 0} label="Erdős" />
        <StatCard icon="🧬" value={stats.evolutionGenes || 0} label="Genes" />
        <StatCard icon="💡" value={stats.insightsProcessed || 0} label="Insights" />
        <StatCard icon="📚" value={stats.learningSessions || 0} label="Sessions" />
        {memoryStats?.vector?.available && (
          <StatCard icon="🔎" value={memoryStats.vector.vector_count || 0} label="Vectors" color="#8b5cf6" />
        )}
        {memoryStats?.knowledge_graph?.available && (
          <StatCard icon="🕸️" value={memoryStats.knowledge_graph.nodes || 0} label="Nodes" color="#06b6d4" />
        )}
      </div>

      <div className="dashboard-nav">
        {[
          { key: SECTIONS.OVERVIEW, label: 'Overview', icon: '📊' },
          { key: SECTIONS.LEARNING, label: 'Learning', icon: '📚' },
          { key: SECTIONS.SELF_REFINE, label: 'Self-Refine', icon: '🔧' },
          { key: SECTIONS.MEMORY, label: 'Memory', icon: '🧠' },
          { key: SECTIONS.HARDWARE, label: 'Hardware', icon: '🖥️' },
        ].map(nav => (
          <button
            key={nav.key}
            className={`nav-btn ${activeSection === nav.key ? 'active' : ''}`}
            onClick={() => setActiveSection(nav.key)}
          >
            {nav.icon} {nav.label}
          </button>
        ))}
      </div>

      <div className="dashboard-content">
        <div className="dashboard-layout">
          <div className="dashboard-main">
            {activeSection === SECTIONS.OVERVIEW && (
              <div className="section overview-section">
                <div className="subsystems-grid">
                  {subsystems.map((sys, i) => (
                    <div key={i} className={`subsystem-card ${sys.status}`}>
                      <div className="subsystem-icon">{sys.icon}</div>
                      <div className="subsystem-info">
                        <div className="subsystem-name">{sys.name}</div>
                        <div className="subsystem-message">{sys.message}</div>
                      </div>
                      <div className={`subsystem-status-dot ${sys.status}`} />
                    </div>
                  ))}
                </div>

                <div className="tools-section">
                  <div className="section-header">
                    <h3>🔧 Tools ({tools.length})</h3>
                    <input
                      type="text"
                      className="search-input"
                      placeholder="Search tools..."
                      value={searchQuery}
                      onChange={e => setSearchQuery(e.target.value)}
                    />
                  </div>
                  <div className="tools-grid">
                    {filteredTools.slice(0, 20).map((tool, i) => (
                      <div key={i} className="tool-item">
                        <span className="tool-name">{tool.name}</span>
                        {tool.requires_permission && <span className="perm-badge">⚠️</span>}
                        <span className="tool-desc">{tool.description.slice(0, 60)}...</span>
                      </div>
                    ))}
                    {filteredTools.length > 20 && (
                      <div className="tool-more">+{filteredTools.length - 20} more</div>
                    )}
                  </div>
                </div>

                <div className="insights-section">
                  <InsightsCard maxInsights={5} />
                </div>

                 <div className="music-section">
                   <MusicCard />
                 </div>

                 <div className="butler-section">
                   <ButlerPanel />
                 </div>
              </div>
            )}

            {activeSection === SECTIONS.LEARNING && (
              <div className="section learning-section">
                <div className="learning-stats">
                  <div className="learning-card">
                    <span className="learning-icon">🧮</span>
                    <div className="learning-info">
                      <span className="learning-value">{stats.erdosProblems?.toLocaleString() || 0}</span>
                      <span className="learning-label">Problems Solved</span>
                    </div>
                  </div>
                  <div className="learning-card">
                    <span className="learning-icon">🧬</span>
                    <div className="learning-info">
                      <span className="learning-value">{stats.evolutionGenes || 0}</span>
                      <span className="learning-label">Evolution Genes</span>
                    </div>
                  </div>
                  <div className="learning-card">
                    <span className="learning-icon">💡</span>
                    <div className="learning-info">
                      <span className="learning-value">{stats.insightsProcessed || 0}</span>
                      <span className="learning-label">Insights</span>
                    </div>
                  </div>
                  <div className="learning-card">
                    <span className="learning-icon">📚</span>
                    <div className="learning-info">
                      <span className="learning-value">{stats.learningSessions || 0}</span>
                      <span className="learning-label">Sessions</span>
                    </div>
                  </div>
                </div>
                <div className="learning-description">
                  <p>The learning system continuously improves through multiple subsystems:</p>
                  <ul>
                    <li><strong>Erdős Engine</strong> — Solves math problems to improve reasoning</li>
                    <li><strong>Evolution</strong> — Adapts parameters based on outcomes</li>
                    <li><strong>Daydream</strong> — Explores novel solutions when idle</li>
                    <li><strong>Intuition Bridge</strong> — Captures and routes insights</li>
                  </ul>
                </div>
              </div>
            )}

            {activeSection === SECTIONS.SELF_REFINE && (
              <div className="section refine-section">
                {refineStatus ? (
                  <>
                    <div className="refine-header">
                      <div className="refine-status">
                        <span className={`status-indicator ${refineStatus.running ? 'running' : 'idle'}`}>
                          {refineStatus.running ? '● Running' : '○ Idle'}
                        </span>
                        <span className="state-badge">{refineStatus.state || 'idle'}</span>
                      </div>
                      <div className="refine-stats">
                        <div className="refine-stat">
                          <span className="value">{refineStatus.statistics.pending}</span>
                          <span className="label">Pending</span>
                        </div>
                        <div className="refine-stat">
                          <span className="value">{refineStatus.statistics.applied}</span>
                          <span className="label">Applied</span>
                        </div>
                        <div className="refine-stat">
                          <span className="value">{refineStatus.statistics.failed}</span>
                          <span className="label">Failed</span>
                        </div>
                      </div>
                    </div>

                    <div className="refine-actions">
                      <button className="scan-btn" onClick={runScan} disabled={isScanning}>
                        {isScanning ? '⏳ Scanning...' : '⚡ Scan Codebase'}
                      </button>
                    </div>

                    <div className="pending-changes">
                      <h4>Pending Changes ({refineStatus.pending_count})</h4>
                      {refineStatus.pending_changes && refineStatus.pending_changes.length > 0 ? (
                        <div className="changes-list">
                          {refineStatus.pending_changes.slice(0, 10).map((change, i) => (
                            <div key={i} className="change-card">
                              <div className="change-header">
                                <span className="change-file">{change.file}:{change.line}</span>
                                <span className="change-severity" style={{ color: getSeverityColor(change.severity) }}>
                                  {change.severity}
                                </span>
                                <span className="change-type">{change.type}</span>
                                <span className="change-risk" style={{ color: getRiskColor(change.risk) }}>
                                  {change.risk}
                                </span>
                                {change.created && <span className="change-time">{formatTime(change.created)}</span>}
                              </div>
                              <div className="change-meta">
                                {change.confidence > 0 && (
                                  <span className="change-confidence">
                                    Confidence: {Math.round(change.confidence * 100)}%
                                  </span>
                                )}
                              </div>
                              
                              <div className="change-section">
                                <div className="change-section-label">Why</div>
                                <div className="change-section-content">{change.explanation || 'No explanation provided'}</div>
                              </div>
                              
                              <div className="change-section">
                                <div className="change-section-label">What</div>
                                <div className="change-section-content">
                                  <span className="change-detail">File: </span>
                                  <code>{change.file}</code>
                                  {change.line > 0 && <><span className="change-detail"> at line </span><code>{change.line}</code></>}
                                  <br />
                                  <span className="change-detail">Type: </span>
                                  <span>{change.type === 'edit' ? 'Modify existing code' : change.type}</span>
                                </div>
                              </div>
                              
                              {(change.original || change.proposed) && (
                                <div className="change-section">
                                  <div className="change-section-label">How (Diff)</div>
                                  <div className="change-diff">
                                    {change.original && (
                                      <div className="diff-section">
                                        <span className="diff-label">- Original</span>
                                        <code className="diff-code removed">{change.original}</code>
                                      </div>
                                    )}
                                    {change.proposed && (
                                      <div className="diff-section">
                                        <span className="diff-label">+ Proposed</span>
                                        <code className="diff-code added">{change.proposed}</code>
                                      </div>
                                    )}
                                  </div>
                                </div>
                              )}
                              
                              <div className="change-actions">
                                <button className="reject-btn" onClick={() => rejectChange(change.id)}>
                                  Reject
                                </button>
                                <button className="apply-btn" onClick={() => applyChange(change.id)}>
                                  Apply
                                </button>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="empty-state">No pending changes. Run a scan to find improvements.</div>
                      )}
                    </div>
                  </>
                ) : (
                  <div className="empty-state">
                    <p>Self-refinement system not initialized.</p>
                    <button className="scan-btn" onClick={runScan} disabled={isScanning}>
                      {isScanning ? '⏳ Starting...' : 'Initialize'}
                    </button>
                  </div>
                )}
              </div>
            )}

            {activeSection === SECTIONS.MEMORY && (
              <div className="section memory-section">
                {memoryStats ? (
                  <div className="memory-grid">
                    <div className="memory-card">
                      <h4>Conversation</h4>
                      <div className="memory-stats">
                        <div className="mem-stat">
                          <span className="value">{memoryStats.conversation?.messages || 0}</span>
                          <span className="label">Messages</span>
                        </div>
                        <div className="mem-stat">
                          <span className="value">{memoryStats.conversation?.turns || 0}</span>
                          <span className="label">Turns</span>
                        </div>
                      </div>
                    </div>
                    {memoryStats.vector?.available && (
                      <div className="memory-card">
                        <h4>Vector Index</h4>
                        <div className="memory-stats">
                          <div className="mem-stat">
                            <span className="value">{memoryStats.vector.vector_count || 0}</span>
                            <span className="label">Vectors</span>
                          </div>
                          <div className="mem-stat">
                            <span className="value">{memoryStats.vector.memory_mb?.toFixed(1) || 0}MB</span>
                            <span className="label">Memory</span>
                          </div>
                        </div>
                      </div>
                    )}
                    {memoryStats.knowledge_graph?.available && (
                      <div className="memory-card">
                        <h4>Knowledge Graph</h4>
                        <div className="memory-stats">
                          <div className="mem-stat">
                            <span className="value">{memoryStats.knowledge_graph.total_nodes || memoryStats.knowledge_graph.nodes || 0}</span>
                            <span className="label">Nodes</span>
                          </div>
                          <div className="mem-stat">
                            <span className="value">{memoryStats.knowledge_graph.total_edges || memoryStats.knowledge_graph.edges || 0}</span>
                            <span className="label">Edges</span>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="empty-state">Memory statistics not available</div>
                )}
              </div>
            )}

            {activeSection === SECTIONS.HARDWARE && (
              <div className="section hardware-section">
                {hardware ? (
                  <div className="hardware-grid">
                    <div className="hardware-card">
                      <div className="hw-icon">CPU</div>
                      <div className="hw-info">
                        <div className="hw-name">{hardware.cpu.cores} Cores</div>
                        <div className="hw-detail">{hardware.cpu.memory_gb}GB RAM</div>
                      </div>
                    </div>
                    <div className={`hardware-card ${hardware.gpu.available ? '' : 'unavailable'}`}>
                      <div className="hw-icon">GPU</div>
                      <div className="hw-info">
                        <div className="hw-name">{hardware.gpu.available ? hardware.gpu.name || 'Available' : 'N/A'}</div>
                        <div className="hw-detail">{hardware.gpu.available ? `${hardware.gpu.memory_gb}GB` : 'CPU fallback'}</div>
                      </div>
                    </div>
                    <div className={`hardware-card ${hardware.npu.available ? '' : 'unavailable'}`}>
                      <div className="hw-icon">NPU</div>
                      <div className="hw-info">
                        <div className="hw-name">{hardware.npu.available ? hardware.npu.name || 'Available' : 'N/A'}</div>
                        <div className="hw-detail">{hardware.npu.available ? 'Accelerated' : 'Not detected'}</div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="empty-state">Hardware information not available</div>
                )}
                <div className="accelerator-badge">
                  Primary Accelerator: <strong>{hardware?.primary_accelerator?.toUpperCase() || 'CPU'}</strong>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}