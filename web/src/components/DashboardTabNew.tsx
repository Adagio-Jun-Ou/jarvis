import { useState, useEffect, useCallback } from 'react';
import { JarvisAvatar } from './JarvisAvatar';
import { ActivityFeed } from './ActivityFeed';
import { InsightsCard } from './InsightsCard';

interface SystemStats {
  model: string;
  backendModel?: string;
  turns: number;
  callsPerMinute: number;
  memoryUsedMb: number;
  memoryPercent: number;
  primaryRequests?: number;
  backendRequests?: number;
  parallelRequests?: number;
  collaborativeRequests?: number;
  embeddingInterconnect?: {
    storedEmbeddings: number;
    primarySuccesses: number;
    backendSuccesses: number;
    synthesisSuccesses?: number;
    sharedContextKeys: number;
  };
}

export function DashboardTabNew() {
  const [stats, setStats] = useState<SystemStats>({
    model: 'kimi-k2.5',
    turns: 0,
    callsPerMinute: 0,
    memoryUsedMb: 0,
    memoryPercent: 0,
  });
  const [autoRefresh, setAutoRefresh] = useState(true);

  const fetchData = useCallback(async () => {
    try {
      const statusResp = await fetch('/api/status').catch(() => null);

      if (statusResp) {
        const data = await statusResp.json();
        setStats({
          model: data.model || 'kimi-k2.5',
          backendModel: data.backend_model,
          turns: data.turns || 0,
          callsPerMinute: data.calls_per_minute || 0,
          memoryUsedMb: data.memory_used_mb || 0,
          memoryPercent: data.memory_percent || 0,
        });
      }

      const orchResp = await fetch('/api/dual-model/status').catch(() => null);
      if (orchResp) {
        const orchData = await orchResp.json();
        if (orchData.success) {
          setStats(prev => ({
            ...prev,
            primaryRequests: orchData.primary_requests || 0,
            backendRequests: orchData.backend_requests || 0,
            parallelRequests: orchData.parallel_requests || 0,
            collaborativeRequests: orchData.collaborative_requests || 0,
          }));
        }
      }

      const embResp = await fetch('/api/dual-model/embeddings').catch(() => null);
      if (embResp) {
        const embData = await embResp.json();
        if (embData.success) {
          setStats(prev => ({
            ...prev,
            embeddingInterconnect: {
              storedEmbeddings: embData.stored_embeddings || 0,
              primarySuccesses: embData.successful_patterns?.primary?.length || 0,
              backendSuccesses: embData.successful_patterns?.backend?.length || 0,
              synthesisSuccesses: embData.successful_patterns?.synthesis?.length || 0,
              sharedContextKeys: embData.shared_context?.total_keys || 0,
            },
          }));
        }
      }
    } catch (e) {
      console.error('Dashboard fetch error:', e);
    }
  }, []);

  useEffect(() => {
    fetchData();
    if (autoRefresh) {
      const interval = setInterval(fetchData, 5000);
      return () => clearInterval(interval);
    }
  }, [fetchData, autoRefresh]);

  return (
    <div className="dashboard-new">
      <div className="dashboard-header-new">
        <h2>Dashboard</h2>
        <label className="auto-refresh-toggle">
          <input type="checkbox" checked={autoRefresh} onChange={e => setAutoRefresh(e.target.checked)} />
          Auto-refresh
        </label>
      </div>

      <div className="dashboard-grid">
        <div className="dashboard-status-section">
          <div className="status-card avatar-card">
            <JarvisAvatar />
            <div className="status-info">
              <div className="status-label">Status</div>
              <div className="status-value online">Online</div>
              <div className="model-display">{stats.model}</div>
            </div>
          </div>

          <div className="status-card stats-card">
            <div className="stats-grid">
              <div className="stat-box">
                <span className="stat-value">{stats.callsPerMinute}</span>
                <span className="stat-label">API/min</span>
              </div>
              <div className="stat-box">
                <span className="stat-value">{stats.turns}</span>
                <span className="stat-label">Turns</span>
              </div>
              <div className="stat-box">
                <span className="stat-value">
                  {stats.memoryUsedMb > 1000 
                    ? `${(stats.memoryUsedMb / 1024).toFixed(1)}G` 
                    : `${Math.round(stats.memoryUsedMb)}M`}
                </span>
                <span className="stat-label">Memory</span>
              </div>
              <div className="stat-box">
                <span className="stat-value">{stats.memoryPercent.toFixed(0)}%</span>
                <span className="stat-label">RAM</span>
              </div>
            </div>
          </div>
        </div>

        <div className="dashboard-model-section">
          <div className="model-card">
            <div className="model-header">
              <span className="model-title">Bi-Model Architecture</span>
              <span className="model-badge active">Active</span>
            </div>
            
            <div className="model-bar">
              <div className="model-segment primary" style={{ 
                width: ((stats.primaryRequests || 0) + (stats.backendRequests || 0)) > 0 
                  ? `${((stats.primaryRequests || 0) / ((stats.primaryRequests || 0) + (stats.backendRequests || 0))) * 100}%` 
                  : '50%'
              }}>
                <span className="segment-label">Kimi</span>
              </div>
              <div className="model-segment backend" style={{ 
                width: ((stats.primaryRequests || 0) + (stats.backendRequests || 0)) > 0 
                  ? `${((stats.backendRequests || 0) / ((stats.primaryRequests || 0) + (stats.backendRequests || 0))) * 100}%` 
                  : '50%'
              }}>
                <span className="segment-label">GLM-5</span>
              </div>
            </div>

            <div className="model-stats-row">
              <div className="model-stat primary">
                <span className="model-stat-label">🌸 Primary</span>
                <span className="model-stat-value">{stats.primaryRequests || 0}</span>
                <span className="model-stat-tasks">chat • creative</span>
              </div>
              <div className="model-stat backend">
                <span className="model-stat-label">🧠 Backend</span>
                <span className="model-stat-value">{stats.backendRequests || 0}</span>
                <span className="model-stat-tasks">code • math</span>
              </div>
            </div>

            <div className="collab-stats">
              <div className="collab-item">
                <span className="collab-value">{stats.parallelRequests || 0}</span>
                <span className="collab-label">Parallel</span>
              </div>
              <div className="collab-item">
                <span className="collab-value">{stats.collaborativeRequests || 0}</span>
                <span className="collab-label">Collab</span>
              </div>
            </div>
          </div>

          <div className="embedding-card">
            <div className="embedding-header">
              <span>🔗 Embedding Interconnect</span>
              <span className="embedding-dim">768-dim Jina v5</span>
            </div>
            <div className="embedding-stats">
              <div className="embedding-stat">
                <span className="embedding-value">{stats.embeddingInterconnect?.storedEmbeddings || 0}</span>
                <span className="embedding-label">Stored</span>
              </div>
              <div className="embedding-stat">
                <span className="embedding-value" style={{ color: '#ec4899' }}>{stats.embeddingInterconnect?.primarySuccesses || 0}</span>
                <span className="embedding-label">Primary</span>
              </div>
              <div className="embedding-stat">
                <span className="embedding-value" style={{ color: '#8b5cf6' }}>{stats.embeddingInterconnect?.backendSuccesses || 0}</span>
                <span className="embedding-label">Backend</span>
              </div>
              <div className="embedding-stat">
                <span className="embedding-value" style={{ color: '#22c55e' }}>{stats.embeddingInterconnect?.synthesisSuccesses || 0}</span>
                <span className="embedding-label">Synthesis</span>
              </div>
            </div>
          </div>
        </div>

        <div className="dashboard-activity-section">
          <ActivityFeed />
        </div>

        <div className="dashboard-insights-section">
          <InsightsCard maxInsights={5} />
        </div>
      </div>
    </div>
  );
}