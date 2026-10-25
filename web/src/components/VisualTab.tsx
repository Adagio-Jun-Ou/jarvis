import { useState, useEffect, useRef, useMemo } from 'react';
import { useDashboard } from '../contexts/DashboardContext';
import { FlowVisualization } from './FlowVisualization';
import './VisualTab.css';

interface ActivityLog {
  timestamp: number;
  subsystem: string;
  event: string;
  details: string;
  type: 'info' | 'success' | 'warning' | 'error' | 'thinking';
  latency?: number;
}

interface Insight {
  id: string;
  source: string;
  type: string;
  category: string;
  content: string;
  summary: string;
  timestamp: string;
  relevance: number;
}

interface VectorEntry {
  id: string;
  content: string;
  text?: string;
  timestamp: string;
  importance?: number;
  access_count?: number;
  metadata: Record<string, any>;
}

interface KnowledgeNode {
  id: string;
  type: string;
  content: string;
  connections: number;
  pagerank: number;
  community_id?: number;
}

interface KnowledgeEdge {
  source: string;
  target: string;
  type: string;
  weight: number;
}

const SUBSYSTEM_ICONS: Record<string, string> = {
  'Erdős Engine': '🔢', 'Daydream': '💭', 'Evolution': '🧬', 'Self-Refine': '🔧',
  'Insight Bridge': '🌉', 'Intuition': '💡', 'Learning': '📚', 'Proactive': '🎯',
  'Defense': '🛡️', 'Free Time': '🏖️', 'Dual Model': '🤝', 'API': '📡',
  'Memory': '💾', 'Knowledge': '🕸️', 'Vector': '🔎', 'Context': '📝',
};

const TYPE_COLORS: Record<string, string> = {
  info: '#3b82f6', success: '#10b981', warning: '#f59e0b', error: '#ef4444', thinking: '#8b5cf6',
};

const VIEW_MODES = ['activity', 'flow', 'logs', 'memory', 'insights', 'streams'] as const;
type ViewMode = typeof VIEW_MODES[number];

interface HeatmapWord {
  word: string;
  count: number;
}

export function VisualTab() {
  const [viewMode, setViewMode] = useState<ViewMode>('activity');
  const [activityLog, setActivityLog] = useState<ActivityLog[]>([]);
  const [logFilter, setLogFilter] = useState<string>('all');
  const [autoScroll, setAutoScroll] = useState(true);
  const [insights, setInsights] = useState<Insight[]>([]);
  const [insightFilter, setInsightFilter] = useState<string>('all');
  const [insightSearch, setInsightSearch] = useState<string>('');
  const [heatmap, setHeatmap] = useState<HeatmapWord[]>([]);
  const [insightSourceFilter, setInsightSourceFilter] = useState<string>('all');
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [vectorEntries, setVectorEntries] = useState<VectorEntry[]>([]);
  const [knowledgeNodes, setKnowledgeNodes] = useState<KnowledgeNode[]>([]);
  const [knowledgeEdges, setKnowledgeEdges] = useState<KnowledgeEdge[]>([]);
  const [streamMessages, setStreamMessages] = useState<any[]>([]);
  const [streamSources, setStreamSources] = useState<string[]>([]);
  const [streamSourceFilter, setStreamSourceFilter] = useState<string>('all');
  const [streamContent, setStreamContent] = useState<{llm_calls: any[]; tool_calls: any[]}>({llm_calls: [], tool_calls: []});
  const logContainerRef = useRef<HTMLDivElement>(null);
  
  const dashboardData = useDashboard();
  const lastDataRef = useRef<any>(null);

  useEffect(() => {
    if (!dashboardData.data) return;
    const data = dashboardData.data;
    const now = Date.now();
    const newLogs: ActivityLog[] = [];

    if (JSON.stringify(lastDataRef.current?.dual_model) !== JSON.stringify(data.dual_model)) {
      const dm = data.dual_model || {};
      if (dm.primary_requests > 0 || dm.backend_requests > 0) {
        newLogs.push({
          timestamp: now, subsystem: 'Dual Model',
          event: dm.parallel_requests > 0 ? 'Parallel collaboration' : dm.sequential_requests > 0 ? 'Sequential task' : 'Model request',
          details: `Primary: ${dm.primary_requests || 0}, Backend: ${dm.backend_requests || 0}`,
          type: 'info', latency: dm.avg_primary_latency_ms || dm.avg_backend_latency_ms,
        });
      }
      if (dm.errors > 0) {
        newLogs.push({ timestamp: now, subsystem: 'Dual Model', event: 'Error occurred', details: `${dm.errors} total errors`, type: 'error' });
      }
    }

    const dd = data.daydream || {};
    if (dd.running !== lastDataRef.current?.daydream?.running) {
      if (dd.running) {
        newLogs.push({ timestamp: now, subsystem: 'Daydream', event: dd.current_topic || 'Exploring', details: `Session #${dd.session_count || 0}`, type: 'thinking' });
      }
    }

    const erdos = data.erdos || {};
    if (erdos.problems_solved > (lastDataRef.current?.erdos?.problems_solved || 0)) {
      newLogs.push({ timestamp: now, subsystem: 'Erdős Engine', event: 'Problem solved', details: erdos.current_problem || 'Math exploration', type: 'success' });
    }

    const subs = data.subsystems || {};
    Object.entries(subs).forEach(([key, subData]: [string, any]) => {
      const prevSub = lastDataRef.current?.subsystems?.[key];
      if (subData.running !== prevSub?.running || subData.status !== prevSub?.status) {
        newLogs.push({
          timestamp: now, subsystem: key,
          event: subData.running ? 'Started' : subData.status === 'error' ? 'Error' : 'Idle',
          details: subData.message || subData.current_task || '',
          type: subData.status === 'error' ? 'error' : subData.running ? 'success' : 'info',
        });
      }
    });

    if (newLogs.length > 0) setActivityLog(prev => [...prev, ...newLogs].slice(-200));
    lastDataRef.current = JSON.parse(JSON.stringify(data));
  }, [dashboardData.data]);

  useEffect(() => {
    if (autoScroll && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [activityLog, autoScroll]);

  useEffect(() => {
    if (dashboardData.data?.insights) {
      setInsights(dashboardData.data.insights.insights || []);
    }
  }, [dashboardData.data?.insights]);

  useEffect(() => {
    if (dashboardData.data?.vector_entries) {
      setVectorEntries(dashboardData.data.vector_entries.entries || []);
    }
  }, [dashboardData.data?.vector_entries]);

  useEffect(() => {
    if (dashboardData.data?.knowledge_nodes) {
      setKnowledgeNodes(dashboardData.data.knowledge_nodes.nodes || []);
    }
  }, [dashboardData.data?.knowledge_nodes]);

  useEffect(() => {
    const fetchKnowledgeEdges = async () => {
      try {
        const resp = await fetch('/api/knowledge-graph/edges?limit=200');
        const data = await resp.json();
        if (data.success) setKnowledgeEdges(data.edges || []);
      } catch (e) {}
    };
    fetchKnowledgeEdges();
    const interval = setInterval(fetchKnowledgeEdges, 5000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const fetchHeatmap = async () => {
      try {
        const resp = await fetch('/api/insights/heatmap?limit=100');
        const data = await resp.json();
        if (data.success) setHeatmap(data.words || []);
      } catch (e) {}
    };
    fetchHeatmap();
    const interval = setInterval(fetchHeatmap, 30000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const fetchStreams = async () => {
      try {
        const [msgResp, contentResp] = await Promise.all([
          fetch('/api/streams/messages?limit=200'),
          fetch('/api/streams/content?limit=50'),
        ]);
        const msgData = await msgResp.json();
        const contentData = await contentResp.json();
        if (msgData.success) {
          setStreamMessages(msgData.messages || []);
          setStreamSources(msgData.sources || []);
        }
        if (contentData.success) {
          setStreamContent({ llm_calls: contentData.llm_calls || [], tool_calls: contentData.tool_calls || [] });
        }
      } catch (e) {}
    };
    fetchStreams();
    const interval = setInterval(fetchStreams, 5000);
    return () => clearInterval(interval);
  }, []);

  const filteredLogs = useMemo(() => {
    if (logFilter === 'all') return activityLog;
    return activityLog.filter(log => log.type === logFilter);
  }, [activityLog, logFilter]);

  const filteredInsights = useMemo(() => {
    let filtered = insights;
    
    // Filter by type
    if (insightFilter !== 'all') {
      filtered = filtered.filter(i => i.type === insightFilter);
    }
    
    // Filter by source
    if (insightSourceFilter !== 'all') {
      filtered = filtered.filter(i => i.source === insightSourceFilter);
    }
    
    // Filter by search
    if (insightSearch.trim()) {
      const searchLower = insightSearch.toLowerCase();
      filtered = filtered.filter(i => 
        i.content.toLowerCase().includes(searchLower) ||
        i.category.toLowerCase().includes(searchLower) ||
        i.type.toLowerCase().includes(searchLower)
      );
    }
    
    return filtered;
  }, [insights, insightFilter, insightSourceFilter, insightSearch]);

  const dualModel = dashboardData.data?.dual_model || {};
  const totalRequests = (dualModel.primary_requests || 0) + (dualModel.backend_requests || 0);
  const primaryPercent = totalRequests > 0 ? ((dualModel.primary_requests || 0) / totalRequests) * 100 : 50;
  
  const memoryStats = dashboardData.data?.memory_stats?.stats || {};
  const embeddings = dashboardData.data?.embeddings || {};
  const subsystems = dashboardData.data?.subsystems || {};
  const context = dashboardData.data?.context || {};

  return (
    <div className="visual-tab">
      <div className="visual-header">
        <h2>👁️ Live Monitoring</h2>
        <div className="view-mode-tabs">
          {VIEW_MODES.map(mode => (
            <button key={mode} className={`view-tab ${viewMode === mode ? 'active' : ''}`} onClick={() => setViewMode(mode)}>
            {mode === 'activity' && '📊 Activity'}
            {mode === 'flow' && '🌐 Flow'}
            {mode === 'logs' && '📜 Live Logs'}
            {mode === 'memory' && '🧠 Memory'}
            {mode === 'insights' && '💡 Insights'}
            {mode === 'streams' && '📡 Streams'}
            </button>
          ))}
        </div>
      </div>

      {viewMode === 'activity' && (
        <div className="activity-view">
          <div className="activity-summary">
            <div className="summary-card">
              <h4>🤝 Bi-Model Architecture</h4>
              <div className="model-usage-bar">
                <div className="model-segment primary" style={{ width: `${primaryPercent}%` }}>
                  <span>Kimi ({dualModel.primary_requests || 0})</span>
                </div>
                <div className="model-segment backend" style={{ width: `${100 - primaryPercent}%` }}>
                  <span>GLM-5 ({dualModel.backend_requests || 0})</span>
                </div>
              </div>
              <div className="model-stats">
                <div className="stat"><span className="label">Parallel:</span><span className="value">{dualModel.parallel_requests || 0}</span></div>
                <div className="stat"><span className="label">Sequential:</span><span className="value">{dualModel.sequential_requests || 0}</span></div>
                <div className="stat"><span className="label">Latency:</span><span className="value">{Math.round(dualModel.avg_primary_latency_ms || 0)}ms</span></div>
                <div className="stat"><span className="label">Errors:</span><span className="value error">{dualModel.errors || 0}</span></div>
              </div>
            </div>

            <div className="summary-card">
              <h4>🔗 Embedding Interconnect</h4>
              <div className="embedding-stats">
                <div className="stat"><span className="label">Stored:</span><span className="value">{embeddings.stored_embeddings || 0}</span></div>
                <div className="stat"><span className="label">Primary:</span><span className="value">{embeddings.successful_patterns?.primary?.length || 0}</span></div>
                <div className="stat"><span className="label">Backend:</span><span className="value">{embeddings.successful_patterns?.backend?.length || 0}</span></div>
                <div className="stat"><span className="label">Shared:</span><span className="value">{embeddings.shared_context?.total_keys || 0}</span></div>
              </div>
            </div>

            <div className="summary-card">
              <h4>⚙️ Active Subsystems</h4>
              <div className="subsystem-grid">
                {Object.entries(subsystems).slice(0, 8).map(([key, sub]: [string, any]) => (
                  <div key={key} className={`sub-status ${(sub as any).running ? 'running' : 'idle'}`}>
                    <span className="icon">{SUBSYSTEM_ICONS[key] || '⚙️'}</span>
                    <span className="name">{key}</span>
                    <span className={`dot ${(sub as any).running ? 'active' : ''}`} />
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="recent-activity">
            <h4>📝 Recent Activity</h4>
            <div className="activity-list">
              {activityLog.slice(-10).reverse().map((log, i) => (
                <div key={i} className={`activity-item ${log.type}`}>
                  <span className="time">{new Date(log.timestamp).toLocaleTimeString()}</span>
                  <span className="icon">{SUBSYSTEM_ICONS[log.subsystem] || '⚙️'}</span>
                  <span className="subsystem">{log.subsystem}</span>
                  <span className="event">{log.event}</span>
                  <span className="details">{log.details}</span>
                  {log.latency && <span className="latency">{log.latency}ms</span>}
                </div>
              ))}
              {activityLog.length === 0 && <div className="no-activity">Waiting for activity...</div>}
            </div>
          </div>
        </div>
      )}

      {viewMode === 'flow' && (
        <FlowVisualization />
      )}

      {viewMode === 'logs' && (
        <div className="logs-view">
          <div className="logs-controls">
            <select value={logFilter} onChange={e => setLogFilter(e.target.value)}>
              <option value="all">All Logs</option>
              <option value="info">ℹ️ Info</option>
              <option value="success">✅ Success</option>
              <option value="warning">⚠️ Warning</option>
              <option value="error">❌ Error</option>
              <option value="thinking">💭 Thinking</option>
            </select>
            <label><input type="checkbox" checked={autoScroll} onChange={e => setAutoScroll(e.target.checked)} /> Auto-scroll</label>
            <button onClick={() => setActivityLog([])}>Clear Logs</button>
          </div>
          <div className="live-log-container" ref={logContainerRef}>
            {filteredLogs.map((log, i) => (
              <div key={i} className={`log-entry ${log.type}`}>
                <span className="log-time">{new Date(log.timestamp).toLocaleTimeString()}</span>
                <span className="log-icon">{SUBSYSTEM_ICONS[log.subsystem] || '⚙️'}</span>
                <span className="log-subsystem">{log.subsystem}</span>
                <span className="log-event">{log.event}</span>
                <span className="log-details">{log.details}</span>
                {log.latency && <span className="log-latency">{log.latency}ms</span>}
              </div>
            ))}
            {filteredLogs.length === 0 && <div className="empty-logs"><span>📜</span><p>No logs yet.</p></div>}
          </div>
        </div>
      )}

      {viewMode === 'memory' && (
        <div className="memory-view">
          <div className="memory-stats-grid">
            <div className="memory-card conversation">
              <div className="card-header"><span className="icon">💬</span><h4>Conversation Memory</h4></div>
              <div className="card-stats">
                <div className="stat"><span className="value">{memoryStats.conversation?.messages || 0}</span><span className="label">Messages</span></div>
                <div className="stat"><span className="value">{memoryStats.conversation?.turns || 0}</span><span className="label">Turns</span></div>
                <div className="stat"><span className="value">{memoryStats.conversation?.enhanced?.total_messages || 0}</span><span className="label">Enhanced</span></div>
              </div>
              <div className="card-note">Working memory for current conversation</div>
            </div>

            <div className="memory-card vector">
              <div className="card-header"><span className="icon">🔎</span><h4>Vector Memory (HNSW)</h4></div>
              {memoryStats.vector?.available ? (
                <div className="card-stats">
                  <div className="stat"><span className="value">{memoryStats.vector.vector_count || 0}</span><span className="label">Vectors</span></div>
                  <div className="stat"><span className="value">{memoryStats.vector.dim || 768}</span><span className="label">Dimensions</span></div>
                  <div className="stat"><span className="value">{memoryStats.vector.max_level || 1}</span><span className="label">Max Level</span></div>
                </div>
              ) : (
                <div className="unavailable">Vector memory not initialized</div>
              )}
              <div className="card-note">150x faster semantic retrieval</div>
            </div>

            <div className="memory-card knowledge">
              <div className="card-header"><span className="icon">🕸️</span><h4>Knowledge Graph</h4></div>
              {memoryStats.knowledge_graph?.available ? (
                <div className="card-stats">
                  <div className="stat"><span className="value">{memoryStats.knowledge_graph.nodes || 0}</span><span className="label">Nodes</span></div>
                  <div className="stat"><span className="value">{memoryStats.knowledge_graph.edges || 0}</span><span className="label">Edges</span></div>
                  <div className="stat"><span className="value">{memoryStats.knowledge_graph.communities || 0}</span><span className="label">Communities</span></div>
                </div>
              ) : (
                <div className="unavailable">Knowledge graph not initialized</div>
              )}
              <div className="card-note">PageRank + community detection for insights</div>
            </div>

            <div className="memory-card context">
              <div className="card-header"><span className="icon">📝</span><h4>Active Context</h4></div>
              <div className="card-stats">
                <div className="stat"><span className="value">{context.active_tokens || 0}</span><span className="label">Tokens</span></div>
                <div className="stat"><span className="value">{context.max_tokens || 96000}</span><span className="label">Max</span></div>
                <div className="stat"><span className="value">{((context.usage_ratio || 0) * 100).toFixed(1)}%</span><span className="label">Usage</span></div>
              </div>
              <div className="card-note">Context window: {context.enhanced_enabled ? 'Enhanced' : 'Standard'}</div>
            </div>

            <div className="memory-card embedding">
              <div className="card-header"><span className="icon">🔗</span><h4>Embedding Interconnect</h4></div>
              <div className="card-stats">
                <div className="stat"><span className="value">{embeddings.stored_embeddings || 0}</span><span className="label">Stored</span></div>
                <div className="stat"><span className="value">{embeddings.successful_patterns?.primary?.length || 0}</span><span className="label">Primary</span></div>
                <div className="stat"><span className="value">{embeddings.successful_patterns?.backend?.length || 0}</span><span className="label">Backend</span></div>
              </div>
              <div className="card-note">Cross-model pattern matching</div>
            </div>

            <div className="memory-card architecture">
              <div className="card-header"><span className="icon">🏗️</span><h4>Memory Layers</h4></div>
              <div className="architecture-diagram">
                <div className="arch-layer working"><span className="layer-name">Working</span><span className="layer-desc">Immediate</span></div>
                <div className="arch-layer session"><span className="layer-name">Session</span><span className="layer-desc">Important</span></div>
                <div className="arch-layer episodic"><span className="layer-name">Episodic</span><span className="layer-desc">Vector</span></div>
                <div className="arch-layer semantic"><span className="layer-name">Semantic</span><span className="layer-desc">Graph</span></div>
              </div>
            </div>
          </div>

          <div className="memory-viz-grid">
            <div className="memory-card graph-viz">
              <div className="card-header"><span className="icon">🕸️</span><h4>Knowledge Graph Visualization</h4></div>
              <div className="graph-container">
                <svg className="kg-svg" viewBox="0 0 500 300" preserveAspectRatio="xMidYMid meet">
                  {(() => {
                    const nodePositions: Record<string, {x: number, y: number}> = {};
                    const visibleNodes = knowledgeNodes.slice(0, 60);
                    const cols = Math.ceil(Math.sqrt(visibleNodes.length) * 1.5);
                    visibleNodes.forEach((node, i) => {
                      const row = Math.floor(i / cols);
                      const col = i % cols;
                      const cellW = 480 / cols;
                      const cellH = 280 / Math.ceil(visibleNodes.length / cols);
                      nodePositions[node.id] = { x: 10 + col * cellW + cellW / 2, y: 10 + row * cellH + cellH / 2 };
                    });
                    return (
                      <>
                        {knowledgeEdges.slice(0, 60).map((edge, i) => {
                          const sourcePos = nodePositions[edge.source];
                          const targetPos = nodePositions[edge.target];
                          if (!sourcePos || !targetPos) return null;
                          return <line key={`e-${i}`} x1={sourcePos.x} y1={sourcePos.y} x2={targetPos.x} y2={targetPos.y} stroke="rgba(124, 58, 237, 0.25)" strokeWidth={Math.max(0.5, edge.weight || 1)} />;
                        })}
                        {visibleNodes.map((node, i) => {
                          const pos = nodePositions[node.id];
                          const color = node.type === 'insight' ? '#22c55e' : node.type === 'problem' ? '#f59e0b' : node.type === 'concept' ? '#3b82f6' : node.type === 'strategy' ? '#ec4899' : '#6b7280';
                          return <circle key={`n-${i}`} cx={pos.x} cy={pos.y} r={3 + (node.pagerank || 0) * 80} fill={color} stroke="rgba(255,255,255,0.15)" strokeWidth={0.5}><title>{`${node.type}: ${(node.content || '').slice(0, 60)}`}</title></circle>;
                        })}
                      </>
                    );
                  })()}
                </svg>
                {knowledgeNodes.length === 0 && <div className="empty-graph">No nodes to display</div>}
              </div>
              <div className="graph-legend">
                <span className="legend-item"><span className="legend-dot" style={{background: '#22c55e'}}/>Insight</span>
                <span className="legend-item"><span className="legend-dot" style={{background: '#f59e0b'}}/>Problem</span>
                <span className="legend-item"><span className="legend-dot" style={{background: '#3b82f6'}}/>Concept</span>
                <span className="legend-item"><span className="legend-dot" style={{background: '#ec4899'}}/>Strategy</span>
              </div>
            </div>

            <div className="memory-card vector-viz">
              <div className="card-header"><span className="icon">🔎</span><h4>Recent Vector Entries</h4></div>
              <div className="vector-list">
                {vectorEntries.length > 0 ? (
                  vectorEntries.slice(0, 6).map((entry, i) => (
                    <div key={entry.id || i} className="vector-entry">
                      <div className="vector-header">
                        <span className="vector-id">{(entry.id || '').slice(0, 10)}</span>
                        <div className="importance-bar-container">
                          <div className="importance-bar" style={{ width: `${((entry as any).importance || 0.5) * 100}%` }} />
                        </div>
                      </div>
                      <div className="vector-text">{(entry.content || entry.text || '').slice(0, 80)}...</div>
                      <div className="vector-meta"><span>📥 {(entry as any).access_count || 0} accesses</span></div>
                    </div>
                  ))
                ) : (
                  <div className="empty-vectors">No vector entries yet</div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {viewMode === 'insights' && (
        <div className="insights-view">
          <div className="insights-controls">
            <input
              type="text"
              className="insight-search"
              placeholder="Search insights..."
              value={insightSearch}
              onChange={e => setInsightSearch(e.target.value)}
            />
            <select value={insightFilter} onChange={e => setInsightFilter(e.target.value)}>
              <option value="all">All Types</option>
              <option value="pattern_discovery">Pattern Discovery</option>
              <option value="strategy_success">Strategy Success</option>
              <option value="cross_domain">Cross Domain</option>
              <option value="math_discovery">Math Discovery</option>
              <option value="exploration">Exploration</option>
              <option value="contribution">Contribution</option>
            </select>
            <select value={insightSourceFilter} onChange={e => setInsightSourceFilter(e.target.value)}>
              <option value="all">All Sources</option>
              <option value="insight_bridge">Insight Bridge</option>
              <option value="erdos">Erdős</option>
              <option value="daydream">Daydream</option>
              <option value="knowledge">Knowledge</option>
            </select>
            <button onClick={() => setShowHeatmap(!showHeatmap)}>
              {showHeatmap ? '📊 Hide Heatmap' : '📊 Word Heatmap'}
            </button>
            <button onClick={() => dashboardData.refresh()}>🔄 Refresh</button>
            <span className="insight-count">{filteredInsights.length} insights</span>
          </div>
          
          {showHeatmap && heatmap.length > 0 && (
            <div className="heatmap-container">
              <h4>Word Frequency Heatmap</h4>
              <div className="heatmap-words">
                {heatmap.slice(0, 50).map((w, i) => {
                  const maxCount = heatmap[0]?.count || 1;
                  const size = 10 + (w.count / maxCount) * 20;
                  const opacity = 0.3 + (w.count / maxCount) * 0.7;
                  return (
                    <span
                      key={i}
                      className="heatmap-word"
                      style={{ fontSize: `${size}px`, opacity }}
                      onClick={() => setInsightSearch(w.word)}
                    >
                      {w.word}
                    </span>
                  );
                })}
              </div>
            </div>
          )}
          
          <div className="insights-grid">
            {filteredInsights.map((insight, i) => (
              <div key={insight.id || i} className="insight-card">
                <div className="insight-header">
                  <span className="insight-source">{SUBSYSTEM_ICONS[insight.source] || '💡'} {insight.source}</span>
                  <span className="insight-type" style={{ backgroundColor: TYPE_COLORS[insight.type] || TYPE_COLORS.info }}>
                    {(insight.type || 'unknown').replace(/_/g, ' ')}
                  </span>
                </div>
                <div className="insight-content">{insight.content}</div>
                <div className="insight-meta">
                  <span className="insight-category">{insight.category}</span>
                  <span className="insight-time">{insight.timestamp ? new Date(insight.timestamp).toLocaleString() : ''}</span>
                </div>
                {insight.relevance > 0 && <div className="insight-relevance">Relevance: {insight.relevance}</div>}
              </div>
            ))}
            {filteredInsights.length === 0 && (
              <div className="no-insights">
                <span className="icon">💭</span>
                <p>No insights match your search. Try different keywords or filters.</p>
              </div>
            )}
          </div>
        </div>
      )}

      {viewMode === 'streams' && (
        <div className="matrix-streams">
          <div className="matrix-rain">
            {Array.from({ length: 30 }).map((_, col) => (
              <div key={col} className="rain-column" style={{
                left: `${(col / 30) * 100}%`,
                animationDuration: `${4 + (col % 7) * 1.5}s`,
                animationDelay: `${(col % 5) * 0.8}s`,
              }}>
                {Array.from({ length: 18 }).map((_, row) => (
                  <span key={row} className="rain-char" style={{
                    animationDelay: `${row * 0.1}s`,
                    opacity: 0.03 + (row / 18) * 0.12,
                  }}>
                    {String.fromCharCode(0x30A0 + Math.floor(Math.random() * 96))}
                  </span>
                ))}
              </div>
            ))}
          </div>

          <div className="matrix-overlay">
            <div className="matrix-header">
              <div className="matrix-title">
                <span className="glitch" data-text="JARVIS STREAM MONITOR">JARVIS STREAM MONITOR</span>
              </div>
              <div className="matrix-controls">
                <select value={streamSourceFilter} onChange={e => setStreamSourceFilter(e.target.value)} className="matrix-select">
                  <option value="all">ALL SOURCES</option>
                  {streamSources.map(s => <option key={s} value={s}>{s.toUpperCase()}</option>)}
                </select>
                <span className="matrix-badge">{streamMessages.length} MSG</span>
                <span className="matrix-badge llm">{streamContent.llm_calls.length} LLM</span>
                <span className="matrix-badge tool">{streamContent.tool_calls.length} TOOL</span>
                <span className="matrix-led active" />
              </div>
            </div>

            <div className="matrix-terminal-grid">
              <div className="matrix-panel">
                <div className="matrix-panel-header">
                  <span className="matrix-dot red" />
                  <span className="matrix-dot yellow" />
                  <span className="matrix-dot green" />
                  <span className="matrix-panel-title">INTER-MODULE BUS</span>
                </div>
                <div className="matrix-scroll">
                  {streamMessages
                    .filter(m => streamSourceFilter === 'all' || m.source === streamSourceFilter || m._type === streamSourceFilter)
                    .slice(0, 100)
                    .map((msg, i) => {
                      const isBridge = msg._type === 'bridge';
                      const ts = msg.timestamp ? new Date(msg.timestamp * 1000).toLocaleTimeString('en-US', { hour12: false }) : '';
                      const evt = msg.event || msg.channel || '';
                      const src = msg.source || 'unknown';
                      const data = msg.data || {};
                      const dataStr = typeof data === 'object' ? JSON.stringify(data).slice(0, 120) : String(data).slice(0, 120);
                      return (
                        <div key={i} className={`matrix-line ${isBridge ? 'bridge' : 'bus'}`}>
                          <span className="mx-ts">{ts}</span>
                          <span className={`mx-tag ${isBridge ? 'bridge' : 'bus'}`}>{isBridge ? 'BRIDGE' : 'BUS'}</span>
                          <span className="mx-src">{src}</span>
                          <span className="mx-evt">{evt}</span>
                          <span className="mx-data">{dataStr}</span>
                        </div>
                      );
                    })}
                  {streamMessages.length === 0 && (
                    <div className="matrix-idle">AWAITING DATA STREAM...</div>
                  )}
                </div>
              </div>

              <div className="matrix-panel">
                <div className="matrix-panel-header">
                  <span className="matrix-dot red" />
                  <span className="matrix-dot yellow" />
                  <span className="matrix-dot green" />
                  <span className="matrix-panel-title">LLM NEURAL LINK</span>
                </div>
                <div className="matrix-scroll">
                  {streamContent.llm_calls.slice(0, 50).map((call, i) => {
                    const ts = call.timestamp ? new Date(call.timestamp * 1000).toLocaleTimeString('en-US', { hour12: false }) : '';
                    const isReq = call.direction === 'request';
                    return (
                      <div key={i} className="matrix-line llm">
                        <span className="mx-ts">{ts}</span>
                        <span className={`mx-tag ${isReq ? 'req' : 'res'}`}>{isReq ? 'REQ' : 'RES'}</span>
                        <span className="mx-src">{call.model || 'unknown'}</span>
                        <span className="mx-data">{(call.content_preview || '').slice(0, 80)}</span>
                        <span className="mx-meta">
                          {call.tokens_in > 0 && `↑${call.tokens_in}`}
                          {call.tokens_out > 0 && ` ↓${call.tokens_out}`}
                          {call.latency_ms > 0 && ` ${call.latency_ms}ms`}
                        </span>
                      </div>
                    );
                  })}
                  {streamContent.llm_calls.length === 0 && (
                    <div className="matrix-idle">NEURAL LINK IDLE...</div>
                  )}
                </div>
              </div>

              <div className="matrix-panel">
                <div className="matrix-panel-header">
                  <span className="matrix-dot red" />
                  <span className="matrix-dot yellow" />
                  <span className="matrix-dot green" />
                  <span className="matrix-panel-title">TOOL INVOCATION</span>
                </div>
                <div className="matrix-scroll">
                  {streamContent.tool_calls.slice(0, 50).map((tc, i) => {
                    const ts = tc.timestamp ? new Date(tc.timestamp * 1000).toLocaleTimeString('en-US', { hour12: false }) : '';
                    return (
                      <div key={i} className="matrix-line tool">
                        <span className="mx-ts">{ts}</span>
                        <span className="mx-tag tool">EXEC</span>
                        <span className="mx-src">{tc.tool}</span>
                        <span className="mx-data">{(tc.args_preview || '').slice(0, 80)}</span>
                        {tc.duration_ms > 0 && <span className="mx-meta">{tc.duration_ms}ms</span>}
                      </div>
                    );
                  })}
                  {streamContent.tool_calls.length === 0 && (
                    <div className="matrix-idle">NO TOOL ACTIVITY...</div>
                  )}
                </div>
              </div>
            </div>

            <div className="matrix-footer">
              <span>SYS: NOMINAL</span>
              <span>UPTIME: {Math.floor((Date.now() % 86400000) / 3600000)}h {Math.floor(((Date.now() % 86400000) % 3600000) / 60000)}m</span>
              <span>BUS: {streamMessages.length} MSG/s</span>
              <span>NEURAL: {streamContent.llm_calls.length} CALL</span>
              <span>TOOLS: {streamContent.tool_calls.length} EXEC</span>
              <span className="matrix-led active" />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}