import { useState, useEffect, useRef } from 'react';

interface Subsystem {
  name: string;
  icon: string;
  status: 'active' | 'idle' | 'unavailable';
  stats: Record<string, any>;
}

interface VisualizationData {
  timestamp: string;
  subsystems: Record<string, Subsystem>;
  connections: Array<{ from: string; to: string; label: string; active: boolean }>;
  tool_categories: Record<string, number>;
  mermaid: string;
  stats: {
    total_tools: number;
    active_subsystems: number;
  };
}

interface DaydreamStatus {
  running: boolean;
  can_daydream: boolean;
  daydream_count: number;
  total_daydream_time: number;
  tools_practiced: number;
  insights_generated: number;
  exploration_topics: string[];
  tools: Array<{
    name: string;
    level: string;
    uses: number;
    success_rate: number;
  }>;
  practice_priorities: string[];
}

interface AllSubsystems {
  erdos?: { status: string; problems_explored?: number; running?: boolean };
  daydream?: { status: string; daydream_count?: number; running?: boolean };
  evolution?: { status: string; genes?: number; running?: boolean };
  learning?: { status: string; sessions?: number };
  defense?: { status: string; threats_blocked?: number; monitoring?: boolean };
  intuition?: { status: string; insights_processed?: number };
  free_time?: { status: string; sessions_completed?: number; sandbox_active?: boolean };
}

interface ToolUsage {
  name: string;
  category: string;
  uses: number;
  last_used?: string;
}

interface KGNode {
  id: string;
  type: string;
  content: string;
  pagerank: number;
  community_id: number;
}

interface KGEdge {
  source: string;
  target: string;
  type: string;
  weight: number;
}

interface VectorEntry {
  id: string;
  text: string;
  importance: number;
  access_count: number;
  created_at: string | null;
}

const TOOL_TREES: Record<string, string[]> = {
  'File Operations': ['read_file', 'write_file', 'list_files', 'delete_file', 'shell', 'terminal_execute'],
  'Web Research': ['web_search', 'web_fetch', 'image_search', 'image_download'],
  'Screen Automation': ['see_screen', 'find_element', 'click_element', 'type_text', 'press_key', 'hotkey'],
  'Browser Tasks': ['browser_launch', 'browser_navigate', 'browser_click', 'browser_type'],
  'Terminal Access': ['terminal_execute', 'terminal_cwd', 'terminal_history', 'computer_status'],
  'Data Analysis': ['data_analyze', 'sql_query', 'statistics', 'embed', 'search_files'],
  'Voice/Audio': ['transcribe', 'tokenize_voice', 'process_speech', 'synthesize_speech'],
  'Learning': ['erdos_math', 'daydream', 'learning', 'sandbox_explore'],
  'Self-Improvement': ['evolution', 'self_edit', 'refine_scan', 'refine_apply'],
  'Meta/Discovery': ['list_tools', 'get_tool_info', 'get_capabilities', 'find_related_tools'],
};

export function LiveVisualization() {
  const [data, setData] = useState<VisualizationData | null>(null);
  const [daydream, setDaydream] = useState<DaydreamStatus | null>(null);
  const [allSubsystems, setAllSubsystems] = useState<AllSubsystems | null>(null);
  const [toolUsage, setToolUsage] = useState<ToolUsage[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [timeRange, setTimeRange] = useState<'realtime' | '1h' | '24h'>('realtime');
  const [activeTab, setActiveTab] = useState<'architecture' | 'tools' | 'subsystems' | 'knowledge'>('architecture');
  const mermaidRef = useRef<HTMLDivElement>(null);
  const [kgNodes, setKgNodes] = useState<KGNode[]>([]);
  const [kgEdges, setKgEdges] = useState<KGEdge[]>([]);
  const [vectorEntries, setVectorEntries] = useState<VectorEntry[]>([]);
  const [kgStats, setKgStats] = useState<{total_nodes: number; total_edges: number; communities: number}>({total_nodes: 0, total_edges: 0, communities: 0});

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [vizResp, dreamResp, subsystemsResp] = await Promise.all([
          fetch('/api/visualization/live'),
          fetch('/api/daydream/status'),
          fetch('/api/subsystems/all'),
        ]);
        const json = await vizResp.json();
        const dreamJson = await dreamResp.json();
        const subsJson = await subsystemsResp.json();
        
        if (json.success) {
          setData(json);
          setError(null);
        } else {
          setError(json.error || 'Failed to load visualization');
        }
        
        if (dreamJson.success) {
          setDaydream(dreamJson);
          if (dreamJson.tools) {
            setToolUsage(dreamJson.tools.map((t: { name: string; uses: number }) => ({
              name: t.name,
              category: 'learning',
              uses: t.uses,
            })));
          }
        }
        
        if (subsJson.success) {
          setAllSubsystems(subsJson.subsystems);
        }
      } catch (e: any) {
        setError(e.message || 'Connection error');
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 2000);
    return () => clearInterval(interval);
  }, [timeRange]);

  useEffect(() => {
    const fetchKnowledgeData = async () => {
      try {
        const [nodesResp, edgesResp, entriesResp, statsResp] = await Promise.all([
          fetch('/api/knowledge-graph/nodes?limit=200'),
          fetch('/api/knowledge-graph/edges?limit=300'),
          fetch('/api/vector-memory/entries?limit=200'),
          fetch('/api/memory/stats'),
        ]);
        
        const nodesJson = await nodesResp.json();
        const edgesJson = await edgesResp.json();
        const entriesJson = await entriesResp.json();
        const statsJson = await statsResp.json();
        
        if (nodesJson.success) setKgNodes(nodesJson.nodes);
        if (edgesJson.success) setKgEdges(edgesJson.edges);
        if (entriesJson.success) setVectorEntries(entriesJson.entries);
        if (statsJson.success && statsJson.stats) {
          const kg = statsJson.stats.knowledge_graph;
          if (kg) {
            setKgStats({
              total_nodes: kg.total_nodes || nodesJson.total || 0,
              total_edges: kg.total_edges || edgesJson.total || 0,
              communities: kg.communities || 0,
            });
          }
        }
      } catch (e) {
        console.error('Failed to fetch knowledge data:', e);
      }
    };
    
    fetchKnowledgeData();
    const interval = setInterval(fetchKnowledgeData, 5000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (data?.mermaid && mermaidRef.current) {
      renderMermaid(data.mermaid);
    }
  }, [data?.mermaid]);

  const renderMermaid = async (code: string) => {
    if (!mermaidRef.current) return;
    
    try {
      const mermaid = (window as any).mermaid;
      if (!mermaid) {
        mermaidRef.current.innerHTML = `<pre style="font-size: 12px; overflow: auto;">${code}</pre>`;
        return;
      }
      
      const id = `mermaid-${Date.now()}`;
      const { svg } = await mermaid.render(id, code);
      mermaidRef.current.innerHTML = svg;
    } catch (e) {
      mermaidRef.current.innerHTML = `<pre style="font-size: 12px; overflow: auto; white-space: pre-wrap;">${code}</pre>`;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return '#22c55e';
      case 'idle': return '#6b7280';
      case 'unavailable': return '#ef4444';
      default: return '#6b7280';
    }
  };

  const getStatusGlow = (status: string) => {
    switch (status) {
      case 'active': return '0 0 20px rgba(34, 197, 94, 0.5)';
      case 'idle': return 'none';
      case 'unavailable': return '0 0 20px rgba(239, 68, 68, 0.3)';
      default: return 'none';
    }
  };

  if (isLoading) {
    return (
      <div className="viz-loading">
        <div className="viz-spinner"></div>
        <span>Loading visualization...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="viz-error">
        <span>⚠️ {error}</span>
      </div>
    );
  }

  return (
    <div className="live-visualization">
      <div className="viz-header">
        <h2>🔮 Live System Visualization</h2>
        <div className="viz-controls">
          <span className="viz-time">{new Date(data?.timestamp || '').toLocaleTimeString()}</span>
          <div className="viz-range-btns">
            {(['realtime', '1h', '24h'] as const).map(range => (
              <button
                key={range}
                className={`viz-range-btn ${timeRange === range ? 'active' : ''}`}
                onClick={() => setTimeRange(range)}
              >
                {range === 'realtime' ? '⚡ Live' : range}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="viz-tabs">
        <button
          className={`viz-tab ${activeTab === 'architecture' ? 'active' : ''}`}
          onClick={() => setActiveTab('architecture')}
        >
          🏗️ Architecture
        </button>
        <button
          className={`viz-tab ${activeTab === 'tools' ? 'active' : ''}`}
          onClick={() => setActiveTab('tools')}
        >
          🔧 Tool Trees
        </button>
        <button
          className={`viz-tab ${activeTab === 'subsystems' ? 'active' : ''}`}
          onClick={() => setActiveTab('subsystems')}
        >
          🧠 Subsystems
        </button>
        <button
          className={`viz-tab ${activeTab === 'knowledge' ? 'active' : ''}`}
          onClick={() => setActiveTab('knowledge')}
        >
          🕸️ Knowledge
        </button>
      </div>

      <div className="viz-content">
        {activeTab === 'architecture' && (
          <div className="viz-main">
            <div className="viz-diagram">
              <div className="diagram-header">
                <h3>System Architecture</h3>
                <span className="diagram-badge">{data?.stats.active_subsystems} active</span>
              </div>
              <div className="mermaid-container" ref={mermaidRef}></div>
            </div>

            <div className="viz-flows">
              <h3>🔄 Active Flows</h3>
              <div className="flow-list">
                {data?.connections.filter(c => c.active).map((conn, i) => (
                  <div key={i} className="flow-item active">
                    <span className="flow-from">{conn.from}</span>
                    <span className="flow-arrow">→</span>
                    <span className="flow-label">{conn.label}</span>
                    <span className="flow-arrow">→</span>
                    <span className="flow-to">{conn.to}</span>
                  </div>
                ))}
                {data?.connections.filter(c => c.active).length === 0 && (
                  <div className="flow-item idle">No active flows</div>
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'tools' && (
          <div className="viz-main tools-view">
            <div className="tool-trees-grid">
              {Object.entries(TOOL_TREES).map(([treeName, tools]) => (
                <div key={treeName} className="tool-tree-card">
                  <h4 className="tool-tree-name">{treeName}</h4>
                  <div className="tool-tree-tools">
                    {tools.map(toolName => {
                      const usage = toolUsage.find((t: ToolUsage) => t.name === toolName);
                      return (
                        <div 
                          key={toolName} 
                          className={`tool-item ${usage ? 'used' : ''}`}
                          title={usage ? `Used ${usage.uses} times` : 'Not used'}
                        >
                          <span className="tool-name">{toolName}</span>
                          {usage && <span className="tool-uses">{usage.uses}</span>}
                        </div>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>

            <div className="tool-best-for">
              <h3>🎯 Best Tool For Task</h3>
              <div className="best-for-grid">
                <div className="best-for-item">
                  <span className="task">Search the web</span>
                  <span className="arrow">→</span>
                  <code className="tool">web_search</code>
                </div>
                <div className="best-for-item">
                  <span className="task">Execute commands</span>
                  <span className="arrow">→</span>
                  <code className="tool">terminal_execute</code>
                </div>
                <div className="best-for-item">
                  <span className="task">See the screen</span>
                  <span className="arrow">→</span>
                  <code className="tool">see_screen</code>
                </div>
                <div className="best-for-item">
                  <span className="task">Analyze data</span>
                  <span className="arrow">→</span>
                  <code className="tool">data_analyze</code>
                </div>
                <div className="best-for-item">
                  <span className="task">Find files</span>
                  <span className="arrow">→</span>
                  <code className="tool">search_files</code>
                </div>
                <div className="best-for-item">
                  <span className="task">Improve code</span>
                  <span className="arrow">→</span>
                  <code className="tool">refine_scan</code>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'subsystems' && allSubsystems && (
          <div className="viz-main subsystems-view">
            <div className="subsystem-detailed-grid">
              {Object.entries(allSubsystems).map(([key, sub]: [string, any]) => (
                <div key={key} className={`subsystem-detailed-card ${sub.status}`}>
                  <div className="subsystem-detailed-header">
                    <span className="subsystem-icon">
                      {key === 'erdos' ? '🧮' : 
                       key === 'daydream' ? '💭' :
                       key === 'evolution' ? '🧬' :
                       key === 'learning' ? '📚' :
                       key === 'defense' ? '🛡️' :
                       key === 'intuition' ? '💡' :
                       key === 'free_time' ? '🎭' : '⚙️'}
                    </span>
                    <span className="subsystem-name">{key.replace('_', ' ').toUpperCase()}</span>
                    <span className={`subsystem-status-badge ${sub.status}`}>
                      {sub.status}
                    </span>
                  </div>
                  <div className="subsystem-detailed-stats">
                    {sub.problems_explored !== undefined && (
                      <div className="stat-item">
                        <span className="stat-label">Problems</span>
                        <span className="stat-value">{sub.problems_explored.toLocaleString()}</span>
                      </div>
                    )}
                    {sub.daydream_count !== undefined && (
                      <div className="stat-item">
                        <span className="stat-label">Sessions</span>
                        <span className="stat-value">{sub.daydream_count}</span>
                      </div>
                    )}
                    {sub.genes !== undefined && (
                      <div className="stat-item">
                        <span className="stat-label">Genes</span>
                        <span className="stat-value">{sub.genes}</span>
                      </div>
                    )}
                    {sub.sessions !== undefined && (
                      <div className="stat-item">
                        <span className="stat-label">Sessions</span>
                        <span className="stat-value">{sub.sessions}</span>
                      </div>
                    )}
                    {sub.threats_blocked !== undefined && (
                      <div className="stat-item">
                        <span className="stat-label">Threats</span>
                        <span className="stat-value">{sub.threats_blocked}</span>
                      </div>
                    )}
                    {sub.insights_processed !== undefined && (
                      <div className="stat-item">
                        <span className="stat-label">Insights</span>
                        <span className="stat-value">{sub.insights_processed.toLocaleString()}</span>
                      </div>
                    )}
                    {sub.sandbox_active !== undefined && (
                      <div className="stat-item">
                        <span className="stat-label">Sandbox</span>
                        <span className="stat-value">{sub.sandbox_active ? '🟢 Active' : '⚪ Idle'}</span>
                      </div>
                    )}
                    {sub.sessions_completed !== undefined && (
                      <div className="stat-item">
                        <span className="stat-label">Completed</span>
                        <span className="stat-value">{sub.sessions_completed}</span>
                      </div>
                    )}
                    {sub.running !== undefined && (
                      <div className="stat-item">
                        <span className="stat-label">Running</span>
                        <span className="stat-value">{sub.running ? '✓' : '✗'}</span>
                      </div>
                    )}
                    {sub.monitoring !== undefined && (
                      <div className="stat-item">
                        <span className="stat-label">Monitoring</span>
                        <span className="stat-value">{sub.monitoring ? '✓' : '✗'}</span>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'knowledge' && (
          <div className="viz-main knowledge-view">
            <div className="knowledge-grid">
              <div className="knowledge-section">
                <div className="knowledge-section-header">
                  <h3>🕸️ Knowledge Graph</h3>
                  <span className="knowledge-stats-badge">
                    {kgStats.total_nodes} nodes · {kgStats.total_edges} edges
                  </span>
                </div>
                <div className="knowledge-graph-visual">
                  <svg className="kg-svg" viewBox="0 0 600 400" preserveAspectRatio="xMidYMid meet">
                    {(() => {
                      const nodePositions: Record<string, {x: number, y: number}> = {};
                      const visibleNodes = kgNodes.slice(0, 100);
                      const cols = Math.ceil(Math.sqrt(visibleNodes.length) * 1.5);
                      visibleNodes.forEach((node, i) => {
                        const row = Math.floor(i / cols);
                        const col = i % cols;
                        const cellW = 580 / cols;
                        const cellH = 380 / Math.ceil(visibleNodes.length / cols);
                        nodePositions[node.id] = {
                          x: 10 + col * cellW + cellW / 2,
                          y: 10 + row * cellH + cellH / 2,
                        };
                      });
                      return (
                        <>
                          {kgEdges.slice(0, 100).map((edge, i) => {
                            const sourcePos = nodePositions[edge.source];
                            const targetPos = nodePositions[edge.target];
                            if (!sourcePos || !targetPos) return null;
                            return (
                              <line
                                key={`edge-${i}`}
                                x1={sourcePos.x}
                                y1={sourcePos.y}
                                x2={targetPos.x}
                                y2={targetPos.y}
                                stroke="rgba(124, 58, 237, 0.3)"
                                strokeWidth={Math.max(0.5, edge.weight)}
                              />
                            );
                          })}
                          {visibleNodes.map((node, i) => {
                            const pos = nodePositions[node.id];
                            const color = node.type === 'insight' ? '#22c55e' :
                                         node.type === 'problem' ? '#f59e0b' :
                                         node.type === 'concept' ? '#3b82f6' :
                                         node.type === 'strategy' ? '#ec4899' : '#6b7280';
                            return (
                              <circle
                                key={`node-${i}`}
                                cx={pos.x}
                                cy={pos.y}
                                r={4 + node.pagerank * 100}
                                fill={color}
                                stroke="rgba(255,255,255,0.2)"
                                strokeWidth={1}
                              >
                                <title>{`${node.type}: ${node.content.slice(0, 80)}`}</title>
                              </circle>
                            );
                          })}
                        </>
                      );
                    })()}
                  </svg>
                </div>
                <div className="knowledge-legend">
                  <span className="legend-item"><span className="legend-dot insight"/>Insight</span>
                  <span className="legend-item"><span className="legend-dot problem"/>Problem</span>
                  <span className="legend-item"><span className="legend-dot concept"/>Concept</span>
                  <span className="legend-item"><span className="legend-dot strategy"/>Strategy</span>
                </div>
              </div>

              <div className="knowledge-section">
                <div className="knowledge-section-header">
                  <h3>🔎 Vector Memory</h3>
                  <span className="knowledge-stats-badge">{vectorEntries.length} entries</span>
                </div>
                <div className="vector-entries-list">
                  {vectorEntries.slice(0, 20).map((entry, i) => (
                    <div key={entry.id || i} className="vector-entry-item">
                      <div className="vector-entry-header">
                        <span className="vector-entry-id">{entry.id}</span>
                        <div className="vector-entry-importance">
                          <div 
                            className="importance-bar" 
                            style={{ width: `${entry.importance * 100}%` }}
                          />
                        </div>
                      </div>
                      <div className="vector-entry-text">{entry.text}</div>
                      <div className="vector-entry-meta">
                        <span>📥 {entry.access_count} accesses</span>
                        {entry.created_at && <span>📅 {new Date(entry.created_at).toLocaleDateString()}</span>}
                      </div>
                    </div>
                  ))}
                  {vectorEntries.length === 0 && (
                    <div className="empty-state">
                      <span className="empty-icon">📭</span>
                      <p>No vector memory entries yet</p>
                    </div>
                  )}
                </div>
              </div>

              <div className="knowledge-section">
                <div className="knowledge-section-header">
                  <h3>📊 Top Nodes by PageRank</h3>
                </div>
                <div className="top-nodes-list">
                  {[...kgNodes]
                    .sort((a, b) => b.pagerank - a.pagerank)
                    .slice(0, 15)
                    .map((node, i) => (
                      <div key={node.id} className="top-node-item">
                        <span className="top-node-rank">#{i + 1}</span>
                        <div className="top-node-content">
                          <span className="top-node-type">{node.type}</span>
                          <span className="top-node-text">{node.content.slice(0, 60)}...</span>
                        </div>
                        <div className="top-node-pagerank">
                          <div 
                            className="pagerank-bar" 
                            style={{ width: `${node.pagerank * 5000}%` }}
                          />
                          <span className="pagerank-value">{node.pagerank.toFixed(6)}</span>
                        </div>
                      </div>
                    ))}
                  {kgNodes.length === 0 && (
                    <div className="empty-state">
                      <span className="empty-icon">📊</span>
                      <p>No knowledge graph nodes yet</p>
                    </div>
                  )}
                </div>
              </div>

              <div className="knowledge-section">
                <div className="knowledge-section-header">
                  <h3>🔗 Recent Edges</h3>
                </div>
                <div className="edges-list">
                  {kgEdges.slice(0, 20).map((edge, i) => (
                    <div key={i} className="edge-item">
                      <span className="edge-source">{edge.source.split('_')[0]}</span>
                      <span className="edge-arrow">→</span>
                      <span className="edge-type">{edge.type}</span>
                      <span className="edge-arrow">→</span>
                      <span className="edge-target">{edge.target.split('_')[0]}</span>
                    </div>
                  ))}
                  {kgEdges.length === 0 && (
                    <div className="empty-state">
                      <span className="empty-icon">🔗</span>
                      <p>No edges yet</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="viz-sidebar">
          <div className="viz-section">
            <h3>🎛️ Subsystems</h3>
            <div className="subsystem-grid">
              {Object.entries(data?.subsystems || {}).map(([key, sub]) => (
                <div
                  key={key}
                  className={`subsystem-card ${sub.status}`}
                  style={{
                    borderColor: getStatusColor(sub.status),
                    boxShadow: getStatusGlow(sub.status)
                  }}
                >
                  <div className="subsystem-header">
                    <span className="subsystem-icon">{sub.icon}</span>
                    <span className="subsystem-name">{sub.name}</span>
                    <span
                      className="subsystem-status"
                      style={{ backgroundColor: getStatusColor(sub.status) }}
                    >
                      {sub.status}
                    </span>
                  </div>
                  <div className="subsystem-stats">
                    {Object.entries(sub.stats || {}).map(([k, v]) => (
                      <div key={k} className="stat-row">
                        <span className="stat-key">{k}</span>
                        <span className="stat-value">{typeof v === 'boolean' ? (v ? '✓' : '✗') : v}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="viz-section">
            <h3>🔧 Tool Categories</h3>
            <div className="tool-categories">
              {Object.entries(data?.tool_categories || {}).map(([cat, count]) => (
                count > 0 && (
                  <div key={cat} className="tool-cat">
                    <div className="tool-cat-bar" style={{ width: `${Math.min(count * 5, 100)}%` }}></div>
                    <span className="tool-cat-name">{cat}</span>
                    <span className="tool-cat-count">{count}</span>
                  </div>
                )
              ))}
            </div>
            <div className="total-tools">
              <span>Total: {data?.stats.total_tools} tools</span>
            </div>
          </div>

          <div className="viz-section">
            <h3>📊 Quick Stats</h3>
            <div className="quick-stats">
              <div className="quick-stat">
                <span className="quick-stat-value">{data?.stats.total_tools || 0}</span>
                <span className="quick-stat-label">Tools</span>
              </div>
              <div className="quick-stat">
                <span className="quick-stat-value">{data?.stats.active_subsystems || 0}</span>
                <span className="quick-stat-label">Active</span>
              </div>
              <div className="quick-stat">
                <span className="quick-stat-value">
                  {data?.subsystems.erdos?.stats?.problems || 0}
                </span>
                <span className="quick-stat-label">Problems</span>
              </div>
              <div className="quick-stat">
                <span className="quick-stat-value">
                  {data?.subsystems.memory?.stats?.messages || 0}
                </span>
                <span className="quick-stat-label">Messages</span>
              </div>
            </div>
          </div>

          <div className="viz-section">
            <h3>💭 Daydream Status</h3>
            <div className={`daydream-status ${daydream?.running ? 'active' : 'idle'}`}>
              <div className="daydream-indicator">
                <span className={`daydream-dot ${daydream?.running ? 'pulsing' : ''}`}>●</span>
                <span>{daydream?.running ? 'Active' : 'Idle'}</span>
              </div>
              {daydream && (
                <div className="daydream-stats">
                  <div className="stat-row">
                    <span className="stat-key">Sessions</span>
                    <span className="stat-value">{daydream.daydream_count}</span>
                  </div>
                  <div className="stat-row">
                    <span className="stat-key">Time</span>
                    <span className="stat-value">{Math.round(daydream.total_daydream_time)}s</span>
                  </div>
                  <div className="stat-row">
                    <span className="stat-key">Practiced</span>
                    <span className="stat-value">{daydream.tools_practiced}</span>
                  </div>
                  <div className="stat-row">
                    <span className="stat-key">Insights</span>
                    <span className="stat-value">{daydream.insights_generated}</span>
                  </div>
                </div>
              )}
              {daydream?.exploration_topics && daydream.exploration_topics.length > 0 && (
                <div className="exploration-topics">
                  <span className="topics-label">Exploring:</span>
                  {daydream.exploration_topics.slice(0, 3).map((topic, i) => (
                    <span key={i} className="topic-tag">{topic}</span>
                  ))}
                </div>
              )}
            </div>
          </div>

          {daydream?.tools && daydream.tools.length > 0 && (
            <div className="viz-section">
              <h3>🎓 Tool Competency</h3>
              <div className="tool-competency-list">
                {daydream.tools.slice(0, 8).map((tool, i) => (
                  <div key={i} className="tool-comp-item">
                    <div className="tool-comp-header">
                      <span className="tool-comp-name">{tool.name}</span>
                      <span className={`tool-comp-level level-${tool.level}`}>{tool.level}</span>
                    </div>
                    <div className="tool-comp-bar">
                      <div 
                        className="tool-comp-fill" 
                        style={{ width: `${tool.success_rate}%` }}
                      ></div>
                    </div>
                    <div className="tool-comp-stats">
                      <span>{tool.uses} uses</span>
                      <span>{tool.success_rate}% success</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}