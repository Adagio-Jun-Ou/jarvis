import { useState, useEffect } from 'react';
import { useDashboard } from '../contexts/DashboardContext';
import './InsightsTab.css';

interface Insight {
  id: string;
  source: string;
  type: string;
  category: string;
  content: string;
  summary: string;
  relevance: number;
  timestamp: string;
}

interface KnowledgeStats {
  nodes: number;
  edges: number;
  communities: number;
}

interface MemoryStats {
  conversation: { messages: number; turns: number };
  vector?: { available: boolean; vector_count?: number };
  knowledge_graph?: { available: boolean; nodes?: number; edges?: number };
}

const VIEW_MODES = ['recent', 'knowledge', 'memory'] as const;
type ViewMode = typeof VIEW_MODES[number];

const SOURCE_ICONS: Record<string, string> = {
  insight_bridge: '💡',
  knowledge: '📚',
  erdos: '🔢',
  daydream: '💭',
  self_refine: '🔧',
  evolution: '🧬',
};

const TYPE_COLORS: Record<string, string> = {
  pattern_discovery: '#10b981',
  strategy_success: '#3b82f6',
  method_insight: '#8b5cf6',
  conjecture: '#f59e0b',
  counterexample: '#ef4444',
  cross_domain: '#06b6d4',
  math_discovery: '#ec4899',
  exploration: '#6366f1',
  unknown: '#6b7280',
};

export function InsightsTab() {
  const [viewMode, setViewMode] = useState<ViewMode>('recent');
  const [insights, setInsights] = useState<Insight[]>([]);
  const [knowledgeStats, setKnowledgeStats] = useState<KnowledgeStats | null>(null);
  const [memoryStats, setMemoryStats] = useState<MemoryStats | null>(null);
  const [loading, setLoading] = useState(true);
  const dashboardData = useDashboard();

  useEffect(() => {
    if (!dashboardData.data) return;
    
    if (dashboardData.data.insights?.insights) {
      setInsights(dashboardData.data.insights.insights);
    }
    
    if (dashboardData.data.knowledge_graph) {
      setKnowledgeStats({
        nodes: dashboardData.data.knowledge_graph.nodes || 0,
        edges: dashboardData.data.knowledge_graph.edges || 0,
        communities: dashboardData.data.knowledge_graph.communities || 0,
      });
    }
    
    if (dashboardData.data.memory_stats?.stats) {
      setMemoryStats(dashboardData.data.memory_stats.stats);
    }
    
    setLoading(false);
  }, [dashboardData.data]);

  const formatTime = (ts: string) => {
    if (!ts) return '';
    try {
      const date = new Date(ts);
      const now = new Date();
      const diff = now.getTime() - date.getTime();
      const mins = Math.floor(diff / 60000);
      if (mins < 60) return `${mins}m ago`;
      const hours = Math.floor(mins / 60);
      if (hours < 24) return `${hours}h ago`;
      return date.toLocaleDateString();
    } catch {
      return '';
    }
  };

  return (
    <div className="insights-tab">
      <div className="insights-header">
        <h2>💡 Insights</h2>
        <div className="insights-tabs">
          {VIEW_MODES.map(mode => (
            <button
              key={mode}
              className={`insights-tab-btn ${viewMode === mode ? 'active' : ''}`}
              onClick={() => setViewMode(mode)}
            >
              {mode === 'recent' && 'Recent'}
              {mode === 'knowledge' && 'Knowledge Graph'}
              {mode === 'memory' && 'Memory'}
            </button>
          ))}
        </div>
      </div>

      <div className="insights-content">
        {viewMode === 'recent' && (
          <div className="insights-recent">
             {loading ? (
               <div className="insights-loading">Loading insights...</div>
             ) : insights.length === 0 ? (
               <div className="insights-empty">
                 <div className="insights-empty-icon">💡</div>
                 <div className="insights-empty-text">No insights yet</div>
                 <div className="insights-empty-hint">Insights will appear as Jarvis learns</div>
               </div>
             ) : (
               <>
                 <div className="insights-count">
                   Showing {insights.length} insights
                 </div>
                 <div className="insights-list">
                   {insights.map(insight => (
                     <div key={insight.id} className="insight-item">
                       <div className="insight-header">
                         <span className="insight-icon">{SOURCE_ICONS[insight.source] || '💡'}</span>
                         <span className="insight-type" style={{ color: TYPE_COLORS[insight.type] || '#6b7280' }}>
                           {(insight.type || 'unknown').replace(/_/g, ' ')}
                         </span>
                         <span className="insight-source">{insight.source}</span>
                         <span className="insight-time">{formatTime(insight.timestamp)}</span>
                       </div>
                       <div className="insight-content">
                         {insight.content}
                       </div>
                       {insight.relevance > 0 && (
                         <div className="insight-relevance">
                           <span className="relevance-label">Relevance:</span>
                           <span className="relevance-value">{insight.relevance.toFixed(2)}</span>
                         </div>
                       )}
                     </div>
                   ))}
                 </div>
               </>
             )}
          </div>
        )}

        {viewMode === 'knowledge' && (
          <div className="insights-knowledge">
            {knowledgeStats && (
              <div className="knowledge-stats">
                <div className="knowledge-stat">
                  <span className="knowledge-value">{knowledgeStats.nodes.toLocaleString()}</span>
                  <span className="knowledge-label">Nodes</span>
                </div>
                <div className="knowledge-stat">
                  <span className="knowledge-value">{knowledgeStats.edges.toLocaleString()}</span>
                  <span className="knowledge-label">Edges</span>
                </div>
                <div className="knowledge-stat">
                  <span className="knowledge-value">{knowledgeStats.communities}</span>
                  <span className="knowledge-label">Communities</span>
                </div>
              </div>
            )}
            <div className="knowledge-info">
              <div className="knowledge-info-icon">🕸️</div>
              <div className="knowledge-info-text">
                <div className="knowledge-info-title">Knowledge Graph</div>
                <div className="knowledge-info-desc">
                  A semantic network of interconnected concepts, insights, and relationships
                  discovered through the insight bridge and daydream engine.
                </div>
              </div>
            </div>
          </div>
        )}

        {viewMode === 'memory' && (
          <div className="insights-memory">
            {memoryStats && (
              <div className="memory-stats">
                <div className="memory-section">
                  <div className="memory-section-header">
                    <span className="memory-section-icon">💬</span>
                    <span className="memory-section-title">Conversation</span>
                  </div>
                  <div className="memory-stat-row">
                    <div className="memory-stat-item">
                      <span className="memory-stat-value">{memoryStats.conversation?.messages || 0}</span>
                      <span className="memory-stat-label">Messages</span>
                    </div>
                    <div className="memory-stat-item">
                      <span className="memory-stat-value">{memoryStats.conversation?.turns || 0}</span>
                      <span className="memory-stat-label">Turns</span>
                    </div>
                  </div>
                </div>

                {memoryStats.vector?.available && (
                  <div className="memory-section">
                    <div className="memory-section-header">
                      <span className="memory-section-icon">🔎</span>
                      <span className="memory-section-title">Vector Memory</span>
                    </div>
                    <div className="memory-stat-row">
                      <div className="memory-stat-item">
                        <span className="memory-stat-value">{memoryStats.vector.vector_count || 0}</span>
                        <span className="memory-stat-label">Vectors</span>
                      </div>
                    </div>
                  </div>
                )}

                {memoryStats.knowledge_graph?.available && (
                  <div className="memory-section">
                    <div className="memory-section-header">
                      <span className="memory-section-icon">🕸️</span>
                      <span className="memory-section-title">Knowledge Graph</span>
                    </div>
                    <div className="memory-stat-row">
                      <div className="memory-stat-item">
                        <span className="memory-stat-value">{memoryStats.knowledge_graph.nodes || 0}</span>
                        <span className="memory-stat-label">Nodes</span>
                      </div>
                      <div className="memory-stat-item">
                        <span className="memory-stat-value">{memoryStats.knowledge_graph.edges || 0}</span>
                        <span className="memory-stat-label">Edges</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}