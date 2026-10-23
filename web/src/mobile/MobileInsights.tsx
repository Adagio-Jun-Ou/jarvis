import React, { useState, useEffect } from 'react';

interface Insight {
  id: string;
  source: string;
  type: string;
  category: string;
  problem: string;
  content: string;
  summary: string;
  title?: string;
  relevance: number;
  timestamp: string;
}

const sourceIcons: Record<string, string> = {
  insight_bridge: '💡',
  knowledge: '📚',
  erdos: '🔢',
  daydream: '💭',
};

const sourceLabels: Record<string, string> = {
  insight_bridge: 'Insight',
  knowledge: 'Knowledge',
  erdos: 'Erdős',
  daydream: 'Daydream',
};

const typeColors: Record<string, string> = {
  pattern_discovery: '#10b981',
  strategy_success: '#3b82f6',
  method_insight: '#8b5cf6',
  conjecture: '#f59e0b',
  counterexample: '#ef4444',
  cross_domain: '#06b6d4',
  math_discovery: '#ec4899',
  exploration: '#6366f1',
  unknown: '#6b7280',
  contribution: '#22c55e',
};

export const MobileInsights: React.FC = () => {
  const [insights, setInsights] = useState<Insight[]>([]);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const fetchInsights = async () => {
    try {
      const resp = await fetch('/api/insights/recent');
      const data = await resp.json();
      if (data.success) {
        setInsights(data.insights || []);
      }
    } catch (e) {
      console.error('Failed to fetch insights:', e);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchInsights();
    const interval = setInterval(fetchInsights, 60000);
    return () => clearInterval(interval);
  }, []);

  const handleRefresh = () => {
    setRefreshing(true);
    fetchInsights();
  };

  const formatTime = (ts: string) => {
    if (!ts) return '';
    try {
      const date = new Date(ts);
      const now = new Date();
      const diff = now.getTime() - date.getTime();
      const mins = Math.floor(diff / 60000);
      if (mins < 60) return `${mins}m`;
      const hours = Math.floor(mins / 60);
      if (hours < 24) return `${hours}h`;
      return date.toLocaleDateString();
    } catch {
      return '';
    }
  };

  if (loading) {
    return (
      <div className="insights-view">
        <div className="insights-header">
          <h2>💡 Insights</h2>
        </div>
        <div className="loading-state">
          <div className="loading-spinner" />
          <span>Loading insights...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="insights-view">
      <div className="insights-header">
        <h2>💡 Insights</h2>
        <button 
          className={`refresh-btn ${refreshing ? 'refreshing' : ''}`}
          onClick={handleRefresh}
          disabled={refreshing}
        >
          ↻
        </button>
      </div>

      <div className="insights-summary">
        <div className="summary-stat">
          <span className="summary-value">{insights.length}</span>
          <span className="summary-label">Total</span>
        </div>
        <div className="summary-stat">
          <span className="summary-value">{insights.filter(i => i.source === 'erdos').length}</span>
          <span className="summary-label">Math</span>
        </div>
        <div className="summary-stat">
          <span className="summary-value">{insights.filter(i => i.source === 'daydream').length}</span>
          <span className="summary-label">Dream</span>
        </div>
      </div>

      {insights.length === 0 ? (
        <div className="empty-insights">
          <div className="empty-icon">💭</div>
          <div className="empty-text">
            No insights yet. Jarvis discovers insights through daydreaming and problem-solving.
          </div>
        </div>
      ) : (
        <div className="insights-scroll">
          {insights.map(insight => (
            <div
              key={insight.id}
              className={`insight-card ${expanded === insight.id ? 'expanded' : ''}`}
              onClick={() => setExpanded(expanded === insight.id ? null : insight.id)}
            >
              <div className="insight-card-header">
                <div className="insight-source-info">
                  <span className="source-icon">{sourceIcons[insight.source] || '💡'}</span>
                  <span className="source-name">{sourceLabels[insight.source] || insight.source}</span>
                </div>
                <span
                  className="type-badge"
                  style={{ background: typeColors[insight.type] || typeColors.unknown }}
                >
                  {insight.type.replace(/_/g, ' ')}
                </span>
              </div>

                <div className="insight-card-content">
                  {insight.title && <div className="insight-card-title">{insight.title}</div>}
                  <div className="insight-card-text">
                    {expanded === insight.id ? insight.content : (insight.summary || insight.content)}
                  </div>
                  {insight.problem && expanded === insight.id && (
                    <div className="insight-problem-context">
                      <span className="problem-label">Problem:</span>
                      <span className="problem-text">{insight.problem}</span>
                    </div>
                  )}
                </div>

                <div className="insight-card-footer">
                  <div className="footer-left">
                    {insight.category && (
                      <span className="insight-category">{insight.category}</span>
                    )}
                    {insight.content && insight.content.length > 150 && (
                      <span className="expand-hint">{expanded === insight.id ? '▲' : '▼'}</span>
                    )}
                  </div>
                  {insight.timestamp && (
                    <span className="insight-time">{formatTime(insight.timestamp)}</span>
                  )}
                </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default MobileInsights;