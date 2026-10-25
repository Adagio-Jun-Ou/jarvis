import React, { useState, useEffect } from 'react';
import { useDashboard } from '../contexts/DashboardContext';

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

interface InsightsCardProps {
  compact?: boolean;
  maxInsights?: number;
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

export const InsightsCard: React.FC<InsightsCardProps> = ({ compact = false, maxInsights = 5 }) => {
  const [insights, setInsights] = useState<Insight[]>([]);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [totalCount, setTotalCount] = useState(0);
  const dashboardData = useDashboard();

  useEffect(() => {
    if (dashboardData.data?.recent_thoughts?.thoughts) {
      setInsights(dashboardData.data.recent_thoughts.thoughts.slice(0, maxInsights));
      setTotalCount(dashboardData.data.recent_thoughts.thoughts.length);
      setLoading(false);
    }
  }, [dashboardData.data, maxInsights]);

  const displayedInsights = insights.slice(0, maxInsights);

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

  if (loading) {
    return (
      <div className="insights-card loading">
        <div className="insights-header">
          <span className="insights-title">💡 Insights</span>
        </div>
        <div className="insights-loading">Loading...</div>
      </div>
    );
  }

  if (insights.length === 0) {
    return (
      <div className="insights-card empty">
        <div className="insights-header">
          <span className="insights-title">💡 Insights</span>
          <span className="insights-count">0</span>
        </div>
        <div className="insights-empty">
          <span className="empty-icon">💭</span>
          <span className="empty-text">No insights yet. Jarvis will discover insights through daydreaming and problem-solving.</span>
        </div>
      </div>
    );
  }

  return (
    <div className={`insights-card ${compact ? 'compact' : ''}`}>
      <div className="insights-header">
        <span className="insights-title">💡 Insights</span>
        <span className="insights-count">{totalCount}</span>
      </div>

      <div className="insights-list">
        {displayedInsights.map(insight => (
          <div
            key={insight.id}
            className={`insight-item ${expanded === insight.id ? 'expanded' : ''}`}
            onClick={() => setExpanded(expanded === insight.id ? null : insight.id)}
          >
            <div className="insight-top">
              <span className="insight-source">
                <span className="source-icon">{sourceIcons[insight.source] || '💡'}</span>
                <span className="source-label">{sourceLabels[insight.source] || insight.source}</span>
              </span>
              <span
                className="insight-type-badge"
                style={{ background: typeColors[insight.type] || typeColors.unknown }}
              >
                {(insight.type || 'unknown').replace(/_/g, ' ')}
              </span>
            </div>

              <div className="insight-content">
                {insight.title && <div className="insight-title">{insight.title}</div>}
                <div className="insight-text">
                  {expanded === insight.id ? insight.content : (insight.summary || insight.content)}
                </div>
                {insight.problem && expanded === insight.id && (
                  <div className="insight-problem">
                    <span className="problem-label">Problem:</span> {insight.problem}
                  </div>
                )}
              </div>

              <div className="insight-footer">
                <div className="insight-meta">
                  {insight.category && (
                    <span className="insight-category">{insight.category}</span>
                  )}
                  {insight.content && insight.content.length > 200 && (
                    <span className="insight-expand-hint">{expanded === insight.id ? '▲ less' : '▼ more'}</span>
                  )}
                </div>
                {insight.timestamp && (
                  <span className="insight-time">{formatTime(insight.timestamp)}</span>
                )}
              </div>
          </div>
        ))}
      </div>

      {insights.length > maxInsights && (
        <div className="insights-more">
          +{insights.length - maxInsights} more insights
        </div>
      )}
    </div>
  );
};

export default InsightsCard;