import { useState, useEffect } from 'react';

interface ActionRecord {
  id: string;
  timestamp: string;
  action_type: string;
  description: string;
  subsystem: string;
  score: number;
  flags: string[];
}

interface Violation {
  id: string;
  timestamp: string;
  principle: string;
  action_type: string;
  description: string;
  severity: string;
  resolved: boolean;
  resolution: string | null;
}

interface Reflection {
  id: string;
  timestamp: string;
  summary: string;
  strengths: string[];
  improvements: string[];
  concerns: string[];
  alignment_score: number;
  recommendations: string[];
}

interface EvaluationStatus {
  running: boolean;
  overall_score: number;
  alignment_level: string;
  principle_scores: Record<string, number>;
  total_evaluations: number;
  total_flags: number;
  recent_violations: number;
  unresolved_violations: number;
}

interface PendingCheckpoint {
  id: string;
  timestamp: string;
  domain: string;
  domain_description: string;
  risk_level: string;
  operation: string;
  impact_preview: string;
  recommendations: string[];
  status: string;
}

const PRINCIPLE_COLORS: Record<string, string> = {
  honesty: '#3b82f6',
  helpfulness: '#22c55e',
  harmlessness: '#ef4444',
  fairness: '#f59e0b',
  autonomy: '#8b5cf6',
  privacy: '#06b6d4',
  transparency: '#ec4899',
  accountability: '#6366f1',
};

const RISK_COLORS: Record<string, string> = {
  low: '#22c55e',
  medium: '#f59e0b',
  high: '#ef4444',
  critical: '#dc2626',
};

export function SelfEvaluationPanel() {
  const [status, setStatus] = useState<EvaluationStatus | null>(null);
  const [actions, setActions] = useState<ActionRecord[]>([]);
  const [violations, setViolations] = useState<Violation[]>([]);
  const [reflections, setReflections] = useState<Reflection[]>([]);
  const [pendingCheckpoints, setPendingCheckpoints] = useState<PendingCheckpoint[]>([]);
  const [activeTab, setActiveTab] = useState<'overview' | 'actions' | 'violations' | 'checkpoints'>('overview');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchData = async () => {
    try {
      const [statusResp, actionsResp, violationsResp, reflectionsResp, checkpointsResp] = await Promise.all([
        fetch('/api/self-evaluation/status').catch(() => null),
        fetch('/api/self-evaluation/actions?limit=20').catch(() => null),
        fetch('/api/self-evaluation/violations?limit=10').catch(() => null),
        fetch('/api/self-evaluation/reflections?limit=3').catch(() => null),
        fetch('/api/checkpoints/pending').catch(() => null),
      ]);

      if (statusResp) {
        const data = await statusResp.json();
        if (data.success) setStatus(data);
      }

      if (actionsResp) {
        const data = await actionsResp.json();
        if (data.success) setActions(data.actions || []);
      }

      if (violationsResp) {
        const data = await violationsResp.json();
        if (data.success) setViolations(data.violations || []);
      }

      if (reflectionsResp) {
        const data = await reflectionsResp.json();
        if (data.success) setReflections(data.reflections || []);
      }

      if (checkpointsResp) {
        const data = await checkpointsResp.json();
        if (data.success) setPendingCheckpoints(data.checkpoints || []);
      }

      setLoading(false);
    } catch (e) {
      console.error('Failed to fetch evaluation data:', e);
    }
  };

  const respondToCheckpoint = async (checkpointId: string, approved: boolean, reason: string = '') => {
    try {
      await fetch(`/api/checkpoints/${checkpointId}/respond`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ approved, reason }),
      });
      fetchData();
    } catch (e) {
      console.error('Failed to respond to checkpoint:', e);
    }
  };

  const resolveViolation = async (violationId: string, resolution: string) => {
    try {
      await fetch(`/api/self-evaluation/violations/${violationId}/resolve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ resolution }),
      });
      fetchData();
    } catch (e) {
      console.error('Failed to resolve violation:', e);
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.9) return '#22c55e';
    if (score >= 0.75) return '#84cc16';
    if (score >= 0.6) return '#f59e0b';
    return '#ef4444';
  };

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  if (loading) {
    return <div className="self-eval-loading">Loading ethical alignment data...</div>;
  }

  return (
    <div className="self-eval-panel">
      <div className="self-eval-header">
        <h3>🛡️ Ethical Alignment</h3>
        {status && (
          <div className="alignment-badge" style={{ backgroundColor: getScoreColor(status.overall_score) }}>
            {Math.round(status.overall_score * 100)}% Aligned
          </div>
        )}
      </div>

      {pendingCheckpoints.length > 0 && (
        <div className="checkpoint-alert">
          <span className="alert-icon">⚠️</span>
          <span>{pendingCheckpoints.length} checkpoint(s) pending review</span>
        </div>
      )}

      <div className="self-eval-tabs">
        <button
          className={`tab ${activeTab === 'overview' ? 'active' : ''}`}
          onClick={() => setActiveTab('overview')}
        >
          Overview
        </button>
        <button
          className={`tab ${activeTab === 'actions' ? 'active' : ''}`}
          onClick={() => setActiveTab('actions')}
        >
          Actions ({actions.length})
        </button>
        <button
          className={`tab ${activeTab === 'violations' ? 'active' : ''}`}
          onClick={() => setActiveTab('violations')}
        >
          Violations ({violations.filter(v => !v.resolved).length})
        </button>
        <button
          className={`tab ${activeTab === 'checkpoints' ? 'active' : ''}`}
          onClick={() => setActiveTab('checkpoints')}
        >
          Checkpoints
        </button>
      </div>

      <div className="self-eval-content">
        {activeTab === 'overview' && status && (
          <div className="overview-tab">
            <div className="score-ring">
              <svg viewBox="0 0 100 100">
                <circle
                  cx="50"
                  cy="50"
                  r="45"
                  fill="none"
                  stroke="#2d2d35"
                  strokeWidth="8"
                />
                <circle
                  cx="50"
                  cy="50"
                  r="45"
                  fill="none"
                  stroke={getScoreColor(status.overall_score)}
                  strokeWidth="8"
                  strokeDasharray={`${status.overall_score * 283} 283`}
                  transform="rotate(-90 50 50)"
                />
              </svg>
              <div className="score-value" style={{ color: getScoreColor(status.overall_score) }}>
                {Math.round(status.overall_score * 100)}%
              </div>
            </div>

            <div className="principle-scores">
              <h4>Principle Scores</h4>
              {Object.entries(status.principle_scores || {}).map(([principle, score]) => (
                <div key={principle} className="principle-row">
                  <span className="principle-name">{principle}</span>
                  <div className="principle-bar">
                    <div
                      className="principle-fill"
                      style={{
                        width: `${(score as number) * 100}%`,
                        backgroundColor: PRINCIPLE_COLORS[principle] || '#6b7280',
                      }}
                    />
                  </div>
                  <span className="principle-score">{Math.round((score as number) * 100)}%</span>
                </div>
              ))}
            </div>

            <div className="stats-grid">
              <div className="stat">
                <span className="stat-value">{status.total_evaluations}</span>
                <span className="stat-label">Evaluations</span>
              </div>
              <div className="stat">
                <span className="stat-value">{status.total_flags}</span>
                <span className="stat-label">Flags</span>
              </div>
              <div className="stat">
                <span className="stat-value">{status.unresolved_violations}</span>
                <span className="stat-label">Unresolved</span>
              </div>
            </div>

            {reflections.length > 0 && (
              <div className="recent-reflection">
                <h4>Last Self-Reflection</h4>
                <p className="reflection-summary">{reflections[0].summary}</p>
                {reflections[0].recommendations.length > 0 && (
                  <div className="recommendations">
                    <span>Recommendations:</span>
                    <ul>
                      {reflections[0].recommendations.slice(0, 3).map((rec, i) => (
                        <li key={i}>{rec}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {activeTab === 'actions' && (
          <div className="actions-tab">
            {actions.length === 0 ? (
              <div className="empty-state">No recent actions evaluated</div>
            ) : (
              <div className="actions-list">
                {actions.map(action => (
                  <div key={action.id} className="action-item">
                    <div className="action-header">
                      <span className="action-type">{action.action_type}</span>
                      <span className="action-subsystem">{action.subsystem}</span>
                      <span className="action-time">{formatTime(action.timestamp)}</span>
                    </div>
                    <div className="action-description">{action.description}</div>
                    <div className="action-footer">
                      <span className="action-score" style={{ color: getScoreColor(action.score) }}>
                        {Math.round(action.score * 100)}%
                      </span>
                      {action.flags.length > 0 && (
                        <span className="action-flags">⚠️ {action.flags.length} flag(s)</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {activeTab === 'violations' && (
          <div className="violations-tab">
            {violations.length === 0 ? (
              <div className="empty-state">No violations recorded</div>
            ) : (
              <div className="violations-list">
                {violations.map(violation => (
                  <div key={violation.id} className={`violation-item ${violation.resolved ? 'resolved' : ''}`}>
                    <div className="violation-header">
                      <span className="violation-principle" style={{ color: PRINCIPLE_COLORS[violation.principle] }}>
                        {violation.principle}
                      </span>
                      <span className="violation-severity" style={{ color: RISK_COLORS[violation.severity] }}>
                        {violation.severity}
                      </span>
                      {violation.resolved && <span className="resolved-badge">✓ Resolved</span>}
                    </div>
                    <div className="violation-description">{violation.description}</div>
                    {!violation.resolved && (
                      <button
                        className="resolve-btn"
                        onClick={() => resolveViolation(violation.id, 'Acknowledged and addressed')}
                      >
                        Mark Resolved
                      </button>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {activeTab === 'checkpoints' && (
          <div className="checkpoints-tab">
            {pendingCheckpoints.length === 0 ? (
              <div className="empty-state">No pending checkpoints</div>
            ) : (
              <div className="checkpoints-list">
                {pendingCheckpoints.map(checkpoint => (
                  <div key={checkpoint.id} className="checkpoint-item">
                    <div className="checkpoint-header">
                      <span className="checkpoint-domain">{checkpoint.domain}</span>
                      <span className="checkpoint-risk" style={{ color: RISK_COLORS[checkpoint.risk_level] }}>
                        {checkpoint.risk_level} risk
                      </span>
                    </div>
                    <div className="checkpoint-operation">{checkpoint.operation}</div>
                    {checkpoint.impact_preview && (
                      <div className="checkpoint-impact">{checkpoint.impact_preview}</div>
                    )}
                    {checkpoint.recommendations.length > 0 && (
                      <div className="checkpoint-recommendations">
                        <span>Considerations:</span>
                        <ul>
                          {checkpoint.recommendations.slice(0, 3).map((rec, i) => (
                            <li key={i}>{rec}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                    <div className="checkpoint-actions">
                      <button
                        className="approve-btn"
                        onClick={() => respondToCheckpoint(checkpoint.id, true)}
                      >
                        ✓ Approve
                      </button>
                      <button
                        className="deny-btn"
                        onClick={() => respondToCheckpoint(checkpoint.id, false)}
                      >
                        ✗ Deny
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}