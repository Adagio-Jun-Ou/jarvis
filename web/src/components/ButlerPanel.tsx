import { useState, useEffect, useCallback } from 'react';
import { useDashboard } from '../contexts/DashboardContext';
import './ButlerPanel.css';

interface PresenceState {
  presence: string;
  idle_seconds: number;
  focus_score: number;
  session_minutes: number;
  return_count: number;
  consecutive_messages: number;
  deep_work_minutes: number;
  distraction_count: number;
  observations: number;
  pending_checkins: number;
  last_transition: string | null;
  rhythm: {
    peak_hours: number[];
    typical_session_minutes: number;
    total_sessions: number;
    deep_work_periods: number;
    breaks_taken: number;
    longest_focus_minutes: number;
  };
}

interface Anticipation {
  description: string;
  confidence: number;
  source: string;
  reasoning: string;
  suggested_action: string;
}

interface Observation {
  type: string;
  category: string;
  content: string;
  salience: number;
  age_seconds: number;
}

interface Habit {
  type: string;
  description: string;
  frequency: number;
  confidence: number;
  time_of_day: number[];
  triggers: string[];
}

interface SentimentState {
  trajectory_trend: string;
  trajectory_confidence: number;
  is_escalating: boolean;
  recent_sentiment: string;
  recent_intensity: number;
  recent_needs: string[];
  total_analyzed: number;
  escalation_count: number;
  de_escalation_count: number;
}

interface PersonalityState {
  mode: string;
  formality: string;
  energy: number;
  verbosity: number;
  warmth: number;
  humor: number;
  proactivity: number;
  patience: number;
  assertiveness: number;
  preferences_learned: number;
  adaptation_count: number;
  recent_interactions: number;
  user_sentiment: string;
  user_energy: string;
}

interface ProactiveState {
  active_patterns: number;
  pending_interventions: number;
  acceptance_rate: number;
  recent_actions: number;
  recent_errors: number;
  outcomes_tracked: number;
}

interface OrchestratorState {
  actions_proposed: number;
  acceptance_rate: number;
  rules_active: number;
  rules_total: number;
  pending_actions: number;
  last_coordination: number;
  cross_module_inferences: number;
  rule_stats: Record<string, { enabled: boolean; acceptances: number; dismissals: number }>;
  coordination_context: string;
  interruption: string | null;
}

interface SessionState {
  in_session: boolean;
  session_count: number;
  turn_count: number;
  errors: number;
  tools_used: number;
  session_duration: number;
  idle_seconds: number;
  topics: string[];
  peak_mode: string;
}

type ButlerTab = 'presence' | 'anticipation' | 'observations' | 'personality' | 'proactive' | 'orchestrator' | 'sentiment' | 'session';

const PRESENCE_ICONS: Record<string, string> = {
  active: '🟢', focused: '🎯', typing: '⌨️', idle: '⏸️',
  away: '🌙', returned: '👋', distracted: '👀',
};

const MODE_ICONS: Record<string, string> = {
  butler: '🎩', collaborator: '🤝', mentor: '🎓', companion: '☕',
};

const MODE_COLORS: Record<string, string> = {
  butler: '#14b8a6', collaborator: '#3b82f6', mentor: '#8b5cf6', companion: '#f59e0b',
};

const MODE_DESCRIPTIONS: Record<string, string> = {
  butler: 'Anticipatory, precise, service-oriented',
  collaborator: 'Peer discussion, building together',
  mentor: 'Patient, thorough, explanatory',
  companion: 'Warm, present, unhurried',
};

const FORMALITY_LABELS: Record<string, string> = { formal: 'Formal', professional: 'Professional', casual: 'Casual', intimate: 'Intimate' };

export function ButlerPanel() {
  const dashboardData = useDashboard();
  const [presence, setPresence] = useState<PresenceState | null>(null);
  const [anticipations, setAnticipations] = useState<Anticipation[]>([]);
  const [observations, setObservations] = useState<Observation[]>([]);
  const [habits, setHabits] = useState<Habit[]>([]);
  const [personality, setPersonality] = useState<PersonalityState | null>(null);
  const [proactive, setProactive] = useState<ProactiveState | null>(null);
  const [orchestrator, setOrchestrator] = useState<OrchestratorState | null>(null);
  const [sentiment, setSentiment] = useState<SentimentState | null>(null);
  const [session, setSession] = useState<SessionState | null>(null);
  const [activeTab, setActiveTab] = useState<ButlerTab>('presence');

  useEffect(() => {
    if (!dashboardData.data) return;
    const d = dashboardData.data;
    if (d.presence) setPresence(d.presence);
    if (d.anticipation?.pending_list) setAnticipations(d.anticipation.pending_list);
    if (d.autonomous_observer?.habits_list) setHabits(d.autonomous_observer.habits_list);
  }, [dashboardData.data]);

  const fetchButlerData = useCallback(async () => {
    try {
      const [obsResp, persResp, proResp, orchResp, sessResp] = await Promise.all([
        fetch('/api/observer/recent?limit=10').catch(() => null),
        fetch('/api/butler_personality/state').catch(() => null),
        fetch('/api/proactive_intelligence/state').catch(() => null),
        fetch('/api/butler_orchestrator/state').catch(() => null),
        fetch('/api/butler_session/state').catch(() => null),
      ]);
      if (obsResp?.ok) { const d = await obsResp.json(); if (d.success) setObservations(d.observations || []); }
      if (persResp?.ok) { const d = await persResp.json(); if (d.success && d.personality) setPersonality(d.personality as PersonalityState); }
      if (proResp?.ok) { const d = await proResp.json(); if (d.success && d.proactive) setProactive(d.proactive as ProactiveState); }
      if (orchResp?.ok) {
        const d = await orchResp.json();
        if (d.success && d.orchestrator) {
          const orchData = d.orchestrator as OrchestratorState;
          const ctxResp = await fetch('/api/butler_orchestrator/context').catch(() => null);
          if (ctxResp?.ok) { const cd = await ctxResp.json(); if (cd.success) orchData.coordination_context = cd.context || ''; }
          const intResp = await fetch('/api/butler_orchestrator/interruption').catch(() => null);
          if (intResp?.ok) { const id = await intResp.json(); if (id.success) orchData.interruption = id.interruption || null; }
          setOrchestrator(orchData);
        }
      }
      if (sessResp?.ok) { const d = await sessResp.json(); if (d.success && d.session) setSession(d.session as SessionState); }
      const sentResp = await fetch('/api/sentiment/state').catch(() => null);
      if (sentResp?.ok) { const d = await sentResp.json(); if (d.success && d.sentiment) setSentiment(d.sentiment as SentimentState); }
    } catch {}
  }, []);

  useEffect(() => {
    fetchButlerData();
    const interval = setInterval(fetchButlerData, 15000);
    return () => clearInterval(interval);
  }, [fetchButlerData]);

  const presenceStatus = presence?.presence || 'idle';
  const presenceIcon = PRESENCE_ICONS[presenceStatus] || '⏸️';
  const focusScore = presence?.focus_score || 0;
  const idleSeconds = presence?.idle_seconds || 0;

  const formatTime = (seconds: number) => {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
    return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
  };

  const modeIcon = MODE_ICONS[personality?.mode || 'butler'] || '🎩';
  const modeColor = MODE_COLORS[personality?.mode || 'butler'] || '#14b8a6';

  return (
    <div className="butler-panel">
      <div className="butler-header">
        <h3>Butler Intelligence</h3>
        <div className="butler-presence-pill">
          <span className={`presence-status-dot ${presenceStatus}`} />
          <span className="presence-status-text">
            {presenceStatus === 'active' ? 'With you' :
             presenceStatus === 'focused' ? 'Attentive' :
             presenceStatus === 'typing' ? 'Listening' :
             presenceStatus === 'idle' ? 'Standing by' :
             presenceStatus === 'away' ? 'On standby' :
             presenceStatus === 'returned' ? 'Welcome back' :
             presenceStatus.charAt(0).toUpperCase() + presenceStatus.slice(1)}
          </span>
        </div>
      </div>

      <div className="butler-tabs">
        <button className={`butler-tab ${activeTab === 'presence' ? 'active' : ''}`} onClick={() => setActiveTab('presence')}>
          Presence
        </button>
        <button className={`butler-tab ${activeTab === 'anticipation' ? 'active' : ''}`} onClick={() => setActiveTab('anticipation')}>
          Anticipation {anticipations.length > 0 && <span className="tab-badge">{anticipations.length}</span>}
        </button>
        <button className={`butler-tab ${activeTab === 'observations' ? 'active' : ''}`} onClick={() => setActiveTab('observations')}>
          Observations {observations.length > 0 && <span className="tab-badge obs">{observations.length}</span>}
        </button>
        <button className={`butler-tab ${activeTab === 'personality' ? 'active' : ''}`} onClick={() => setActiveTab('personality')}>
          Personality
        </button>
        <button className={`butler-tab ${activeTab === 'proactive' ? 'active' : ''}`} onClick={() => setActiveTab('proactive')}>
          Proactive {(proactive?.active_patterns ?? 0) > 0 && <span className="tab-badge">{proactive?.active_patterns}</span>}
        </button>
        <button className={`butler-tab ${activeTab === 'orchestrator' ? 'active' : ''}`} onClick={() => setActiveTab('orchestrator')}>
          Orchestrator
        </button>
        <button className={`butler-tab ${activeTab === 'sentiment' ? 'active' : ''}`} onClick={() => setActiveTab('sentiment')}>
          Sentiment {sentiment?.is_escalating && <span className="tab-badge">!</span>}
        </button>
        <button className={`butler-tab ${activeTab === 'session' ? 'active' : ''}`} onClick={() => setActiveTab('session')}>
          Session
        </button>
      </div>

      <div className="butler-content">
        {activeTab === 'presence' && (
          <div className="presence-tab">
            <div className="presence-main">
              <div className="presence-icon-large">{presenceIcon}</div>
              <div className="presence-details">
                <div className="presence-state-label">
                  {presenceStatus.charAt(0).toUpperCase() + presenceStatus.slice(1)}
                </div>
                <div className="presence-meta">
                  {idleSeconds > 0 && <span className="meta-item">Idle: {formatTime(idleSeconds)}</span>}
                  <span className="meta-item">Session: {formatTime((presence?.session_minutes || 0) * 60)}</span>
                  <span className="meta-item">Returns: {presence?.return_count || 0}</span>
                  {(presence?.deep_work_minutes || 0) > 0 && <span className="meta-item">Deep Work: {formatTime((presence?.deep_work_minutes || 0) * 60)}</span>}
                  {(presence?.distraction_count || 0) > 0 && <span className="meta-item">Distractions: {(presence?.distraction_count || 0)}</span>}
                </div>
              </div>
            </div>

            <div className="focus-bar-container">
              <div className="focus-bar-label">
                <span>Focus</span>
                <span>{Math.round(focusScore * 100)}%</span>
              </div>
              <div className="focus-bar">
                <div className="focus-bar-fill" style={{ width: `${focusScore * 100}%` }} />
              </div>
            </div>

            {presence?.rhythm && (
              <div className="rhythm-section">
                <div className="rhythm-label">Session Rhythm</div>
                <div className="rhythm-stats">
                  <span className="rhythm-stat">Peak: {presence.rhythm.peak_hours.map((h: number) => `${h}:00`).join(', ') || 'Learning...'}</span>
                  <span className="rhythm-stat">Sessions: {presence.rhythm.total_sessions}</span>
                  <span className="rhythm-stat">Avg: {presence.rhythm.typical_session_minutes}min</span>
                  {(presence.rhythm.deep_work_periods || 0) > 0 && <span className="rhythm-stat">Deep Work: {presence.rhythm.deep_work_periods}</span>}
                  {(presence.rhythm.longest_focus_minutes || 0) > 0 && <span className="rhythm-stat">Longest: {presence.rhythm.longest_focus_minutes}min</span>}
                </div>
              </div>
            )}

            {habits.length > 0 && (
              <div className="habits-section">
                <div className="habits-label">Learned Habits</div>
                {habits.slice(0, 3).map((h, i) => (
                  <div key={i} className="habit-item">
                    <span className="habit-type">{h.type}</span>
                    <span className="habit-desc">{h.description}</span>
                    <span className="habit-freq">{h.frequency}x</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {activeTab === 'anticipation' && (
          <div className="anticipation-tab">
            {anticipations.length === 0 ? (
              <div className="empty-state">No active anticipations — learning your patterns...</div>
            ) : (
              anticipations.map((a, i) => (
                <div key={i} className="anticipation-card">
                  <div className="anticipation-header">
                    <span className="anticipation-icon">🎯</span>
                    <span className="anticipation-desc">{a.description}</span>
                    <span className="anticipation-confidence">{Math.round(a.confidence * 100)}%</span>
                  </div>
                  <div className="anticipation-reasoning">{a.reasoning}</div>
                  {a.suggested_action && (
                    <div className="anticipation-action">
                      <span className="action-label">Suggested:</span> {a.suggested_action}
                    </div>
                  )}
                  <div className="anticipation-source">
                    <span className={`source-tag ${a.source}`}>{a.source}</span>
                  </div>
                </div>
              ))
            )}
          </div>
        )}

        {activeTab === 'observations' && (
          <div className="observations-tab">
            {observations.length === 0 ? (
              <div className="empty-state">No observations yet — building awareness...</div>
            ) : (
              observations.map((o, i) => (
                <div key={i} className={`observation-item salience-${o.salience}`}>
                  <div className="obs-type-tag">{o.type}</div>
                  <div className="obs-category-tag">{o.category}</div>
                  <div className="obs-content">{o.content}</div>
                  <div className="obs-age">{formatTime(o.age_seconds)} ago</div>
                </div>
              ))
            )}
          </div>
        )}

        {activeTab === 'personality' && personality && (
          <div className="personality-tab">
            <div className="personality-mode-card" style={{ borderColor: modeColor }}>
              <div className="personality-mode-icon">{modeIcon}</div>
              <div className="personality-mode-detail">
                <div className="personality-mode-name" style={{ color: modeColor }}>
                  {(personality.mode || 'butler').charAt(0).toUpperCase() + (personality.mode || 'butler').slice(1)} Mode
                </div>
                <div className="personality-mode-subtitle">
                  {MODE_DESCRIPTIONS[personality.mode || 'butler'] || 'Adaptive assistant'}
                </div>
              </div>
            </div>

            <div className="personality-traits">
              <div className="trait-bar-row">
                <span className="trait-label">Warmth</span>
                <div className="trait-bar"><div className="trait-bar-fill warmth" style={{ width: `${(personality.warmth || 0.5) * 100}%` }} /></div>
                <span className="trait-value">{Math.round((personality.warmth || 0.5) * 100)}%</span>
              </div>
              <div className="trait-bar-row">
                <span className="trait-label">Humor</span>
                <div className="trait-bar"><div className="trait-bar-fill humor" style={{ width: `${(personality.humor || 0.3) * 100}%` }} /></div>
                <span className="trait-value">{Math.round((personality.humor || 0.3) * 100)}%</span>
              </div>
              <div className="trait-bar-row">
                <span className="trait-label">Proactivity</span>
                <div className="trait-bar"><div className="trait-bar-fill proactivity" style={{ width: `${(personality.proactivity || 0.5) * 100}%` }} /></div>
                <span className="trait-value">{Math.round((personality.proactivity || 0.5) * 100)}%</span>
              </div>
              <div className="trait-bar-row">
                <span className="trait-label">Patience</span>
                <div className="trait-bar"><div className="trait-bar-fill patience" style={{ width: `${(personality.patience || 0.5) * 100}%` }} /></div>
                <span className="trait-value">{Math.round((personality.patience || 0.5) * 100)}%</span>
              </div>
              <div className="trait-bar-row">
                <span className="trait-label">Assertiveness</span>
                <div className="trait-bar"><div className="trait-bar-fill assertiveness" style={{ width: `${(personality.assertiveness || 0.5) * 100}%` }} /></div>
                <span className="trait-value">{Math.round((personality.assertiveness || 0.5) * 100)}%</span>
              </div>
            </div>

            <div className="personality-stats">
              <div className="stat-pill">Formality: <b>{FORMALITY_LABELS[personality.formality || 'professional'] || personality.formality || 'Professional'}</b></div>
              <div className="stat-pill">Verbosity: <b>{['Terse', 'Concise', 'Moderate', 'Detailed'][personality.verbosity ?? 2]}</b></div>
              <div className="stat-pill">Energy: <b>{['Low', 'Moderate', 'High', 'Intense'][personality.energy ?? 1]}</b></div>
              <div className="stat-pill">Sentiment: <b>{personality.user_sentiment || 'neutral'}</b></div>
              <div className="stat-pill">Energy Level: <b>{personality.user_energy || 'moderate'}</b></div>
              <div className="stat-pill">Prefs: <b>{personality.preferences_learned || 0}</b></div>
              <div className="stat-pill">Adaptations: <b>{personality.adaptation_count || 0}</b></div>
            </div>
          </div>
        )}

        {activeTab === 'personality' && !personality && (
          <div className="empty-state">Personality engine initializing...</div>
        )}

        {activeTab === 'proactive' && proactive && (
          <div className="proactive-tab">
            <div className="proactive-grid">
              <div className="proactive-metric">
                <div className="metric-value">{proactive.active_patterns}</div>
                <div className="metric-label">Active Patterns</div>
              </div>
              <div className="proactive-metric">
                <div className="metric-value">{proactive.pending_interventions}</div>
                <div className="metric-label">Pending</div>
              </div>
              <div className="proactive-metric">
                <div className="metric-value">{Math.round(proactive.acceptance_rate * 100)}%</div>
                <div className="metric-label">Acceptance</div>
              </div>
              <div className="proactive-metric">
                <div className="metric-value">{proactive.recent_errors}</div>
                <div className="metric-label">Recent Errors</div>
              </div>
            </div>

            <div className="acceptance-bar-container">
              <div className="acceptance-bar-label">
                <span>Acceptance Rate</span>
                <span className={proactive.acceptance_rate > 0.7 ? 'rate-good' : proactive.acceptance_rate > 0.3 ? 'rate-ok' : 'rate-low'}>
                  {Math.round(proactive.acceptance_rate * 100)}%
                </span>
              </div>
              <div className="acceptance-bar">
                <div className="acceptance-bar-fill" style={{ width: `${proactive.acceptance_rate * 100}%` }} />
              </div>
            </div>

            <div className="proactive-stats">
              <span className="stat-pill">Actions tracked: <b>{proactive.recent_actions}</b></span>
              <span className="stat-pill">Outcomes: <b>{proactive.outcomes_tracked}</b></span>
            </div>
          </div>
        )}

        {activeTab === 'proactive' && !proactive && (
          <div className="empty-state">Proactive intelligence initializing...</div>
        )}

        {activeTab === 'orchestrator' && orchestrator && (
          <div className="orchestrator-tab">
            {orchestrator.interruption && (
              <div className="orch-interruption">
                {orchestrator.interruption}
              </div>
            )}
            <div className="orch-grid">
              <div className="orch-metric">
                <div className="metric-value">{orchestrator.actions_proposed}</div>
                <div className="metric-label">Actions</div>
              </div>
              <div className="orch-metric">
                <div className="metric-value">{orchestrator.cross_module_inferences}</div>
                <div className="metric-label">Inferences</div>
              </div>
              <div className="orch-metric">
                <div className="metric-value">{orchestrator.rules_active}/{orchestrator.rules_total || 8}</div>
                <div className="metric-label">Rules</div>
              </div>
              <div className="orch-metric">
                <div className="metric-value">{Math.round(orchestrator.acceptance_rate * 100)}%</div>
                <div className="metric-label">Accept%</div>
              </div>
            </div>

            {orchestrator.rule_stats && Object.keys(orchestrator.rule_stats).length > 0 && (
              <div className="orch-rules-section">
                <div className="orch-rules-label">Rule Acceptance</div>
                {Object.entries(orchestrator.rule_stats).map(([name, stats]) => (
                  <div key={name} className="orch-rule-row">
                    <span className="orch-rule-name">{name.replace(/_/g, ' ')}</span>
                    <span className={`orch-rule-status ${stats.enabled ? 'enabled' : 'disabled'}`}>{stats.enabled ? 'On' : 'Off'}</span>
                    <span className="orch-rule-accepts">{stats.acceptances}✓ {stats.dismissals}✗</span>
                  </div>
                ))}
              </div>
            )}

            {orchestrator.coordination_context && (
              <div className="orch-context-section">
                <div className="orch-context-label">Current Context</div>
                <div className="orch-context-text">{orchestrator.coordination_context}</div>
              </div>
            )}

            <div className="orch-status">
              <div className="orch-status-row">
                <span className="orch-status-label">Pending Actions</span>
                <span className="orch-status-value">{orchestrator.pending_actions}</span>
              </div>
              <div className="orch-status-row">
                <span className="orch-status-label">Last Coordination</span>
                <span className="orch-status-value">{orchestrator.last_coordination}s ago</span>
              </div>
              <div className="orch-status-row">
                <span className="orch-status-label">Acceptance Rate</span>
                <span className={`orch-status-value ${orchestrator.acceptance_rate > 0.7 ? 'rate-good' : orchestrator.acceptance_rate > 0.3 ? 'rate-ok' : 'rate-low'}`}>
                  {Math.round(orchestrator.acceptance_rate * 100)}%
                </span>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'orchestrator' && !orchestrator && (
          <div className="empty-state">Orchestrator initializing...</div>
        )}

        {activeTab === 'sentiment' && sentiment && (
          <div className="sentiment-tab">
            <div className="sentiment-main-row">
              <div className={`sentiment-badge ${sentiment.recent_sentiment}`}>
                {sentiment.recent_sentiment}
              </div>
              <div className="sentiment-info">
                <div className="sentiment-trend">
                  Trend: <span className={`trend-${sentiment.trajectory_trend}`}>
                    {sentiment.trajectory_trend}
                  </span>
                  <span className="trend-confidence">({Math.round(sentiment.trajectory_confidence * 100)}%)</span>
                </div>
                <div className="sentiment-intensity">
                  Intensity: {Math.round(sentiment.recent_intensity * 100)}%
                </div>
              </div>
            </div>

            <div className="sentiment-needs">
              {sentiment.recent_needs.length > 0 && (
                <div className="needs-label">Detected Needs</div>
              )}
              {sentiment.recent_needs.map((need, i) => (
                <span key={i} className="need-pill">{need}</span>
              ))}
            </div>

            <div className="sentiment-stats">
              <span className="stat-pill">Analyzed: <b>{sentiment.total_analyzed}</b></span>
              {(sentiment.escalation_count || 0) > 0 && <span className="stat-pill">Escalations: <b>{sentiment.escalation_count}</b></span>}
              {(sentiment.de_escalation_count || 0) > 0 && <span className="stat-pill">De-escalations: <b>{sentiment.de_escalation_count}</b></span>}
              {sentiment.is_escalating && (
                <span className="stat-pill escalation-warning">Escalating</span>
              )}
            </div>
          </div>
        )}

        {activeTab === 'sentiment' && !sentiment && (
          <div className="empty-state">Sentiment analyzer initializing...</div>
        )}

        {activeTab === 'session' && session && (
          <div className="session-tab">
            <div className="session-status-card">
              <div className={`session-indicator ${session.in_session ? 'active' : 'inactive'}`} />
              <div className="session-status-detail">
                <div className="session-status-label">{session.in_session ? 'Active Session' : 'No Active Session'}</div>
                <div className="session-session-num">Session #{session.session_count || 0}</div>
              </div>
            </div>

            <div className="session-grid">
              <div className="session-metric">
                <div className="metric-value">{session.turn_count || 0}</div>
                <div className="metric-label">Turns</div>
              </div>
              <div className="session-metric">
                <div className="metric-value">{formatTime(session.session_duration || 0)}</div>
                <div className="metric-label">Duration</div>
              </div>
              <div className="session-metric">
                <div className="metric-value">{session.errors || 0}</div>
                <div className="metric-label">Errors</div>
              </div>
              <div className="session-metric">
                <div className="metric-value">{session.tools_used || 0}</div>
                <div className="metric-label">Tools</div>
              </div>
            </div>

            {session.topics && session.topics.length > 0 && (
              <div className="session-topics">
                <div className="session-topics-label">Topics</div>
                <div className="session-topics-list">
                  {session.topics.map((topic, i) => (
                    <span key={i} className="topic-pill">{topic}</span>
                  ))}
                </div>
              </div>
            )}

            <div className="session-stats">
              <span className="stat-pill">Mode: <b>{(session.peak_mode || 'butler').charAt(0).toUpperCase() + (session.peak_mode || 'butler').slice(1)}</b></span>
              {(session.idle_seconds || 0) > 60 && <span className="stat-pill">Idle: <b>{formatTime(session.idle_seconds)}</b></span>}
            </div>
          </div>
        )}

        {activeTab === 'session' && !session && (
          <div className="empty-state">Session tracker initializing...</div>
        )}
      </div>
    </div>
  );
}