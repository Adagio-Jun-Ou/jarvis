import { useState, useEffect, useRef } from 'react';
import { useDashboard } from '../contexts/DashboardContext';

interface SystemState {
  status: 'idle' | 'thinking' | 'working' | 'listening' | 'speaking' | 'daydreaming' | 'collaborating' | 'watching' | 'anticipating';
  activity: string;
  thought: string | null;
  subsystemActive: string | null;
  toolsUsed: string[];
  lastAction: string;
  daydreaming: boolean;
  idleTimeSeconds: number;
  callsPerMinute: number;
  turns: number;
  activeModel: string;
  contextUsage: number;
  proactiveMode: boolean;
  modelCoordinator: {
    activeModel: string;
    availableModels: string[];
    crossModelCalls: number;
    totalHandoffs: number;
  } | null;
  memoryProtection: {
    totalProtected: number;
    byLevel: { [key: string]: number };
  } | null;
  recentThoughts: { source: string; content: string }[];
  presenceState: string | null;
  anticipation: { description: string; confidence: number; reasoning: string } | null;
  notifiableObservations: number;
}

const STATUS_COLORS = {
  idle: { primary: '#6b7280', glow: 'rgba(107, 114, 128, 0.4)', pulse: 'rgba(107, 114, 128, 0.2)' },
  thinking: { primary: '#8b5cf6', glow: 'rgba(139, 92, 246, 0.6)', pulse: 'rgba(139, 92, 246, 0.3)' },
  working: { primary: '#f59e0b', glow: 'rgba(245, 158, 11, 0.6)', pulse: 'rgba(245, 158, 11, 0.3)' },
  listening: { primary: '#22c55e', glow: 'rgba(34, 197, 94, 0.6)', pulse: 'rgba(34, 197, 94, 0.3)' },
  speaking: { primary: '#3b82f6', glow: 'rgba(59, 130, 246, 0.6)', pulse: 'rgba(59, 130, 246, 0.3)' },
  daydreaming: { primary: '#06b6d4', glow: 'rgba(6, 182, 212, 0.6)', pulse: 'rgba(6, 182, 212, 0.3)' },
  collaborating: { primary: '#ec4899', glow: 'rgba(236, 72, 153, 0.6)', pulse: 'rgba(236, 72, 153, 0.3)' },
  watching: { primary: '#14b8a6', glow: 'rgba(20, 184, 166, 0.6)', pulse: 'rgba(20, 184, 166, 0.3)' },
  anticipating: { primary: '#f97316', glow: 'rgba(249, 115, 22, 0.6)', pulse: 'rgba(249, 115, 22, 0.3)' },
};

// Bi-model system: Kimi K2.5 (primary/chat) + GLM-5 (backend/reasoning)
const MODEL_COLORS: { [key: string]: string } = {
  'kimi-k2.5': '#ec4899',  // Pink - Primary/chat model
  'glm-5': '#8b5cf6',      // Purple - Backend/reasoning model
};

const PARTICLE_COUNT = 16;

interface Particle {
  id: number;
  angle: number;
  distance: number;
  size: number;
  speed: number;
  opacity: number;
  layer: number;
}

interface ModelIndicator {
  modelId: string;
  active: boolean;
  angle: number;
}

export function JarvisAvatar({ compact = false }: { compact?: boolean }) {
  const [state, setState] = useState<SystemState>({
    status: 'idle',
    activity: 'Initializing...',
    thought: '',
    subsystemActive: null,
    toolsUsed: [],
    lastAction: '',
    daydreaming: false,
    idleTimeSeconds: 0,
    callsPerMinute: 0,
    turns: 0,
    activeModel: 'kimi-k2.5',
    contextUsage: 0,
    proactiveMode: false,
    modelCoordinator: null,
    memoryProtection: null,
    recentThoughts: [],
    presenceState: null,
    anticipation: null,
    notifiableObservations: 0,
  });
  
  const [particles, setParticles] = useState<Particle[]>([]);
  const [modelIndicators, setModelIndicators] = useState<ModelIndicator[]>([]);
  const [rotation, setRotation] = useState(0);
  const [pulsePhase, setPulsePhase] = useState(0);
  const [dataPulse, setDataPulse] = useState(0);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | undefined>(undefined);

  useEffect(() => {
    const initialParticles: Particle[] = [];
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      initialParticles.push({
        id: i,
        angle: (Math.PI * 2 * i) / PARTICLE_COUNT,
        distance: 35 + Math.random() * 20,
        size: 1.5 + Math.random() * 3,
        speed: 0.3 + Math.random() * 0.8,
        opacity: 0.3 + Math.random() * 0.5,
        layer: Math.floor(i / 4),
      });
    }
    setParticles(initialParticles);
  }, []);

  useEffect(() => {
    let frame = 0;
    const animate = () => {
      frame++;
      setRotation(prev => prev + 0.3);
      setPulsePhase(prev => (prev + 0.04) % (Math.PI * 2));
      setDataPulse(prev => (prev + 0.02) % 1);
      
      setParticles(prev => prev.map(p => ({
        ...p,
        angle: p.angle + 0.015 * p.speed * (p.layer % 2 === 0 ? 1 : -1),
        distance: 35 + Math.sin(frame * 0.015 + p.id * 0.7) * 12,
        opacity: 0.3 + Math.sin(frame * 0.025 + p.id * 0.3) * 0.35,
      })));
      
      animationRef.current = requestAnimationFrame(animate);
    };
    
    animate();
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  const dashboardData = useDashboard();

  useEffect(() => {
    if (!dashboardData.data) return;
    
    const data = dashboardData.data;
    let newState: Partial<SystemState> = {};
    
    if (data.status) {
      newState.callsPerMinute = data.status.calls_per_minute || 0;
      newState.turns = data.status.turns || 0;
      newState.activeModel = data.status.model || 'kimi-k2.5';
      
      if (data.status.turns > 0) {
        newState.status = 'thinking';
        newState.activity = 'Processing request...';
      }
      if (data.status.is_idle && data.daydream?.can_daydream) {
        newState.status = 'daydreaming';
        newState.activity = 'Exploring ideas...';
        newState.daydreaming = true;
      }
      if (data.status.memory_used_mb) {
        newState.contextUsage = Math.min(100, (data.status.memory_used_mb / 24000) * 100);
      }
    }
    
    if (data.jarvis_state?.state) {
      const s = data.jarvis_state.state;
      if (s.status !== 'idle') {
        newState.status = s.status;
      }
      if (s.activity && s.activity !== 'Ready') newState.activity = s.activity;
      if (s.thought) newState.thought = s.thought;
      if (s.subsystem_active) newState.subsystemActive = s.subsystem_active;
      newState.daydreaming = s.daydreaming || false;
      newState.idleTimeSeconds = s.idle_time_seconds || 0;
    }
    
    if (data.daydream?.running && data.daydream?.can_daydream && !newState.status) {
      newState.status = 'daydreaming';
      newState.activity = 'Exploring ideas...';
      newState.daydreaming = true;
      if (data.daydream.exploration_topics && data.daydream.exploration_topics.length > 0) {
        const topic = data.daydream.exploration_topics[data.daydream.exploration_topics.length - 1];
        newState.thought = `Exploring: ${topic}`;
      }
    }
    
    if (data.models_coordinator) {
      const mc = data.models_coordinator;
      newState.modelCoordinator = {
        activeModel: mc.active_model,
        availableModels: mc.connected_models || mc.available_models || [],
        crossModelCalls: mc.cross_model_calls || 0,
        totalHandoffs: mc.total_handoffs || 0,
      };
      
      if (mc.cross_model_calls > 0 || mc.total_handoffs > 0) {
        newState.status = 'collaborating';
        newState.activity = `Models collaborating (${mc.cross_model_calls} calls)`;
      }
      
      const connectedModels = mc.connected_models || mc.available_models || [];
      const indicators: ModelIndicator[] = connectedModels.map((m: string, i: number) => ({
        modelId: m,
        active: m === mc.active_model,
        angle: (Math.PI * 2 * i) / connectedModels.length,
      }));
      setModelIndicators(indicators);
    }
    
    if (data.proactive) {
      newState.proactiveMode = data.proactive.is_proactive_mode || false;
      if (data.proactive.total_calls > 0 && !newState.status) {
        newState.activity = `Proactive: ${data.proactive.total_calls} tasks`;
      }
    }
    
    if (data.recent_thoughts?.thoughts && data.recent_thoughts.thoughts.length > 0) {
      newState.recentThoughts = data.recent_thoughts.thoughts.slice(0, 3);
    }
    
    if (Object.keys(newState).length > 0) {
      setState(prev => ({ ...prev, ...newState }));
    }
  }, [dashboardData.data]);

  useEffect(() => {
    const pollPresence = async () => {
      try {
        const [presenceResp, anticipationResp, observerResp] = await Promise.all([
          fetch('/api/presence/state').catch(() => null),
          fetch('/api/anticipation/pending').catch(() => null),
          fetch('/api/observer/state').catch(() => null),
        ]);

        const updates: Partial<SystemState> = {};

        if (presenceResp && presenceResp.ok) {
          const pd = await presenceResp.json();
          if (pd.success && pd.presence) {
            updates.presenceState = pd.presence.presence;
            if (pd.presence.presence === 'focused' && (!state.status || state.status === 'idle' || state.status === 'listening')) {
              updates.status = 'watching';
              updates.activity = 'Watching attentively...';
            } else if (pd.presence.presence === 'away') {
              updates.status = 'daydreaming';
              updates.activity = 'User away — background processing';
            }
          }
        }

        if (anticipationResp && anticipationResp.ok) {
          const ad = await anticipationResp.json();
          if (ad.success && ad.pending && ad.pending.length > 0) {
            updates.anticipation = ad.pending[0];
            if (!updates.status && state.status === 'idle') {
              updates.status = 'anticipating';
              updates.activity = ad.pending[0].description;
            }
          }
        }

        if (observerResp && observerResp.ok) {
          const od = await observerResp.json();
          if (od.success && od.observer) {
            updates.notifiableObservations = od.observer.notifiable || 0;
          }
        }

        if (Object.keys(updates).length > 0) {
          setState(prev => ({ ...prev, ...updates }));
        }
      } catch {}
    };

    pollPresence();
    const interval = setInterval(pollPresence, 10000);
    return () => clearInterval(interval);
  }, [state.status]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const colors = STATUS_COLORS[state.status];
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const pulseSize = 1 + Math.sin(pulsePhase) * 0.08;
    const baseRadius = compact ? 28 : 45;
    const radius = baseRadius * pulseSize;
    
    // Outer glow based on calls per minute
    if (state.callsPerMinute > 0) {
      const glowIntensity = Math.min(1, state.callsPerMinute / 10);
      const activityGlow = ctx.createRadialGradient(
        centerX, centerY, radius * 0.8,
        centerX, centerY, radius * 2
      );
      activityGlow.addColorStop(0, `rgba(139, 92, 246, ${glowIntensity * 0.5})`);
      activityGlow.addColorStop(1, 'transparent');
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius * 2, 0, Math.PI * 2);
      ctx.fillStyle = activityGlow;
      ctx.fill();
    }
    
    // Main glow
    const glowGradient = ctx.createRadialGradient(
      centerX, centerY, radius * 0.4,
      centerX, centerY, radius * 2.2
    );
    glowGradient.addColorStop(0, colors.glow);
    glowGradient.addColorStop(1, 'transparent');
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius * 2.2, 0, Math.PI * 2);
    ctx.fillStyle = glowGradient;
    ctx.fill();
    
    // Model indicators (orbiting dots for each available model)
    if (modelIndicators.length > 0 && !compact) {
      modelIndicators.forEach((mi) => {
        const orbitRadius = radius * 1.8;
        const x = centerX + Math.cos(mi.angle + rotation * 0.008) * orbitRadius;
        const y = centerY + Math.sin(mi.angle + rotation * 0.008) * orbitRadius;
        const modelColor = MODEL_COLORS[mi.modelId] || '#6b7280';
        
        // Orbit ring
        ctx.beginPath();
        ctx.arc(centerX, centerY, orbitRadius, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(100, 100, 100, 0.1)`;
        ctx.lineWidth = 1;
        ctx.stroke();
        
        // Model dot
        ctx.beginPath();
        ctx.arc(x, y, mi.active ? 6 : 3, 0, Math.PI * 2);
        ctx.fillStyle = mi.active ? modelColor : `rgba(100, 100, 100, 0.5)`;
        ctx.fill();
        
        if (mi.active) {
          ctx.beginPath();
          ctx.arc(x, y, 10, 0, Math.PI * 2);
          ctx.strokeStyle = modelColor;
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }
      });
    }
    
    // Particles
    particles.forEach(p => {
      const scale = compact ? 0.65 : 1;
      const x = centerX + Math.cos(p.angle + rotation * 0.005) * (p.distance * scale);
      const y = centerY + Math.sin(p.angle + rotation * 0.005) * (p.distance * scale);
      
      ctx.beginPath();
      ctx.arc(x, y, p.size * scale, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(255, 255, 255, ${p.opacity * 0.8})`;
      ctx.fill();
    });
    
    // Context usage ring
    if (state.contextUsage > 0 && !compact) {
      const contextAngle = (state.contextUsage / 100) * Math.PI * 2;
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius + 8, -Math.PI / 2, -Math.PI / 2 + contextAngle);
      ctx.strokeStyle = state.contextUsage > 80 ? '#f59e0b' : '#22c55e';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
    
    // Main orb
    const orbGradient = ctx.createRadialGradient(
      centerX - radius * 0.35, centerY - radius * 0.35, 0,
      centerX, centerY, radius
    );
    orbGradient.addColorStop(0, '#ffffff');
    orbGradient.addColorStop(0.15, colors.primary);
    orbGradient.addColorStop(0.85, colors.primary);
    orbGradient.addColorStop(1, colors.glow);
    
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
    ctx.fillStyle = orbGradient;
    ctx.fill();
    
    // Inner patterns
    const waveOffset = rotation * 0.03;
    for (let i = 0; i < 3; i++) {
      const waveRadius = radius * (0.25 + i * 0.15);
      ctx.beginPath();
      for (let angle = 0; angle < Math.PI * 2; angle += 0.08) {
        const wave = Math.sin(angle * 3 + waveOffset + i * 0.5) * 2.5;
        const x = centerX + Math.cos(angle) * (waveRadius + wave);
        const y = centerY + Math.sin(angle) * (waveRadius + wave);
        if (angle === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.strokeStyle = `rgba(255, 255, 255, ${0.15 - i * 0.04})`;
      ctx.lineWidth = 0.8;
      ctx.stroke();
    }
    
    // Activity ring for thinking/working
    if (state.status === 'thinking' || state.status === 'working' || state.status === 'collaborating') {
      const ringRadius = radius + 12 + Math.sin(pulsePhase * 2.5) * 4;
      ctx.beginPath();
      ctx.arc(centerX, centerY, ringRadius * (compact ? 0.65 : 1), 0, Math.PI * 2);
      ctx.strokeStyle = colors.glow;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 8]);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    
    // Data flow lines (when collaborating between models)
    if (state.status === 'collaborating' && modelIndicators.length > 1 && !compact) {
      const activeModel = modelIndicators.find(m => m.active);
      if (activeModel) {
        modelIndicators.forEach(mi => {
          if (!mi.active) {
            const orbitRadius = radius * 1.8;
            const x1 = centerX + Math.cos(activeModel.angle + rotation * 0.008) * orbitRadius;
            const y1 = centerY + Math.sin(activeModel.angle + rotation * 0.008) * orbitRadius;
            const x2 = centerX + Math.cos(mi.angle + rotation * 0.008) * orbitRadius;
            const y2 = centerY + Math.sin(mi.angle + rotation * 0.008) * orbitRadius;
            
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            const midX = (x1 + x2) / 2 + Math.sin(pulsePhase * 3) * 10;
            const midY = (y1 + y2) / 2 + Math.cos(pulsePhase * 3) * 10;
            ctx.quadraticCurveTo(midX, midY, x2, y2);
            ctx.strokeStyle = `rgba(236, 72, 153, ${0.3 + Math.sin(pulsePhase * 4) * 0.2})`;
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        });
      }
    }
    
  }, [state, particles, modelIndicators, rotation, pulsePhase, dataPulse, compact]);

  if (compact) {
    return (
      <div className="jarvis-avatar-compact">
        <canvas 
          ref={canvasRef} 
          width={80} 
          height={80}
          className="avatar-canvas"
        />
        <div className="compact-status-indicator">
          <span className={`status-dot ${state.status}`} />
          <span className="status-text">{state.activity}</span>
        </div>
        {state.callsPerMinute > 0 && (
          <div className="compact-calls">{state.callsPerMinute}/min</div>
        )}
      </div>
    );
  }

  return (
    <div className={`jarvis-avatar ${state.status}`}>
      <div className="avatar-container">
        <canvas 
          ref={canvasRef} 
          width={200} 
          height={200}
          className="avatar-canvas"
        />
        
        <div className="avatar-rings">
          <div className="ring ring-1" />
          <div className="ring ring-2" />
          <div className="ring ring-3" />
        </div>
      </div>
      
      <div className="avatar-info">
        <div className="status-label">
          <span className={`status-indicator ${state.status}`} />
          <span className="status-text">
            {state.status === 'collaborating' ? 'Model Collaboration' : 
             state.status === 'daydreaming' ? 'Daydreaming' : 
             state.status.charAt(0).toUpperCase() + state.status.slice(1)}
          </span>
        </div>
        
        <div className="activity-text">{state.activity}</div>
        
        <div className="metrics-row">
          {state.callsPerMinute > 0 && (
            <div className="metric">
              <span className="metric-value">{state.callsPerMinute}</span>
              <span className="metric-label">calls/min</span>
            </div>
          )}
          {state.activeModel && (
            <div className="metric model-metric">
              <span className="metric-value" style={{ color: MODEL_COLORS[state.activeModel] || '#8b5cf6' }}>
                {state.activeModel.split(':')[0]}
              </span>
              <span className="metric-label">active</span>
            </div>
          )}
          {state.contextUsage > 0 && (
            <div className="metric">
              <span className={`metric-value ${state.contextUsage > 80 ? 'warning' : ''}`}>
                {state.contextUsage}%
              </span>
              <span className="metric-label">context</span>
            </div>
          )}
        </div>
        
        {state.daydreaming && state.idleTimeSeconds > 60 && (
          <div className="idle-indicator">
            Idle for {Math.floor(state.idleTimeSeconds / 60)}m
          </div>
        )}
        
        {state.proactiveMode && !state.daydreaming && state.callsPerMinute === 0 && (
          <div className="proactive-indicator">
            <span className="pulse-dot" /> Proactive mode
          </div>
        )}
        
        {state.recentThoughts.length > 0 && (
          <div className="recent-thoughts">
            {state.recentThoughts.slice(0, 2).map((thought, i) => (
              <div key={i} className="thought-item">
                <span className="thought-source">{thought.source}</span>
                <span className="thought-preview">{thought.content.slice(0, 50)}...</span>
              </div>
            ))}
          </div>
        )}
        
        {state.modelCoordinator && state.modelCoordinator.crossModelCalls > 0 && (
          <div className="collaboration-indicator">
            <span className="collab-icon">🔗</span>
            <span className="collab-text">
              {state.modelCoordinator.crossModelCalls} cross-model calls
            </span>
          </div>
        )}
        
        {state.subsystemActive && (
          <div className="subsystem-indicator">
            <span className="subsystem-icon">⚙️</span>
            <span className="subsystem-name">{state.subsystemActive} active</span>
          </div>
        )}

        {state.presenceState && (
          <div className="presence-indicator">
            <span className={`presence-dot ${state.presenceState}`} />
            <span className="presence-text">
              {state.presenceState === 'active' ? 'With you' :
               state.presenceState === 'focused' ? 'Attentive' :
               state.presenceState === 'typing' ? 'Listening...' :
               state.presenceState === 'idle' ? 'Waiting' :
               state.presenceState === 'away' ? 'On standby' :
               state.presenceState === 'returned' ? 'Welcome back!' :
               state.presenceState.charAt(0).toUpperCase() + state.presenceState.slice(1)}
            </span>
          </div>
        )}

        {state.anticipation && (
          <div className="anticipation-indicator">
            <span className="anticipation-icon">🎯</span>
            <span className="anticipation-text">
              {state.anticipation.description} ({Math.round(state.anticipation.confidence * 100)}%)
            </span>
          </div>
        )}

        {state.notifiableObservations > 0 && (
          <div className="observation-alert">
            <span className="alert-dot" />
            <span className="alert-text">{state.notifiableObservations} observation{state.notifiableObservations > 1 ? 's' : ''} to review</span>
          </div>
        )}
      </div>
      
      <div className="activity-trail">
        {particles.slice(0, 5).map((p) => (
          <div 
            key={p.id}
            className="trail-particle"
            style={{
              animationDelay: `${p.id * 0.1}s`,
            }}
          />
        ))}
      </div>
    </div>
  );
}

export function JarvisMinimalAvatar() {
  const [status, setStatus] = useState<'idle' | 'active' | 'daydreaming' | 'collaborating'>('idle');
  const [pulsePhase, setPulsePhase] = useState(0);
  const [callsPerMinute, setCallsPerMinute] = useState(0);
  
  useEffect(() => {
    const interval = setInterval(() => {
      setPulsePhase(p => (p + 0.1) % (Math.PI * 2));
    }, 50);
    return () => clearInterval(interval);
  }, []);
  
  useEffect(() => {
    const checkStatus = async () => {
      try {
        const [statusResp, jarvisResp, coordinatorResp] = await Promise.all([
          fetch('/api/status'),
          fetch('/api/jarvis/state').catch(() => null),
          fetch('/api/models/coordinator').catch(() => null),
        ]);
        const data = await statusResp.json();
        setCallsPerMinute(data.calls_per_minute || 0);
        
        if (coordinatorResp) {
          const coordData = await coordinatorResp.json();
          if (coordData.success && (coordData.cross_model_calls > 0 || coordData.total_handoffs > 0)) {
            setStatus('collaborating');
            return;
          }
        }
        
        if (jarvisResp) {
          const jd = await jarvisResp.json();
          if (jd.success && jd.state?.daydreaming) {
            setStatus('daydreaming');
            return;
          }
        }
        
        setStatus(data.turns > 0 || data.calls_per_minute > 0 ? 'active' : 'idle');
      } catch {
        setStatus('idle');
      }
    };
    checkStatus();
    const interval = setInterval(checkStatus, 2000);
    return () => clearInterval(interval);
  }, []);
  
  const colors = status === 'collaborating'
    ? { primary: '#ec4899', glow: 'rgba(236, 72, 153, 0.6)' }
    : status === 'daydreaming' 
      ? { primary: '#06b6d4', glow: 'rgba(6, 182, 212, 0.6)' }
      : status === 'active' 
        ? { primary: '#8b5cf6', glow: 'rgba(139, 92, 246, 0.6)' }
        : { primary: '#6b7280', glow: 'rgba(107, 114, 128, 0.4)' };
  
  return (
    <div className="jarvis-minimal">
      <div 
        className="minimal-orb"
        style={{
          background: `radial-gradient(circle at 30% 30%, white, ${colors.primary})`,
          boxShadow: `0 0 ${15 + callsPerMinute * 2}px ${colors.glow}, inset 0 0 10px rgba(255,255,255,0.3)`,
          transform: `scale(${1 + Math.sin(pulsePhase) * 0.05})`,
        }}
      >
        <div className="orb-shine" />
      </div>
      <div 
        className="minimal-ring"
        style={{ borderColor: colors.glow }}
      />
      {callsPerMinute > 0 && (
        <div className="minimal-calls">{callsPerMinute}/m</div>
      )}
    </div>
  );
}