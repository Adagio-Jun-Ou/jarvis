import { useState, useEffect, useMemo, useRef } from 'react';
import { useDashboard } from '../contexts/DashboardContext';

interface ActivityEvent {
  id: string;
  timestamp: Date;
  subsystem: string;
  action: string;
  details: string;
  fullDetails?: string;
  type: 'thinking' | 'tool' | 'learning' | 'evolution' | 'insight' | 'error';
  model?: 'primary' | 'backend' | 'parallel';
  modelName?: string;
  latency?: number;
  promptPreview?: string;
  resultPreview?: string;
  status?: 'running' | 'completed' | 'error';
  tokens?: number;
  metadata?: Record<string, any>;
}

const SUBSYSTEM_ICONS: Record<string, string> = {
  'Erdős Engine': '🧮',
  'Learning Pipeline': '📚',
  'Evolution': '🧬',
  'Daydream': '💭',
  'Self-Refine': '🔧',
  'Defense': '🛡️',
  'Intuition': '💡',
  'Knowledge Graph': '🕸️',
  'Vector Memory': '🔎',
  'Agent': '🤖',
  'LLM': '🧠',
  'Tool': '🔨',
  'Free Time': '🧪',
  'Dual Model': '🤝',
  'Kimi K2.5': '🎭',
  'GLM-5': '🔮',
  'Insight Bridge': '🌉',
  'Proactive': '🎯',
  'API': '📡',
  'Memory': '💾',
};

const MODEL_COLORS: Record<string, string> = {
  'primary': '#ec4899',
  'backend': '#8b5cf6',
  'parallel': '#22c55e',
};

const TYPE_COLORS: Record<string, string> = {
  thinking: '#8b5cf6',
  tool: '#f59e0b',
  learning: '#22c55e',
  evolution: '#ec4899',
  insight: '#06b6d4',
  error: '#ef4444',
};

const SUBSYSTEM_COLORS: Record<string, string> = {
  'Erdős Engine': 'linear-gradient(135deg, #8b5cf6, #7c3aed)',
  'Learning Pipeline': 'linear-gradient(135deg, #22c55e, #16a34a)',
  'Evolution': 'linear-gradient(135deg, #ec4899, #db2777)',
  'Daydream': 'linear-gradient(135deg, #06b6d4, #0891b2)',
  'Self-Refine': 'linear-gradient(135deg, #f59e0b, #d97706)',
  'Defense': 'linear-gradient(135deg, #ef4444, #dc2626)',
  'Intuition': 'linear-gradient(135deg, #a855f7, #9333ea)',
  'Memory': 'linear-gradient(135deg, #64748b, #475569)',
  'API': 'linear-gradient(135deg, #3b82f6, #2563eb)',
  'Free Time': 'linear-gradient(135deg, #14b8a6, #0d9488)',
};

export function ActivityFeed() {
  const [activities, setActivities] = useState<ActivityEvent[]>([]);
  const [isPaused, setIsPaused] = useState(false);
  const [filter, setFilter] = useState<string>('all');
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const lastPollRef = useRef<number>(0);
  const dashboardData = useDashboard();

  useEffect(() => {
    if (isPaused || !dashboardData.data) return;

    const now = Date.now();
    if (now - lastPollRef.current < 3000) return;
    lastPollRef.current = now;

    const newActivities: ActivityEvent[] = [];
    const data = dashboardData.data;

    // Erdős Engine activity
    if (data.erdos?.running) {
      const erdos = data.erdos;
      const strategies = erdos.best_strategies?.slice(0, 5).map((s: any) => 
        `${s.strategy} (${s.successful}/${s.attempts} = ${((s.successful / Math.max(1, s.attempts)) * 100).toFixed(0)}% success)`
      ).join('\n• ') || 'No strategies yet';
      
      const graphTheory = erdos.by_category?.graph_theory || 0;
      const combinatorics = erdos.by_category?.combinatorics || 0;
      const numberTheory = erdos.by_category?.number_theory || 0;
      const totalCategorized = graphTheory + combinatorics + numberTheory;
      
      newActivities.push({
        id: `erdos_${now}`,
        timestamp: new Date(),
        subsystem: 'Erdős Engine',
        action: `Solving ${erdos.current_problems || 0} math problems`,
        details: `${erdos.total_problems} total problems • ${erdos.insights_count} insights generated • ${(graphTheory/totalCategorized*100||0).toFixed(0)}% Graph Theory`,
        fullDetails: `📊 Problem Statistics\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nTotal Problems: ${erdos.total_problems}\nCurrently Solving: ${erdos.current_problems || 0}\nInsights Generated: ${erdos.insights_count}\n\n📁 By Category\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n• Graph Theory: ${graphTheory} (${(graphTheory/totalCategorized*100||0).toFixed(1)}%)\n• Combinatorics: ${combinatorics} (${(combinatorics/totalCategorized*100||0).toFixed(1)}%)\n• Number Theory: ${numberTheory} (${(numberTheory/totalCategorized*100||0).toFixed(1)}%)\n\n🎯 Top Strategies\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n• ${strategies}\n\n💡 Recent Insights: ${erdos.recent_insights?.length || 0} pending`,
        type: 'thinking',
        metadata: {
          totalProblems: erdos.total_problems,
          insights: erdos.insights_count,
          strategies: erdos.best_strategies,
          categories: { graphTheory, combinatorics, numberTheory },
          successRate: erdos.best_strategies?.[0]?.successful / Math.max(1, erdos.best_strategies?.[0]?.attempts) || 0,
        }
      });
    }

    // Daydream activity
    if (data.daydream?.running) {
      const dd = data.daydream;
      const topics = dd.exploration_topics?.slice(0, 5) || [];
      const efficiency = dd.tools_practiced > 0 ? (dd.insights_generated / dd.tools_practiced * 100).toFixed(1) : '0.0';
      
      newActivities.push({
        id: `daydream_${now}`,
        timestamp: new Date(),
        subsystem: 'Daydream',
        action: dd.can_daydream ? 'Exploring ideas autonomously' : 'Waiting for idle time',
        details: `${dd.tools_practiced} tools practiced • ${dd.insights_generated} insights (${efficiency}% efficiency) • ${dd.daydream_count} sessions`,
        fullDetails: `🌙 Daydream Status\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nState: ${dd.can_daydream ? 'Active - Exploring' : 'Idle - Waiting'}\nSession Count: ${dd.daydream_count}\nDaydreams Today: ${dd.daydreams_today || 0}\n\n🔧 Tool Practice\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nTools Practiced: ${dd.tools_practiced}\nInsights Generated: ${dd.insights_generated}\nEfficiency: ${efficiency}% insights per tool\n\n📚 Exploration Topics\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n${topics.map((t: string) => `• ${t}`).join('\n') || '• No active topics'}\n\n⏱️ Last Run: ${dd.last_run ? new Date(dd.last_run * 1000).toLocaleTimeString() : 'N/A'}`,
        type: 'learning',
        metadata: {
          toolsPracticed: dd.tools_practiced,
          insights: dd.insights_generated,
          topics: topics,
          canDaydream: dd.can_daydream,
          efficiency: parseFloat(efficiency),
        }
      });
    }

    // Evolution activity
    if (data.evolution?.running) {
      const evo = data.evolution;
      const pathway = evo.pathway || {};
      const genes = pathway.gene_pool || {};
      const caps = pathway.capability_scores || {};
      const sortedCaps = Object.entries(caps).sort((a: any, b: any) => b[1] - a[1]);
      const topCaps = sortedCaps.slice(0, 5);
      
      // Calculate gene statistics
      const geneValues = Object.values(genes).map((g: any) => g.value || 0).filter((v: number) => v > 0);
      const avgGene = geneValues.length > 0 ? (geneValues.reduce((a: number, b: number) => a + b, 0) / geneValues.length) : 0;
      
      newActivities.push({
        id: `evolution_${now}`,
        timestamp: new Date(),
        subsystem: 'Evolution',
        action: `Stage: ${pathway.current_stage || 'initialized'} • Generation ${pathway.generation || 0}`,
        details: `${pathway.total_evolutions || 0} total evolutions • ${(pathway.success_rate * 100 || 0).toFixed(0)}% success rate • ${Object.keys(genes).length} active genes`,
        fullDetails: `🧬 Evolution Status\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nCurrent Stage: ${pathway.current_stage || 'initialized'}\nGeneration: ${pathway.generation || 0}\nTotal Evolutions: ${pathway.total_evolutions || 0}\nSuccess Rate: ${(pathway.success_rate * 100 || 0).toFixed(1)}%\n\n📊 Gene Pool (${Object.keys(genes).length} genes, avg: ${avgGene.toFixed(2)})\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n${Object.entries(genes).slice(0, 8).map(([k, v]: any) => `• ${k}: ${v.value?.toFixed(3) || 'N/A'}`).join('\n')}\n\n🏆 Top Capabilities\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n${topCaps.map(([k, v]: any) => `• ${k}: ${(v * 100).toFixed(1)}%`).join('\n')}\n\n📈 Improvement\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nBest Improvement: ${pathway.best_improvement?.toFixed(2) || 'N/A'}\nAvg Improvement: ${pathway.avg_improvement?.toFixed(2) || 'N/A'}`,
        type: 'evolution',
        metadata: {
          stage: pathway.current_stage,
          generation: pathway.generation,
          evolutions: pathway.total_evolutions,
          successRate: pathway.success_rate,
          genes: genes,
          capabilities: caps,
          avgGeneValue: avgGene,
        }
      });
    }

    // Learning Pipeline activity
    if (data.learning?.running) {
      const learn = data.learning;
      const metrics = learn.metrics || {};
      const solveRate = (metrics.solve_rate * 100 || 0).toFixed(1);
      const avgTime = metrics.total_learning_time_hours || 0;
      
      // Calculate phase progress
      const phases = ['exploration', 'practice', 'validation', 'integration'];
      const phaseIndex = phases.indexOf(learn.current_phase || 'exploration');
      const phaseProgress = ((phaseIndex + 1) / phases.length * 100).toFixed(0);
      
      newActivities.push({
        id: `learning_${now}`,
        timestamp: new Date(),
        subsystem: 'Learning Pipeline',
        action: `Phase: ${learn.current_phase || 'unknown'} (${phaseProgress}% complete)`,
        details: `${metrics.problems_attempted || 0} problems • ${solveRate}% solve rate • ${avgTime.toFixed(1)}h learning`,
        fullDetails: `📚 Learning Pipeline Status\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nCurrent Phase: ${learn.current_phase}\nPhase Progress: ${phaseProgress}%\n\n🎯 Problem Solving\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nProblems Attempted: ${metrics.problems_attempted || 0}\nProblems Solved: ${metrics.problems_solved || 0}\nSolve Rate: ${solveRate}%\nAvg Time per Problem: ${metrics.avg_time_per_problem?.toFixed(1) || 'N/A'}s\n\n🔐 Security Analysis\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nThreats Analyzed: ${metrics.threats_analyzed || 0}\nVulnerabilities Found: ${metrics.vulnerabilities_found || 0}\nSecurity Score: ${metrics.security_score || 'N/A'}\n\n💡 Insights\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nDream Insights: ${metrics.dream_insights || 0}\nBest Math Strategy: ${metrics.best_math_strategy || 'none'}\n\n⏱️ Time Spent\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nTotal Learning Time: ${avgTime.toFixed(2)} hours\nEvolution Generation: ${metrics.evolution_generation || 0}`,
        type: 'learning',
        metadata: {
          phase: learn.current_phase,
          phaseProgress: parseFloat(phaseProgress),
          metrics: metrics,
          solveRate: parseFloat(solveRate),
        }
      });
    }

    // Defense activity
    if (data.defense?.status === 'active') {
      newActivities.push({
        id: `defense_${now}`,
        timestamp: new Date(),
        subsystem: 'Defense',
        action: 'Monitoring for threats',
        details: 'Security monitoring active',
        fullDetails: 'Defense system is actively monitoring for security threats and anomalies.',
        type: 'thinking',
      });
    }

    // Dual Model activity
    if (data.dual_model) {
      const dm = data.dual_model;
      const total = dm.primary_requests + dm.backend_requests;
      if (total > 0 || dm.total_requests_per_min > 0) {
        const primaryPercent = total > 0 ? ((dm.primary_requests / total) * 100).toFixed(0) : '0';
        const backendPercent = total > 0 ? ((dm.backend_requests / total) * 100).toFixed(0) : '0';
        const totalPerMin = (dm.primary_requests_per_min || 0) + (dm.backend_requests_per_min || 0);
        const parallelPercent = total > 0 ? ((dm.parallel_requests / total) * 100).toFixed(0) : '0';
        
        newActivities.push({
          id: `dualmodel_${now}`,
          timestamp: new Date(),
          subsystem: 'Dual Model',
          action: `Orchestrating ${totalPerMin}/min requests`,
          details: `Primary ${primaryPercent}% (${dm.primary_requests}) • Backend ${backendPercent}% (${dm.backend_requests}) • ${parallelPercent}% parallel`,
          fullDetails: `🤖 Dual Model Architecture\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nTotal Requests: ${total}\nRequests/min: ${totalPerMin}\n\n🌸 Primary Model (Kimi K2.5)\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nRequests: ${dm.primary_requests}\nRate: ${dm.primary_requests_per_min || 0}/min\nAvg Latency: ${dm.avg_primary_latency_ms?.toFixed(0) || 0}ms\nShare: ${primaryPercent}%\nCapabilities: chat, creative, abstraction\n\n🧠 Backend Model (GLM-5)\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nRequests: ${dm.backend_requests}\nRate: ${dm.backend_requests_per_min || 0}/min\nAvg Latency: ${dm.avg_backend_latency_ms?.toFixed(0) || 0}ms\nShare: ${backendPercent}%\nCapabilities: code, math, reasoning\n\n🤝 Collaboration\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nParallel Requests: ${dm.parallel_requests} (${parallelPercent}%)\nCollaborative: ${dm.collaborative_requests || 0}\nSequential: ${dm.sequential_requests || 0}\n\n❌ Errors: ${dm.errors || 0}`,
          type: 'thinking',
          model: total > dm.primary_requests ? 'parallel' : dm.primary_requests > 0 ? 'primary' : 'backend',
          metadata: dm,
        });
      }
    }

    // Knowledge Graph activity
    if (data.knowledge_graph) {
      const kg = data.knowledge_graph;
      const density = kg.nodes > 0 ? ((kg.edges / (kg.nodes * (kg.nodes - 1) / 2)) * 100).toFixed(1) : '0.0';
      const avgCommunitySize = kg.communities > 0 ? (kg.nodes / kg.communities).toFixed(1) : '0.0';
      
      newActivities.push({
        id: `kgraph_${now}`,
        timestamp: new Date(),
        subsystem: 'Knowledge Graph',
        action: `Maintaining ${kg.nodes} knowledge nodes`,
        details: `${kg.nodes} nodes • ${kg.edges} edges • ${kg.communities} communities • ${density}% density`,
        fullDetails: `🕸️ Knowledge Graph\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nNodes: ${kg.nodes}\nEdges: ${kg.edges}\nCommunities: ${kg.communities}\n\n📊 Graph Metrics\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nEdge Density: ${density}%\nAvg Connections: ${kg.nodes > 0 ? (kg.edges * 2 / kg.nodes).toFixed(1) : '0.0'}\nAvg Community Size: ${avgCommunitySize}\n\n🔗 PageRank\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nComputed for importance ranking\nTop nodes have highest influence\n\n📐 Community Detection\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nUsed for clustering related concepts\nHelps semantic retrieval`,
        type: 'insight',
      });
    }

    // Insights activity
    if (data.insights && Array.isArray(data.insights) && data.insights.length > 0) {
      const recentInsight = data.insights[0];
      if (recentInsight && recentInsight.content) {
        const insightCount = data.insights.length;
        const categories = [...new Set(data.insights.map((i: any) => i.category).filter(Boolean))];
        
        newActivities.push({
          id: `insight_${now}`,
          timestamp: new Date(),
          subsystem: 'Insight Bridge',
          action: `${recentInsight.type || 'New'} insight processed`,
          details: `"${recentInsight.content.slice(0, 60)}..." • ${insightCount} total • ${categories.length} categories`,
          fullDetails: `💡 Insight Details\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nType: ${recentInsight.type || 'general'}\nCategory: ${recentInsight.category || 'uncategorized'}\nSource: ${recentInsight.source || 'unknown'}\n\n📝 Content\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n${recentInsight.content}\n\n📊 Insight Statistics\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nTotal Insights: ${insightCount}\nCategories: ${categories.join(', ') || 'none'}\nRecent: ${data.insights.slice(0, 3).map((i: any) => i.type || 'general').join(', ')}\n\n🔗 Connections\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nStored in knowledge graph\nLinked to relevant problems\nAvailable for retrieval`,
          type: 'insight',
        });
      }
    }

    // Model Coordinator activity
    if (data.models_coordinator) {
      const mc = data.models_coordinator;
      if (mc.active_model || mc.total_handoffs > 0) {
        const modelsList = mc.connected_models?.join(', ') || 'none';
        const handoffRate = mc.total_handoffs > 0 ? (mc.cross_model_calls / mc.total_handoffs * 100).toFixed(0) : '0';
        
        newActivities.push({
          id: `models_${now}`,
          timestamp: new Date(),
          subsystem: 'Model Coordinator',
          action: mc.active_model ? `Routing to ${mc.active_model}` : 'Coordinating model routing',
          details: `${mc.connected_models?.length || 0} models connected • ${mc.total_handoffs || 0} handoffs • ${handoffRate}% cross-model`,
          fullDetails: `🔄 Model Coordinator\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nActive Model: ${mc.active_model || 'none'}\n\n🤖 Connected Models\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n${modelsList}\n\n📊 Statistics\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nTotal Handoffs: ${mc.total_handoffs || 0}\nCross-Model Calls: ${mc.cross_model_calls || 0}\nHandoff Rate: ${handoffRate}%\n\n⚡ Performance\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nSelection based on task type\nOptimized for capabilities\nSeamless model switching`,
          type: 'thinking',
        });
      }
    }

    // API status
    if (data.status) {
      const status = data.status;
      if (status.calls_per_minute > 0 || status.total_calls > 0) {
        const cachePercent = status.cache_rate || 0;
        const latencySec = ((status.avg_latency || 0) / 1000).toFixed(2);
        
        newActivities.push({
          id: `api_${now}`,
          timestamp: new Date(),
          subsystem: 'API',
          action: `Processing ${status.calls_per_minute || 0} requests/min`,
          details: `${status.total_calls || 0} total • ${latencySec}s avg latency • ${cachePercent}% cached`,
          fullDetails: `📡 API Statistics\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nTotal Calls: ${status.total_calls || 0}\nCalls/Minute: ${status.calls_per_minute || 0}\nModel: ${status.model || 'unknown'}\n\n⏱️ Performance\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nAvg Latency: ${latencySec}s\nCache Rate: ${cachePercent}%\n\n💾 Caching\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nCached responses improve speed\nReduced LLM calls\nOptimized token usage`,
          type: 'tool',
          latency: status.avg_latency,
        });
      }
    }

    if (newActivities.length > 0) {
      setActivities(prev => {
        const combined = [...newActivities.reverse(), ...prev];
        // Dedupe by subsystem
        const seen = new Set();
        return combined.filter(a => {
          if (seen.has(a.subsystem)) return false;
          seen.add(a.subsystem);
          return true;
        }).slice(0, 50);
      });
    }
  }, [dashboardData.data, isPaused]);

  const filteredActivities = filter === 'all' 
    ? activities 
    : activities.filter(a => a.type === filter);

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', { 
      hour12: false, 
      hour: '2-digit', 
      minute: '2-digit',
      second: '2-digit',
    });
  };

  const toggleExpand = (id: string) => {
    setExpandedId(expandedId === id ? null : id);
  };

  return (
    <div className="activity-feed">
      <div className="feed-header">
        <h3>Live Activity</h3>
        <div className="feed-controls">
          <select 
            value={filter} 
            onChange={e => setFilter(e.target.value)}
            className="filter-select"
          >
            <option value="all">All Activity</option>
            <option value="thinking">Thinking</option>
            <option value="tool">Tools</option>
            <option value="learning">Learning</option>
            <option value="evolution">Evolution</option>
            <option value="insight">Insights</option>
          </select>
          <button 
            className={`pause-btn ${isPaused ? 'paused' : ''}`}
            onClick={() => setIsPaused(!isPaused)}
          >
            {isPaused ? '▶️' : '⏸️'}
          </button>
        </div>
      </div>
      
      <div className="feed-content">
        {filteredActivities.length === 0 ? (
          <div className="feed-empty">
            <span className="empty-icon">🔄</span>
            <span>Waiting for activity...</span>
          </div>
        ) : (
          filteredActivities.map((activity, i) => (
            <div 
              key={activity.id} 
              className={`activity-item ${expandedId === activity.id ? 'expanded' : ''}`}
              style={{ 
                animationDelay: `${i * 0.05}s`,
                borderLeftColor: TYPE_COLORS[activity.type],
              }}
              onClick={() => toggleExpand(activity.id)}
            >
              <div className="activity-icon">
                {SUBSYSTEM_ICONS[activity.subsystem] || '⚙️'}
              </div>
              <div className="activity-content">
                <div className="activity-header">
                  <span 
                    className="activity-subsystem" 
                    style={{ background: SUBSYSTEM_COLORS[activity.subsystem] || 'linear-gradient(135deg, #64748b, #475569)' }}
                  >
                    {activity.subsystem}
                  </span>
                  {activity.model && (
                    <span 
                      className="activity-model-badge"
                      style={{ 
                        background: activity.model === 'parallel' ? 'linear-gradient(135deg, #ec4899, #8b5cf6)' : MODEL_COLORS[activity.model],
                        marginLeft: '6px'
                      }}
                    >
                      {activity.modelName || (activity.model === 'primary' ? 'Kimi' : activity.model === 'backend' ? 'GLM' : 'Both')}
                    </span>
                  )}
                  <span className="activity-time">{formatTime(activity.timestamp)}</span>
                </div>
                <div className="activity-action">{activity.action}</div>
                <div className="activity-details">{activity.details}</div>
                {expandedId === activity.id && activity.fullDetails && (
                  <div className="activity-full-details">
                    <pre>{activity.fullDetails}</pre>
                  </div>
                )}
                {activity.promptPreview && (
                  <div className="activity-prompt" title={activity.promptPreview}>
                    📝 {activity.promptPreview}
                  </div>
                )}
                {activity.resultPreview && (
                  <div className="activity-result" title={activity.resultPreview}>
                    ✨ {activity.resultPreview}
                  </div>
                )}
                {activity.latency && activity.latency > 1000 && (
                  <div className="activity-latency">
                    ⏱️ {(activity.latency / 1000).toFixed(1)}s
                  </div>
                )}
              </div>
              <div 
                className="activity-type-badge"
                style={{ backgroundColor: TYPE_COLORS[activity.type] }}
              >
                {activity.type}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export function ThinkingStream({ messages = [], isThinking = false }: { messages?: Array<{ id: string; role: string; content: string; thinking?: string; timestamp: number }>; isThinking?: boolean }) {
  const [displayedThought, setDisplayedThought] = useState('');
  const [charIndex, setCharIndex] = useState(0);
  const [recentThoughts, setRecentThoughts] = useState<string[]>([]);
  const [subsystemThoughts, setSubsystemThoughts] = useState<Array<{source: string; content: string}>>([]);

  const realThinkingContent = useMemo(() => {
    const thoughts: string[] = [];
    for (const msg of messages) {
      if (msg.thinking && msg.thinking.trim()) {
        thoughts.push(msg.thinking.trim());
      }
    }
    return thoughts;
  }, [messages]);

  const lastThinking = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      const msg = messages[i];
      if (msg.thinking && msg.thinking.trim()) {
        return msg.thinking.trim();
      }
    }
    return null;
  }, [messages]);

  useEffect(() => {
    const fetchSubsystemThoughts = async () => {
      try {
        const resp = await fetch('/api/recent-thoughts');
        const data = await resp.json();
        if (data.success && data.thoughts) {
          setSubsystemThoughts(data.thoughts);
        }
      } catch (e) {
        // Ignore
      }
    };
    fetchSubsystemThoughts();
    const interval = setInterval(fetchSubsystemThoughts, 3000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (!isThinking || !lastThinking) return;

    const thinkingToShow = lastThinking.slice(0, 500);
    
    if (displayedThought !== thinkingToShow && charIndex < thinkingToShow.length) {
      const timer = setTimeout(() => {
        setDisplayedThought(thinkingToShow.slice(0, charIndex + 1));
        setCharIndex(prev => prev + 1);
      }, 15);
      return () => clearTimeout(timer);
    }
  }, [isThinking, lastThinking, displayedThought, charIndex]);

  useEffect(() => {
    if (!isThinking && displayedThought) {
      const trimmed = displayedThought.trim();
      if (trimmed && !recentThoughts.includes(trimmed)) {
        setRecentThoughts(prev => [trimmed, ...prev].slice(0, 5));
      }
      setDisplayedThought('');
      setCharIndex(0);
    }
  }, [isThinking, displayedThought, recentThoughts]);

  useEffect(() => {
    if (!isThinking) {
      setDisplayedThought('');
      setCharIndex(0);
    }
  }, [isThinking]);

  const fallbackThoughts = [
    'Analyzing query intent...',
    'Searching knowledge base...',
    'Processing semantics...',
    'Building response structure...',
    'Applying learned patterns...',
  ];

  const staticThoughts = useMemo(() => {
    const all: string[] = [];
    for (const s of subsystemThoughts) {
      all.push(`[${s.source}] ${s.content}`);
    }
    if (realThinkingContent.length > 0) {
      all.push(...realThinkingContent.slice(-3).reverse());
    }
    if (all.length === 0) {
      return fallbackThoughts;
    }
    return all;
  }, [realThinkingContent, subsystemThoughts]);

  return (
    <div className="thinking-stream">
      <div className="thinking-header">
        <span className="thinking-icon">💭</span>
        <span className="thinking-label">
          {isThinking ? 'Jarvis is thinking...' : 'Recent thoughts'}
        </span>
      </div>
      <div className="current-thought">
        {isThinking && displayedThought ? (
          <>
            {displayedThought}
            <span className="cursor">|</span>
          </>
        ) : isThinking ? (
          <>
            <span className="thinking-waiting">Processing...</span>
            <span className="cursor">|</span>
          </>
        ) : (
          <span className="thinking-idle">Ready</span>
        )}
      </div>
      <div className="thought-history">
        {(isThinking ? recentThoughts : staticThoughts).map((thought, i) => (
          <div key={i} className="past-thought">{thought}</div>
        ))}
      </div>
    </div>
  );
}