import { useState, useEffect, useRef, useCallback } from 'react';
import './FlowVisualization.css';

interface FlowNode {
  id: string;
  label: string;
  sublabel?: string;
  icon: string;
  x: number;
  y: number;
  type: 'input' | 'processor' | 'hub' | 'storage' | 'model' | 'subsystem' | 'tool' | 'bridge';
}

interface FlowEdge {
  from: string;
  to: string;
  label?: string;
  type: 'input' | 'request' | 'bidirectional' | 'embedding' | 'insight' | 'tool' | 'model' | 'internal';
}

interface Particle {
  id: number;
  fromId: string;
  toId: string;
  color: string;
  startTime: number;
  duration: number;
}

const NODES: FlowNode[] = [
  // Layer 1: Input/Agent/Context/PFC (y=30)
  { id: 'user', label: 'User', sublabel: 'Query', x: 80, y: 30, type: 'input', icon: '👤' },
  { id: 'agent', label: 'Jarvis Agent', sublabel: 'Orchestrator', x: 240, y: 30, type: 'processor', icon: '🤖' },
  { id: 'context', label: 'Context Engine', sublabel: 'Unified State', x: 400, y: 30, type: 'processor', icon: '📝' },
  { id: 'pfc', label: 'Prefrontal Cortex', sublabel: 'LCM (~100M params)', x: 560, y: 30, type: 'hub', icon: '🧠' },
  
  // Layer 2: Memory/Knowledge/Embedding (y=100)
  { id: 'memory', label: 'Vector Memory', sublabel: 'HNSW Index', x: 160, y: 100, type: 'storage', icon: '💾' },
  { id: 'knowledge', label: 'Knowledge Graph', sublabel: 'PageRank + Communities', x: 320, y: 100, type: 'storage', icon: '🕸️' },
  { id: 'embedding', label: 'Embedding Hub', sublabel: 'Jina v5 (768d)', x: 480, y: 100, type: 'hub', icon: '🔗' },
  
  // Layer 3: Models (y=170)
  { id: 'primary', label: 'Primary', sublabel: 'Kimi K2.5', x: 80, y: 170, type: 'model', icon: '🎭' },
  { id: 'backend', label: 'Backend', sublabel: 'GLM-5', x: 160, y: 170, type: 'model', icon: '🧙' },
  { id: 'audio_io', label: 'Audio I/O', sublabel: 'Voice Tokens', x: 240, y: 170, type: 'model', icon: '🔊' },
  
  // Layer 4: Tools Hub (y=240)
  { id: 'tools', label: 'Tools Registry', sublabel: '95 Tools', x: 320, y: 240, type: 'tool', icon: '🔧' },
  { id: 'tool_learning', label: 'Tool Learning', sublabel: 'Pattern Discovery', x: 400, y: 240, type: 'processor', icon: '📚' },
  
  // Tool Categories - Group 1 & 2: Right Side Column 1 (x=750)
  // Direct tools and knowledge tools above subsystems
  { id: 'cat_web', label: 'Web', sublabel: '4', x: 750, y: 30, type: 'tool', icon: '🌐' },
  { id: 'cat_browser', label: 'Browser', sublabel: '7', x: 750, y: 60, type: 'tool', icon: '🌐' },
  { id: 'cat_screen', label: 'Screen', sublabel: '13', x: 750, y: 90, type: 'tool', icon: '🖥️' },
  { id: 'cat_terminal', label: 'Terminal', sublabel: '4', x: 750, y: 120, type: 'tool', icon: '💻' },
  { id: 'cat_system', label: 'System', sublabel: '4', x: 750, y: 150, type: 'tool', icon: '⚙️' },
  { id: 'cat_files', label: 'Files', sublabel: '4', x: 750, y: 180, type: 'tool', icon: '📁' },
  { id: 'cat_data', label: 'Data', sublabel: '7', x: 750, y: 210, type: 'tool', icon: '📊' },
  
  { id: 'cat_multimodal', label: 'Multimodal', sublabel: '9', x: 750, y: 250, type: 'tool', icon: '🎬' },
  { id: 'cat_memory', label: 'Memory', sublabel: '3', x: 750, y: 280, type: 'tool', icon: '🧠' },
  { id: 'cat_security', label: 'Security', sublabel: '4', x: 750, y: 310, type: 'tool', icon: '🔒' },
  
  // Tool Categories - Group 3: Right Side Column 2 (x=850) - System tools
  // Positioned higher alongside main flow
  { id: 'cat_home', label: 'Home', sublabel: '1', x: 850, y: 30, type: 'tool', icon: '🏠' },
  { id: 'cat_automation', label: 'Automation', sublabel: '2', x: 850, y: 60, type: 'tool', icon: '🤖' },
  { id: 'cat_spatial', label: 'Spatial', sublabel: '4', x: 850, y: 90, type: 'tool', icon: '📐' },
  { id: 'cat_meta', label: 'Meta', sublabel: '17', x: 850, y: 120, type: 'tool', icon: '🔧' },
  { id: 'cat_music', label: 'Music', sublabel: '6', x: 850, y: 150, type: 'tool', icon: '🎵' },
  { id: 'cat_learning', label: 'Learning', sublabel: '6', x: 850, y: 180, type: 'tool', icon: '📚' },
  { id: 'cat_evolution', label: 'Evolution', sublabel: '6', x: 850, y: 210, type: 'tool', icon: '🧬' },
  { id: 'cat_refinement', label: 'Refine', sublabel: '4', x: 850, y: 240, type: 'tool', icon: '✨' },
  

  
  // Layer 5: Bridge Layer (y=240)
  { id: 'insight_bridge', label: 'Insight Bridge', sublabel: 'Math→Evolution', x: 320, y: 240, type: 'bridge', icon: '🌉' },
  { id: 'permission_queue', label: 'Permission Queue', sublabel: 'Human Approval', x: 480, y: 240, type: 'processor', icon: '🔐' },
  
  // Layer 6: Background Subsystems (y=400)
  { id: 'erdos', label: 'Erdős Engine', sublabel: 'Backend: Math', x: 40, y: 400, type: 'subsystem', icon: '🔢' },
  { id: 'daydream', label: 'Daydream', sublabel: 'Dual-model: Explore', x: 140, y: 400, type: 'subsystem', icon: '💭' },
  { id: 'evolution', label: 'Evolution', sublabel: 'Dual-model: Propose', x: 240, y: 400, type: 'subsystem', icon: '🧬' },
  { id: 'learning', label: 'Learning', sublabel: 'Dual-model: Patterns', x: 340, y: 400, type: 'subsystem', icon: '📈' },
  { id: 'defense', label: 'Defense', sublabel: 'Threat Monitor', x: 440, y: 400, type: 'subsystem', icon: '🛡️' },
  { id: 'proactive', label: 'Proactive', sublabel: 'Curiosity', x: 540, y: 400, type: 'subsystem', icon: '🎯' },
  { id: 'music_creation', label: 'Music Creation', sublabel: 'Always-on Composition', x: 640, y: 400, type: 'subsystem', icon: '🎵' },
  { id: 'intuition', label: 'Intuition', sublabel: 'Bridge', x: 740, y: 400, type: 'bridge', icon: '🔮' },
];

const EDGES: FlowEdge[] = [
  // Layer 1: Input flow
  { from: 'user', to: 'agent', label: 'query', type: 'input' },
  { from: 'agent', to: 'context', label: 'assemble', type: 'request' },
  { from: 'agent', to: 'pfc', label: 'observe', type: 'bidirectional' },
  { from: 'context', to: 'pfc', label: 'concepts', type: 'bidirectional' },
  
  // Layer 2: Memory/Knowledge flow
  { from: 'context', to: 'memory', type: 'bidirectional' },
  { from: 'context', to: 'knowledge', type: 'bidirectional' },
  { from: 'context', to: 'embedding', type: 'bidirectional' },
  { from: 'memory', to: 'embedding', type: 'bidirectional' },
  { from: 'knowledge', to: 'embedding', type: 'bidirectional' },
  
  // PFC connections
  { from: 'pfc', to: 'embedding', label: 'vectors', type: 'embedding' },
  { from: 'pfc', to: 'memory', label: 'patterns', type: 'insight' },
  { from: 'pfc', to: 'knowledge', label: 'concepts', type: 'insight' },
  
  // Layer 3: Models
  { from: 'agent', to: 'primary', label: 'chat', type: 'model' },
  { from: 'agent', to: 'backend', label: 'code/math', type: 'model' },
  { from: 'primary', to: 'embedding', type: 'bidirectional' },
  { from: 'backend', to: 'embedding', type: 'bidirectional' },
  { from: 'primary', to: 'memory', type: 'bidirectional' },
  { from: 'backend', to: 'memory', type: 'bidirectional' },
  { from: 'primary', to: 'pfc', label: 'response', type: 'model' },
  { from: 'backend', to: 'pfc', label: 'response', type: 'model' },
  { from: 'agent', to: 'audio_io', label: 'voice', type: 'model' },
  { from: 'audio_io', to: 'pfc', label: 'tokens', type: 'insight' },
  { from: 'audio_io', to: 'embedding', type: 'bidirectional' },
  
  // Layer 4: Tools
  { from: 'tools', to: 'tool_learning', type: 'internal' },
  { from: 'tool_learning', to: 'memory', type: 'bidirectional' },
  { from: 'tool_learning', to: 'embedding', type: 'bidirectional' },
  { from: 'tool_learning', to: 'pfc', label: 'patterns', type: 'insight' },
  
  // Tool Categories - grouped connections
  { from: 'tools', to: 'cat_web', type: 'tool' },
  { from: 'tools', to: 'cat_browser', type: 'tool' },
  { from: 'tools', to: 'cat_screen', type: 'tool' },
  { from: 'tools', to: 'cat_terminal', type: 'tool' },
  { from: 'tools', to: 'cat_system', type: 'tool' },
  { from: 'tools', to: 'cat_files', type: 'tool' },
  { from: 'tools', to: 'cat_data', type: 'tool' },
  { from: 'tools', to: 'cat_multimodal', type: 'tool' },
  { from: 'tools', to: 'cat_memory', type: 'tool' },
  { from: 'tools', to: 'cat_security', type: 'tool' },
  { from: 'tools', to: 'cat_home', type: 'tool' },
  { from: 'tools', to: 'cat_automation', type: 'tool' },
  { from: 'tools', to: 'cat_spatial', type: 'tool' },
  { from: 'tools', to: 'cat_meta', type: 'tool' },
  { from: 'tools', to: 'cat_music', type: 'tool' },
  { from: 'tools', to: 'cat_learning', type: 'tool' },
  { from: 'tools', to: 'cat_evolution', type: 'tool' },
  { from: 'tools', to: 'cat_refinement', type: 'tool' },
  
  // Layer 5: Bridge
  { from: 'erdos', to: 'insight_bridge', type: 'bidirectional' },
  { from: 'daydream', to: 'insight_bridge', type: 'bidirectional' },
  { from: 'insight_bridge', to: 'knowledge', type: 'bidirectional' },
  { from: 'insight_bridge', to: 'evolution', type: 'bidirectional' },
  { from: 'permission_queue', to: 'user', type: 'internal' },
  { from: 'permission_queue', to: 'pfc', label: 'decisions', type: 'internal' },
  
  // Layer 6: Subsystems → PFC
  { from: 'erdos', to: 'pfc', label: 'math', type: 'insight' },
  { from: 'daydream', to: 'pfc', label: 'explore', type: 'insight' },
  { from: 'evolution', to: 'pfc', label: 'proposals', type: 'insight' },
  { from: 'learning', to: 'pfc', label: 'patterns', type: 'insight' },
  { from: 'defense', to: 'pfc', label: 'security', type: 'insight' },
  { from: 'proactive', to: 'pfc', label: 'curiosity', type: 'insight' },
  { from: 'music_creation', to: 'pfc', label: 'acoustic', type: 'insight' },
  
  // Music Creation Engine connections
  { from: 'music_creation', to: 'embedding', type: 'bidirectional' },
  { from: 'music_creation', to: 'knowledge', type: 'bidirectional' },
  { from: 'music_creation', to: 'insight_bridge', type: 'bidirectional' },
  
  // Subsystems ↔ Models
  { from: 'erdos', to: 'backend', type: 'bidirectional' },
  { from: 'daydream', to: 'primary', type: 'bidirectional' },
  { from: 'daydream', to: 'backend', type: 'bidirectional' },
  { from: 'evolution', to: 'primary', type: 'bidirectional' },
  { from: 'evolution', to: 'backend', type: 'bidirectional' },
  { from: 'learning', to: 'primary', type: 'bidirectional' },
  { from: 'learning', to: 'backend', type: 'bidirectional' },
  
  // Subsystems ↔ Memory/Knowledge
  { from: 'learning', to: 'memory', type: 'bidirectional' },
  { from: 'defense', to: 'knowledge', type: 'bidirectional' },
  { from: 'proactive', to: 'knowledge', type: 'bidirectional' },
  { from: 'erdos', to: 'knowledge', type: 'bidirectional' },
  { from: 'daydream', to: 'embedding', type: 'bidirectional' },
  
  // Intuition Bridge
  { from: 'intuition', to: 'erdos', type: 'bidirectional' },
  { from: 'intuition', to: 'daydream', type: 'bidirectional' },
  { from: 'intuition', to: 'learning', type: 'bidirectional' },
  { from: 'intuition', to: 'pfc', label: 'intuition', type: 'insight' },
  
  // Evolution → Permission Queue
  { from: 'evolution', to: 'permission_queue', type: 'internal' },
];

const TYPE_COLORS: Record<string, string> = {
  input: '#ec4899',
  request: '#64748b',
  bidirectional: '#22c55e',
  embedding: '#a855f7',
  insight: '#f97316',
  tool: '#14b8a6',
  model: '#6366f1',
  internal: '#fbbf24',
};

export function FlowVisualization() {
  const [particles, setParticles] = useState<Particle[]>([]);
  const [nodeStates, setNodeStates] = useState<Record<string, { active: boolean }>>({});
  const [flowData, setFlowData] = useState<any>(null);
  const particleIdRef = useRef(0);
  const lastCountsRef = useRef({ primary: 0, backend: 0, embeddings: 0, audio_io: 0 });
  const lastSubsystemStateRef = useRef<Record<string, boolean>>({});
  const lastActiveToolsRef = useRef<string[]>([]);  // Track active tools
  
  const createParticle = useCallback((fromId: string, toId: string, color: string, duration = 1000) => {
    const id = particleIdRef.current++;
    setParticles(prev => [...prev.slice(-100), { id, fromId, toId, color, startTime: Date.now(), duration }]);
  }, []);
  
  const createBidirectionalParticles = useCallback((fromId: string, toId: string, color: string, duration = 1000) => {
    const id1 = particleIdRef.current++;
    const id2 = particleIdRef.current++;
    setParticles(prev => [...prev.slice(-100), 
      { id: id1, fromId, toId, color, startTime: Date.now(), duration },
      { id: id2, fromId: toId, toId: fromId, color, startTime: Date.now() + 150, duration }
    ]);
  }, []);
  
  const animateNode = useCallback((nodeId: string) => {
    setNodeStates(prev => ({ ...prev, [nodeId]: { active: true } }));
    setTimeout(() => setNodeStates(prev => { const n = { ...prev }; delete n[nodeId]; return n; }), 1200);
  }, []);
  
  // Track last successful API response to detect stale/offline data
  const lastSuccessfulFetchRef = useRef<number>(0);
  const lastActiveSubsystemsRef = useRef<Record<string, boolean>>({});
  
  // Fetch flow data only - fast endpoint
  useEffect(() => {
    let mounted = true;
    
    const fetchData = async () => {
      if (!mounted) return;
      
      try {
        const resp = await fetch('/api/flow/live');
        const flow = resp.ok ? await resp.json() : {};
        
        if (!mounted) return;
        
        // Mark successful fetch
        lastSuccessfulFetchRef.current = Date.now();
        
        setFlowData(flow);
        
        const stats = flow?.stats || {};
        const prev = lastCountsRef.current;
        const curr = {
          primary: stats.dual_model?.primary_requests || 0,
          backend: stats.dual_model?.backend_requests || 0,
          audio_io: stats.audio_io?.inputs_processed || 0,
          embeddings: stats.embedding_interconnect?.total || 0,
        };
        
        // Primary model activity
        if (curr.primary > prev.primary) {
          createParticle('agent', 'primary', '#ec4899', 800);
          createBidirectionalParticles('context', 'memory', '#22c55e', 900);
          createBidirectionalParticles('context', 'knowledge', '#22c55e', 900);
          createBidirectionalParticles('context', 'embedding', '#06b6d4', 900);
          createBidirectionalParticles('agent', 'pfc', '#22c55e', 900);
          setTimeout(() => {
            createBidirectionalParticles('primary', 'embedding', '#a855f7', 800);
            createBidirectionalParticles('primary', 'memory', '#a855f7', 800);
            createBidirectionalParticles('primary', 'pfc', '#a855f7', 800);
          }, 100);
          animateNode('agent');
          animateNode('primary');
          animateNode('context');
          animateNode('pfc');
        }
        
        // Backend model activity
        if (curr.backend > prev.backend) {
          createParticle('agent', 'backend', '#6366f1', 800);
          createBidirectionalParticles('context', 'memory', '#22c55e', 900);
          createBidirectionalParticles('context', 'knowledge', '#22c55e', 900);
          createBidirectionalParticles('context', 'embedding', '#06b6d4', 900);
          createBidirectionalParticles('agent', 'pfc', '#22c55e', 900);
          setTimeout(() => {
            createBidirectionalParticles('backend', 'embedding', '#a855f7', 800);
            createBidirectionalParticles('backend', 'memory', '#a855f7', 800);
            createBidirectionalParticles('backend', 'pfc', '#a855f7', 800);
          }, 100);
          animateNode('agent');
          animateNode('backend');
          animateNode('pfc');
        }
        
        // Embedding activity
        if (curr.embeddings > prev.embeddings) {
          createBidirectionalParticles('memory', 'embedding', '#a855f7', 900);
          createBidirectionalParticles('knowledge', 'embedding', '#a855f7', 900);
          createBidirectionalParticles('memory', 'pfc', '#a855f7', 900);
          createParticle('embedding', 'memory', '#06b6d4', 700);
          animateNode('embedding');
          animateNode('memory');
          animateNode('knowledge');
        }
        
        // Audio I/O activity
        if (curr.audio_io > prev.audio_io) {
          createParticle('agent', 'audio_io', '#f59e0b', 800);
          setTimeout(() => {
            createBidirectionalParticles('audio_io', 'pfc', '#f97316', 900);
            createBidirectionalParticles('audio_io', 'embedding', '#a855f7', 800);
          }, 100);
          animateNode('agent');
          animateNode('audio_io');
          animateNode('pfc');
        }
        
        lastCountsRef.current = curr;
        
        // Track real tool usage from API
        const toolUsage = stats?.tool_usage || {};
        const activeFromAPI = toolUsage.active || [];
        const frequency = toolUsage.frequency || {};
        
        // Convert tool names to visualization node IDs
        // Based on core/tool_registry.py ToolCategory enum
        const toolToNodeMap: Record<string, string> = {
          // WEB category (ToolCategory.WEB)
          'web_search': 'cat_web',
          'web_fetch': 'cat_web',
          'image_search': 'cat_web',
          'image_download': 'cat_web',
          'image_read': 'cat_web',
          
          // BROWSER category (ToolCategory.BROWSER)
          'browser_launch': 'cat_browser',
          'browser_navigate': 'cat_browser',
          'browser_click': 'cat_browser',
          'browser_type': 'cat_browser',
          'browser_screenshot': 'cat_browser',
          'browser_get_content': 'cat_browser',
          'browser_evaluate': 'cat_browser',
          'browser_wait': 'cat_browser',
          'browser_list': 'cat_browser',
          
          // SCREEN category (ToolCategory.SCREEN)
          'see_screen': 'cat_screen',
          'find_element': 'cat_screen',
          'click_element': 'cat_screen',
          'type_text': 'cat_screen',
          'press_key': 'cat_screen',
          'hotkey': 'cat_screen',
          'scroll': 'cat_screen',
          'drag': 'cat_screen',
          'move_mouse': 'cat_screen',
          'get_mouse_position': 'cat_screen',
          'screenshot_region': 'cat_screen',
          'ocr_text': 'cat_screen',
          'ui_sense': 'cat_screen',
          
          // TERMINAL category (ToolCategory.TERMINAL)
          'terminal_execute': 'cat_terminal',
          'terminal_cwd': 'cat_terminal',
          'terminal_history': 'cat_terminal',
          'computer_status': 'cat_terminal',
          'install_package': 'cat_terminal',
          
          // SYSTEM category (ToolCategory.SYSTEM)
          'shell': 'cat_system',
          'read_file': 'cat_system',
          'write_file': 'cat_system',
          'list_files': 'cat_system',
          'delete_file': 'cat_system',
          
          // DATA category (ToolCategory.DATA)
          'data_analyze': 'cat_data',
          'data_visualize': 'cat_data',
          'sql_query': 'cat_data',
          'statistics': 'cat_data',
          'physics_sim': 'cat_data',
          'art_analysis': 'cat_data',
          'collision_detect': 'cat_data',
          
          // MULTIMODAL category (ToolCategory.MULTIMODAL)
          'vision': 'cat_multimodal',
          'webcam': 'cat_multimodal',
          'transcribe': 'cat_multimodal',
          'speech_tokenize': 'cat_multimodal',
          'process_speech': 'cat_multimodal',
          'speech_synthesize': 'cat_multimodal',
          'analyze_tone': 'cat_multimodal',
          'voice_stats': 'cat_multimodal',
          'identify_speaker': 'cat_multimodal',
          
          // MEMORY category (ToolCategory.MEMORY)
          'insight_search': 'cat_memory',
          'insight_heatmap': 'cat_memory',
          'insight_stats': 'cat_memory',
          'knowledge_feed': 'cat_memory',
          'get_context': 'cat_memory',
          
          // SECURITY category (ToolCategory.SECURITY)
          'security_scan': 'cat_security',
          'security_audit': 'cat_security',
          'defense': 'cat_security',
          'dlp_whitelist': 'cat_security',
          
          // HOME category (ToolCategory.HOME)
          'home_control': 'cat_home',
          'home_status': 'cat_home',
          
          // AUTOMATION category (ToolCategory.AUTOMATION)
          'docker': 'cat_automation',
          'docker_compose': 'cat_automation',
          'n8n': 'cat_automation',
          
          // SPATIAL category (ToolCategory.SPATIAL)
          'particle_sim': 'cat_spatial',
          'physics_analyze': 'cat_spatial',
          'spatial_calc': 'cat_spatial',
          'design_calc': 'cat_spatial',
          
          // META category (ToolCategory.META)
          'list_tools': 'cat_meta',
          'get_tool_info': 'cat_meta',
          'get_capabilities': 'cat_meta',
          'find_related_tools': 'cat_meta',
          'self_reflect': 'cat_meta',
          'design_process': 'cat_meta',
          'review_process': 'cat_meta',
          
          // LEARNING category (ToolCategory.LEARNING)
          'erdos_math': 'cat_learning',
          'daydream': 'cat_learning',
          'learning': 'cat_learning',
          'sandbox_explore': 'cat_learning',
          'sketch': 'cat_learning',
          'verified_solutions': 'cat_learning',
          
          // EVOLUTION category (ToolCategory.EVOLUTION)
          'evolution': 'cat_evolution',
          'self_edit': 'cat_evolution',
          'evolution_status': 'cat_evolution',
          'evolution_pathway': 'cat_evolution',
          'gene_list': 'cat_evolution',
          'gene_approval': 'cat_evolution',
          
          // REFINEMENT category (ToolCategory.REFINEMENT)
          'refine_scan': 'cat_refinement',
          'refine_apply': 'cat_refinement',
          'refine_status': 'cat_refinement',
          'refine_revert': 'cat_refinement',
          
          // MUSIC tools (custom - part of other)
          'generate_composition': 'cat_music',
          'generate_variation': 'cat_music',
          'tokenize_music_file': 'cat_music',
          'analyze_harmony': 'cat_music',
          'classify_style': 'cat_music',
          'ingest_music_library': 'cat_music',
          
          // Notes/reminders (system category)
          'notes': 'cat_system',
          'reminders': 'cat_system',
        };
        
        // Get tools actually being used (from active list or high frequency)
        const activeToolNodes: string[] = [];
        activeFromAPI.forEach((toolName: string) => {
          const nodeId = toolToNodeMap[toolName] || `cat_${toolName}`;
          if (NODES.find(n => n.id === nodeId)) {
            activeToolNodes.push(nodeId);
          }
        });
        
        // Also check frequency counts for recently used tools
        Object.entries(frequency).forEach(([toolName, count]) => {
          if ((count as number) > 0) {
            const nodeId = toolToNodeMap[toolName] || `cat_${toolName}`;
            if (NODES.find(n => n.id === nodeId) && !activeToolNodes.includes(nodeId)) {
              activeToolNodes.push(nodeId);
            }
          }
        });
        
        lastActiveToolsRef.current = activeToolNodes;
        
        // Only animate if there are actual tools being used
        if (activeToolNodes.length > 0) {
          // DO NOT animate 'tools' node - it would make ALL tool edges light up
          // Only animate specific category nodes
          
          // Animate only the active tool categories
          const uniqueCategories = [...new Set(activeToolNodes)]; // Remove duplicates
          uniqueCategories.slice(0, 3).forEach((catId: string, idx: number) => {
            setTimeout(() => {
              createBidirectionalParticles('tools', catId, '#14b8a6', 700);
              animateNode(catId);
            }, idx * 100);
          });
          
          // Animate tool learning path
          setTimeout(() => {
            if (activeToolNodes.length > 0) {
              createBidirectionalParticles('tools', 'tool_learning', '#14b8a6', 800);
              animateNode('tool_learning');
              createBidirectionalParticles('tool_learning', 'embedding', '#a855f7', 800);
              createBidirectionalParticles('tool_learning', 'memory', '#a59e0b', 800);
              createBidirectionalParticles('tool_learning', 'pfc', '#f97316', 1000);
            }
          }, 400);
        }
        
        // Subsystem activity
        const subsystems = flow?.stats?.subsystems?.active || [];
        const prevSubs = lastSubsystemStateRef.current;
        
        subsystems.forEach((event: any) => {
          const key = event.name?.toLowerCase();
          if (key && event.status === 'active' && !prevSubs[key]) {
            animateNode(key);
          }
        });
        
        lastSubsystemStateRef.current = Object.fromEntries(
          subsystems.map((e: any) => [e.name?.toLowerCase(), e.status === 'active'])
        );
        
      } catch (e) {
        console.error('Flow visualization error:', e);
      }
    };
    
    fetchData();
    const interval = setInterval(fetchData, 200);
    return () => { mounted = false; clearInterval(interval); };
  }, [createParticle, createBidirectionalParticles, animateNode]);
  
  // Ambient particles for ALL edges - ONLY when there's real verified activity
  useEffect(() => {
    const createAmbient = () => {
      // Check if API is responding (data must be fresh within last 3 seconds)
      const now = Date.now();
      const dataAge = now - lastSuccessfulFetchRef.current;
      if (dataAge > 3000) {
        // Data is stale, API might be down - don't create particles
        return;
      }
      
      const activeSubs = flowData?.stats?.subsystems?.active || [];
      const rate = flowData?.stats?.embedding_interconnect?.rate_per_min || 0;
      
      // Only show activity if there's actual real activity happening
      const hasRealActivity = activeSubs.length > 0 || rate > 0;
      if (!hasRealActivity) return;
      
      const rand = Math.random();
      
      // 1. User query flow
      if (rand < 0.06) {
        createParticle('user', 'agent', '#ec4899', 1000);
        createBidirectionalParticles('agent', 'pfc', '#22c55e', 1000);
      }
      
      // 2. Context bidirectional flows
      if (rand < 0.18) {
        createBidirectionalParticles('context', 'memory', '#22c55e', 1000);
        createBidirectionalParticles('context', 'knowledge', '#22c55e', 1000);
        createBidirectionalParticles('context', 'embedding', '#06b6d4', 1000);
        createBidirectionalParticles('context', 'pfc', '#06b6d4', 1000);
      }
      
      // 3. Memory/Knowledge/PFC flows
      if (rand < 0.18) {
        createBidirectionalParticles('memory', 'embedding', '#a855f7', 1000);
        createBidirectionalParticles('memory', 'pfc', '#a855f7', 1000);
        createBidirectionalParticles('knowledge', 'embedding', '#a855f7', 1000);
        createBidirectionalParticles('knowledge', 'pfc', '#a855f7', 1000);
      }
      
      // 4. Agent to models
      if (rand < 0.12) {
        const which = Math.random();
        if (which < 0.5) {
          createParticle('agent', 'primary', '#6366f1', 800);
          setTimeout(() => {
            createBidirectionalParticles('primary', 'embedding', '#a855f7', 800);
            createBidirectionalParticles('primary', 'memory', '#a855f7', 800);
            createBidirectionalParticles('primary', 'pfc', '#a855f7', 800);
          }, 150);
        } else {
          createParticle('agent', 'backend', '#6366f1', 800);
          setTimeout(() => {
            createBidirectionalParticles('backend', 'embedding', '#a855f7', 800);
            createBidirectionalParticles('backend', 'memory', '#a855f7', 800);
            createBidirectionalParticles('backend', 'pfc', '#a855f7', 800);
          }, 150);
        }
      }
      
      // 5. No ambient tool activity - only show tools when actually used
      // Removed random tool activation that made all tools light up
      
      // Subsystem activities - only for ACTIVELY running subsystems
      activeSubs.forEach((event: any) => {
        const key = event.name?.toLowerCase();
        if (!key) return;
        
        // Track which subsystems we've seen
        lastActiveSubsystemsRef.current[key] = true;
        
        animateNode(key);
        
        if (key === 'erdos' && Math.random() > 0.5) {
          createBidirectionalParticles('erdos', 'backend', '#fbbf24', 1000);
          createBidirectionalParticles('erdos', 'insight_bridge', '#f59e0b', 1000);
          createBidirectionalParticles('erdos', 'knowledge', '#f59e0b', 1000);
          setTimeout(() => {
            createBidirectionalParticles('insight_bridge', 'knowledge', '#f97316', 1000);
            createBidirectionalParticles('insight_bridge', 'evolution', '#f97316', 1000);
            createBidirectionalParticles('erdos', 'pfc', '#fbbf24', 1000);
          }, 200);
        }
        
        if (key === 'daydream' && Math.random() > 0.5) {
          createBidirectionalParticles('daydream', 'primary', '#fbbf24', 1000);
          createBidirectionalParticles('daydream', 'backend', '#fbbf24', 1000);
          createBidirectionalParticles('daydream', 'insight_bridge', '#f59e0b', 1000);
          createBidirectionalParticles('daydream', 'embedding', '#fbbf24', 1000);
          setTimeout(() => {
            createBidirectionalParticles('insight_bridge', 'knowledge', '#f97316', 1000);
            createBidirectionalParticles('daydream', 'pfc', '#fbbf24', 1000);
          }, 200);
        }
        
        // Check for music exploration during daydream
        const currentDaydreamActivity = event.current_activity;
        if (currentDaydreamActivity === 'music') {
          createParticle('daydream', 'tools', '#fbbf24', 700);
          setTimeout(() => {
            createBidirectionalParticles('tools', 'cat_music', '#14b8a6', 800);
            animateNode('cat_music');
            createBidirectionalParticles('cat_music', 'pfc', '#f97316', 1000);
          }, 200);
        }
        
        if (key === 'evolution' && Math.random() > 0.5) {
          createBidirectionalParticles('evolution', 'primary', '#fbbf24', 1000);
          createBidirectionalParticles('evolution', 'backend', '#fbbf24', 1000);
          createParticle('evolution', 'permission_queue', '#fbbf24', 1000);
          setTimeout(() => {
            createParticle('permission_queue', 'user', '#fbbf24', 800);
            createBidirectionalParticles('permission_queue', 'pfc', '#fbbf24', 1000);
          }, 200);
        }
        
        if (key === 'learning' && Math.random() > 0.5) {
          createBidirectionalParticles('learning', 'primary', '#fbbf24', 1000);
          createBidirectionalParticles('learning', 'backend', '#fbbf24', 1000);
          createBidirectionalParticles('learning', 'memory', '#f59e0b', 1000);
          createBidirectionalParticles('learning', 'pfc', '#fbbf24', 1000);
        }
        
        if (key === 'defense' && Math.random() > 0.5) {
          createBidirectionalParticles('defense', 'knowledge', '#f59e0b', 1000);
          createBidirectionalParticles('defense', 'pfc', '#f59e0b', 1000);
        }
        
        if (key === 'proactive' && Math.random() > 0.5) {
          createBidirectionalParticles('proactive', 'knowledge', '#f59e0b', 1000);
          createBidirectionalParticles('proactive', 'pfc', '#f59e0b', 1000);
        }
        
        if (key === 'music_creation' || key === 'music creation') {
          console.log('[FlowViz] Music creation active - creating particles');
          animateNode('music_creation');
          createBidirectionalParticles('music_creation', 'pfc', '#22c55e', 1000);
          createBidirectionalParticles('music_creation', 'embedding', '#a855f7', 1000);
          createBidirectionalParticles('music_creation', 'knowledge', '#06b6d4', 1000);
          setTimeout(() => {
            createBidirectionalParticles('insight_bridge', 'knowledge', '#f97316', 1000);
          }, 200);
        }
      });
      
      // Clean up subsystems that are no longer active
      Object.keys(lastActiveSubsystemsRef.current).forEach(key => {
        const stillActive = activeSubs.some((e: any) => e.name?.toLowerCase() === key && e.status === 'active');
        if (!stillActive) {
          delete lastActiveSubsystemsRef.current[key];
        }
      });
    };
    
    // Only create ambient particles if we have recent successful fetches
    const ambientInterval = setInterval(() => {
      if (lastSuccessfulFetchRef.current > 0) {
        createAmbient();
      }
    }, 350);
    return () => clearInterval(ambientInterval);
  }, [flowData, createParticle, createBidirectionalParticles, animateNode]);
  
  // Cleanup old particles
  useEffect(() => {
    const cleanup = setInterval(() => {
      const now = Date.now();
      setParticles(prev => prev.filter(p => now - p.startTime < 2500));
    }, 100);
    return () => clearInterval(cleanup);
  }, []);
  
  const getNodeById = (id: string) => NODES.find(n => n.id === id);
  
  const getEdgePositions = (fromId: string, toId: string) => {
    const from = getNodeById(fromId);
    const to = getNodeById(toId);
    return from && to ? { x1: from.x, y1: from.y, x2: to.x, y2: to.y } : null;
  };
  
  const getParticlePosition = (particle: Particle, time: number) => {
    const pos = getEdgePositions(particle.fromId, particle.toId);
    if (!pos) return null;
    const elapsed = time - particle.startTime;
    const progress = Math.min(elapsed / particle.duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    return { x: pos.x1 + (pos.x2 - pos.x1) * eased, y: pos.y1 + (pos.y2 - pos.y1) * eased, opacity: Math.sin(progress * Math.PI) };
  };
  
  const [, forceUpdate] = useState(0);
  useEffect(() => {
    let frame: number;
    const animate = () => { forceUpdate(n => n + 1); frame = requestAnimationFrame(animate); };
    frame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frame);
  }, []);
  
  const time = Date.now();
  const stats = flowData?.stats || {};
  const embeddingRate = stats.embedding_interconnect?.rate_per_min || 0;
  const activeSubs = stats.subsystems?.active || [];
  const activeCount = activeSubs.length;
  
  return (
    <div className="flow-viz">
      <div className="flow-viz-header">
        <h3>Jarvis Architecture Flow</h3>
        <div className="flow-viz-status">
          <span className={`live-badge ${particles.length > 0 ? 'active' : ''}`}>● LIVE</span>
          <span className="stats-summary">
            {embeddingRate > 0 && <span className="rate-indicator">{embeddingRate.toFixed(1)}/min</span>}
            {stats.dual_model?.total_requests || 0} req
          </span>
        </div>
      </div>
      
      <div className="flow-diagram-container">
        <svg className="flow-svg" viewBox="0 0 920 550" preserveAspectRatio="xMidYMid meet">
          <defs>
            <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#666" /></marker>
            <filter id="pulse-glow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="3" result="blur"><animate attributeName="stdDeviation" values="3;1.5;3" dur="1s" repeatCount="indefinite" /></feGaussianBlur>
              <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
            <filter id="particle-glow">
              <feGaussianBlur stdDeviation="1.5" result="blur" />
              <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
          </defs>
          
          <g className="edges-layer">
            {EDGES.map((edge, i) => {
              const pos = getEdgePositions(edge.from, edge.to);
              if (!pos) return null;
              const isActive = nodeStates[edge.from]?.active || nodeStates[edge.to]?.active;
              return (
                <line 
                  key={i}
                  x1={pos.x1} y1={pos.y1} x2={pos.x2} y2={pos.y2} 
                  stroke={isActive ? '#22c55e' : TYPE_COLORS[edge.type] || '#4b5563'} 
                  strokeWidth={isActive ? 2 : 1} 
                  strokeDasharray={edge.type === 'bidirectional' ? '3,3' : 'none'} 
                  opacity={isActive ? 1 : 0.35}
                />
              );
            })}
          </g>
          
          <g className="particles-layer" filter="url(#particle-glow)">
            {particles.map(p => {
              const pos = getParticlePosition(p, time);
              if (!pos || pos.opacity < 0.1) return null;
              return <circle key={p.id} cx={pos.x} cy={pos.y} r="5" fill={p.color} opacity={pos.opacity} />;
            })}
          </g>
          
          <g className="nodes-layer">
            {NODES.map(node => {
              const isActive = nodeStates[node.id]?.active;
              const size = node.type === 'hub' ? 24 : node.type === 'bridge' ? 22 : 18;
              return (
                <g key={node.id} transform={`translate(${node.x}, ${node.y})`} className={`flow-node-group ${isActive ? 'active' : ''}`}>
                  <rect x={-size} y={-size/2} width={size * 2} height={size} rx={4} className={`flow-node node-${node.type}`} filter={isActive ? 'url(#pulse-glow)' : undefined} />
                  <text y={-3} textAnchor="middle" fontSize="10" className="node-icon">{node.icon}</text>
                  <text y={8} textAnchor="middle" fontSize="7" className="node-label">{node.label}</text>
                </g>
              );
            })}
          </g>
        </svg>
      </div>
      
      <div className="flow-legend">
        <div className="legend-row">
          <span className="legend-item"><span className="legend-dot" style={{background: '#ec4899'}}></span>Input</span>
          <span className="legend-item"><span className="legend-dot" style={{background: '#6366f1'}}></span>Model Call</span>
          <span className="legend-item"><span className="legend-dot" style={{background: '#a855f7'}}></span>Embedding</span>
          <span className="legend-item"><span className="legend-dot" style={{background: '#22c55e'}}></span>Bidirectional</span>
          <span className="legend-item"><span className="legend-dot" style={{background: '#f97316'}}></span>Insight</span>
          <span className="legend-item"><span className="legend-dot" style={{background: '#f59e0b'}}></span>Internal</span>
          <span className="legend-item"><span className="legend-dot" style={{background: '#14b8a6'}}></span>Tool</span>
        </div>
      </div>
      
      <div className="flow-stats-grid">
        <div className="stat-card"><span className="stat-icon">🎭</span><span className="stat-label">Primary</span><span className="stat-value">{(stats.dual_model?.primary_requests || 0).toLocaleString()}</span></div>
        <div className="stat-card"><span className="stat-icon">🧙</span><span className="stat-label">Backend</span><span className="stat-value">{(stats.dual_model?.backend_requests || 0).toLocaleString()}</span></div>
        <div className="stat-card"><span className="stat-icon">🔊</span><span className="stat-label">Audio I/O</span><span className="stat-value">{(stats.audio_io?.inputs_processed || 0).toLocaleString()}</span></div>
        <div className="stat-card hub"><span className="stat-icon">🔗</span><span className="stat-label">Embeddings</span><span className="stat-value">{(stats.embedding_interconnect?.total || 0).toLocaleString()}</span></div>
        <div className="stat-card"><span className="stat-icon">💾</span><span className="stat-label">Vectors</span><span className="stat-value">{(stats.vector_memory?.vector_count || 0).toLocaleString()}</span></div>
        <div className="stat-card"><span className="stat-icon">🕸️</span><span className="stat-label">Knowledge</span><span className="stat-value">{(stats.knowledge_graph?.nodes || 0).toLocaleString()}</span></div>
        <div className="stat-card"><span className="stat-icon">⚡</span><span className="stat-label">Rate/min</span><span className="stat-value">{(stats.dual_model?.total_requests_per_min || 0)}</span></div>
        <div className="stat-card"><span className="stat-icon">📊</span><span className="stat-label">Total</span><span className="stat-value">{(stats.dual_model?.total_requests || 0).toLocaleString()}</span></div>
      </div>
      
      <div className="flow-stats-grid pfc-stats">
        <div className="stat-card pfc"><span className="stat-icon">🧠</span><span className="stat-label">Prefrontal Cortex</span><span className="stat-value">{(stats.prefrontal_cortex?.total_concepts || 0).toLocaleString()} concepts</span></div>
        <div className="stat-card"><span className="stat-icon">📊</span><span className="stat-label">Training Steps</span><span className="stat-value">{(stats.prefrontal_cortex?.training_steps || 0).toLocaleString()}</span></div>
        <div className="stat-card"><span className="stat-icon">📈</span><span className="stat-label">Confidence</span><span className="stat-value">{((stats.prefrontal_cortex?.current_confidence || 0) * 100).toFixed(1)}%</span></div>
        <div className="stat-card"><span className="stat-icon">📚</span><span className="stat-label">Training Buffer</span><span className="stat-value">{(stats.prefrontal_cortex?.training_buffer || 0).toLocaleString()}</span></div>
        <div className="stat-card"><span className="stat-icon">🔤</span><span className="stat-label">Semantic Tokens</span><span className="stat-value">{(stats.prefrontal_cortex?.total_semantic_tokens || 0).toLocaleString()}</span></div>
        {stats.prefrontal_cortex?.avg_loss && <div className="stat-card"><span className="stat-icon">📉</span><span className="stat-label">Avg Loss</span><span className="stat-value">{(stats.prefrontal_cortex.avg_loss).toFixed(4)}</span></div>}
      </div>
      
      <div className="flow-subsystems">
        <div className="subs-header">
          <span className="subs-title">Background Subsystems</span>
          {activeCount > 0 && <span className="active-badge">{activeCount} running</span>}
        </div>
        <div className="subsystems-grid">
          {['erdos', 'daydream', 'evolution', 'learning', 'defense', 'proactive'].map(name => {
            const subsInfo: Record<string, {icon: string, label: string, model: string}> = {
              erdos: { icon: '🔢', label: 'Erdős', model: 'Backend' },
              daydream: { icon: '💭', label: 'Daydream', model: 'Dual' },
              evolution: { icon: '🧬', label: 'Evolution', model: 'Dual' },
              learning: { icon: '📈', label: 'Learning', model: 'Dual' },
              defense: { icon: '🛡️', label: 'Defense', model: 'Monitor' },
              proactive: { icon: '🎯', label: 'Proactive', model: 'Dual' },
            };
            const info = subsInfo[name];
            const isActive = activeSubs.some((s: any) => s.name?.toLowerCase() === name);
            return (
              <div key={name} className={`subsystem-card ${isActive ? 'running' : 'idle'}`}>
                <span className="sub-icon">{info.icon}</span>
                <span className="sub-name">{info.label}</span>
                <span className="sub-model">{info.model}</span>
                <span className={`sub-status ${isActive ? 'active' : ''}`}>{isActive ? '●' : '○'}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}