import React, { useState, useEffect, useRef, useCallback } from 'react';

interface GraphNode {
  id: string;
  label: string;
  type: 'concept' | 'memory' | 'insight' | 'tool';
  x: number;
  y: number;
  connections: string[];
}

interface VizData {
  nodes: GraphNode[];
  concepts: string[];
  recent_insights: string[];
  tool_usage: Record<string, number>;
}

export const MobileVisual: React.FC = () => {
  const [vizData, setVizData] = useState<VizData | null>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [viewMode, setViewMode] = useState<'graph' | 'insights' | 'tools'>('insights');
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const fetchVizData = async () => {
      try {
        const [statusResp, kgResp] = await Promise.all([
          fetch('/api/status'),
          fetch('/api/knowledge-graph/stats'),
        ]);
        
        const status = await statusResp.json();
        let kgData = { nodes: [], edges: [] };
        
        try {
          kgData = await kgResp.json();
        } catch {
          // Knowledge graph might not be available
        }
        
        const nodes: GraphNode[] = [];
        const concepts: string[] = [];
        const recent_insights: string[] = [];
        const tool_usage: Record<string, number> = {};
        
        // Add memory nodes
        if (status.memory_messages) {
          nodes.push({
            id: 'memory',
            label: `${status.memory_messages} Messages`,
            type: 'memory',
            x: 100,
            y: 100,
            connections: [],
          });
        }
        
        // Add model nodes
        if (status.model) {
          nodes.push({
            id: 'primary',
            label: status.model,
            type: 'concept',
            x: 200,
            y: 80,
            connections: ['memory'],
          });
        }
        
        if (status.backend_model) {
          nodes.push({
            id: 'backend',
            label: status.backend_model,
            type: 'concept',
            x: 200,
            y: 140,
            connections: ['memory'],
          });
        }
        
        // Add subsystem nodes
        if (status.erdos_problems) {
          nodes.push({
            id: 'erdos',
            label: `Erdős (${status.erdos_problems.toLocaleString()})`,
            type: 'tool',
            x: 80,
            y: 200,
            connections: ['backend'],
          });
        }
        
        if (status.daydream?.daydream_count) {
          nodes.push({
            id: 'daydream',
            label: `Daydream (${status.daydream.daydream_count})`,
            type: 'tool',
            x: 200,
            y: 200,
            connections: ['primary', 'backend'],
          });
        }
        
        if (status.insights_processed) {
          nodes.push({
            id: 'insights',
            label: `${status.insights_processed} Insights`,
            type: 'insight',
            x: 320,
            y: 200,
            connections: ['daydream', 'erdos'],
          });
        }
        
        if (status.evolution_genes) {
          nodes.push({
            id: 'evolution',
            label: `Evolution (${status.evolution_genes})`,
            type: 'tool',
            x: 140,
            y: 260,
            connections: ['insights'],
          });
        }
        
        // Use knowledge graph if available
        if (kgData.nodes && kgData.nodes.length > 0) {
          kgData.nodes.slice(0, 10).forEach((n: any, i: number) => {
            nodes.push({
              id: n.id || `kg-${i}`,
              label: n.label || n.name || `Node ${i}`,
              type: 'concept',
              x: 50 + (i % 4) * 80,
              y: 320 + Math.floor(i / 4) * 50,
              connections: [],
            });
          });
        }
        
        setVizData({
          nodes,
          concepts,
          recent_insights,
          tool_usage,
        });
        
      } catch (e) {
        console.error('Failed to fetch viz data:', e);
      }
    };

    fetchVizData();
    const interval = setInterval(fetchVizData, 5000);
    return () => clearInterval(interval);
  }, []);

  const drawGraph = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !vizData) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, rect.width, rect.height);

    // Draw connections
    ctx.strokeStyle = 'rgba(124, 58, 237, 0.3)';
    ctx.lineWidth = 1;
    
    vizData.nodes.forEach(node => {
      node.connections.forEach(targetId => {
        const target = vizData.nodes.find(n => n.id === targetId);
        if (target) {
          ctx.beginPath();
          ctx.moveTo(node.x, node.y);
          ctx.lineTo(target.x, target.y);
          ctx.stroke();
        }
      });
    });

    // Draw nodes
    vizData.nodes.forEach(node => {
      const colors: Record<string, string> = {
        concept: '#7c3aed',
        memory: '#2563eb',
        insight: '#10b981',
        tool: '#f59e0b',
      };
      
      ctx.fillStyle = colors[node.type] || '#7c3aed';
      ctx.beginPath();
      ctx.arc(node.x, node.y, 20, 0, Math.PI * 2);
      ctx.fill();
      
      ctx.fillStyle = '#fff';
      ctx.font = '10px system-ui';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      
      const label = node.label.length > 15 ? node.label.slice(0, 15) + '...' : node.label;
      ctx.fillText(label, node.x, node.y + 32);
    });
  }, [vizData]);

  useEffect(() => {
    if (viewMode === 'graph') {
      drawGraph();
    }
  }, [viewMode, drawGraph]);

  return (
    <div className="visual-view">
      <div className="visual-header">
        <h2>🔮 Visual</h2>
        <div className="view-toggle">
          <button 
            className={viewMode === 'insights' ? 'active' : ''} 
            onClick={() => setViewMode('insights')}
          >
            💡 Insights
          </button>
          <button 
            className={viewMode === 'graph' ? 'active' : ''} 
            onClick={() => setViewMode('graph')}
          >
            🔗 Graph
          </button>
          <button 
            className={viewMode === 'tools' ? 'active' : ''} 
            onClick={() => setViewMode('tools')}
          >
            🔧 Tools
          </button>
        </div>
      </div>

      {viewMode === 'graph' && (
        <div className="graph-container">
          <canvas 
            ref={canvasRef} 
            className="graph-canvas"
            onClick={(e) => {
              const rect = e.currentTarget.getBoundingClientRect();
              const x = e.clientX - rect.left;
              const y = e.clientY - rect.top;
              
              if (vizData) {
                const clicked = vizData.nodes.find(
                  node => Math.abs(node.x - x) < 25 && Math.abs(node.y - y) < 25
                );
                setSelectedNode(clicked || null);
              }
            }}
          />
          {selectedNode && (
            <div className="node-detail">
              <h4>{selectedNode.label}</h4>
              <p>Type: {selectedNode.type}</p>
              <p>Connections: {selectedNode.connections.length}</p>
            </div>
          )}
        </div>
      )}

      {viewMode === 'insights' && vizData && (
        <div className="insights-grid">
          <div className="insight-section">
            <h3>🧠 Active Systems</h3>
            <div className="system-cards">
              {vizData.nodes.filter(n => n.type === 'tool' || n.type === 'insight').map(node => (
                <div key={node.id} className="system-card">
                  <span className="system-label">{node.label}</span>
                  <span className={`system-type ${node.type}`}>{node.type}</span>
                </div>
              ))}
            </div>
          </div>
          
          <div className="insight-section">
            <h3>🤖 Model Architecture</h3>
            <div className="model-flow">
              {vizData.nodes.filter(n => n.type === 'concept').map(node => (
                <div key={node.id} className="flow-node">
                  <span className="flow-label">{node.label}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="insight-section">
            <h3>💾 Memory</h3>
            <div className="memory-stats">
              {vizData.nodes.filter(n => n.type === 'memory').map(node => (
                <div key={node.id} className="memory-card">
                  <div className="memory-icon">📝</div>
                  <div className="memory-info">
                    <span className="memory-value">{node.label}</span>
                    <span className="memory-label">Stored</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {viewMode === 'tools' && vizData && (
        <div className="tools-view">
          <h3>🔧 Active Tools</h3>
          <div className="tool-list">
            {vizData.nodes.filter(n => n.type === 'tool').map(node => (
              <div key={node.id} className="tool-card">
                <div className="tool-header">
                  <span className="tool-name">{node.id}</span>
                  <span className="tool-status">active</span>
                </div>
                <div className="tool-description">{node.label}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default MobileVisual;