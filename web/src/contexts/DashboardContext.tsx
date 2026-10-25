import React, { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';

interface DashboardData {
  timestamp: number;
  status: any;
  dual_model: any;
  daydream: any;
  erdos: any;
  evolution: any;
  refine: any;
  memory: any;
  memory_stats: any;
  models_coordinator: any;
  knowledge_graph: any;
  subsystems: any;
  tools: any;
  hardware: any;
  freetime: any;
  jarvis_state: any;
  recent_thoughts: any;
  ui_state: any;
  proactive: any;
  embeddings: any;
  insights: any;
  vector_entries: any;
  knowledge_nodes: any;
  context: any;
  learning: any;
  defense: any;
  presence?: any;
  anticipation?: any;
  autonomous_observer?: any;
}

interface DashboardContextType {
  data: DashboardData | null;
  loading: boolean;
  error: string | null;
  refresh: () => void;
}

const DashboardContext = createContext<DashboardContextType>({
  data: null,
  loading: false,
  error: null,
  refresh: () => {},
});

export const useDashboard = () => useContext(DashboardContext);

export const DashboardProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const mountedRef = useRef(true);
  const fetchingRef = useRef(false);

  const fetchData = useCallback(async () => {
    if (fetchingRef.current) return;
    fetchingRef.current = true;
    try {
      const resp = await fetch('/api/dashboard');
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const json = await resp.json();
      if (mountedRef.current) {
        setData(json);
        setError(null);
      }
    } catch (e: any) {
      if (mountedRef.current) {
        setError(e.message || 'Failed to fetch dashboard');
      }
    } finally {
      if (mountedRef.current) {
        setLoading(false);
      }
      fetchingRef.current = false;
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    fetchData();
    const interval = setInterval(fetchData, 3000);
    return () => {
      mountedRef.current = false;
      clearInterval(interval);
    };
  }, [fetchData]);

  const refresh = useCallback(() => {
    setLoading(true);
    fetchData();
  }, [fetchData]);

  return (
    <DashboardContext.Provider value={{ data, loading, error, refresh }}>
      {children}
    </DashboardContext.Provider>
  );
};