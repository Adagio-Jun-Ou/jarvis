import { useState, useEffect } from 'react';
import './MusicCard.css';

interface Composition {
  id: string;
  title: string;
  style: string;
  form: string;
  key: string;
  tempo: number;
  character: string;
  observations: number;
  created_at: string;
}

interface MusicStats {
  running: boolean;
  compositions_created: number;
  tokens_observed: number;
  current_style: string | null;
}

export function MusicCard() {
  const [compositions, setCompositions] = useState<Composition[]>([]);
  const [stats, setStats] = useState<MusicStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchMusicData();
    const interval = setInterval(fetchMusicData, 10000);
    return () => clearInterval(interval);
  }, []);

  const fetchMusicData = async () => {
    try {
      const res = await fetch('/api/music/compositions');
      if (res.ok) {
        const data = await res.json();
        setCompositions(data.compositions || []);
        setStats(data.stats || null);
      }
    } catch (error) {
      console.error('Failed to fetch music data:', error);
    } finally {
      setLoading(false);
    }
  };

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

  const STYLE_EMOJIS: Record<string, string> = {
    baroque: '🏰',
    classical: '🎻',
    romantic: '🎹',
    modern: '🎺',
  };

  if (loading) {
    return (
      <div className="music-card loading">
        <div className="music-card-header">
          <h3>🎵 Music Creation</h3>
        </div>
        <div className="music-card-loading">Loading...</div>
      </div>
    );
  }

  return (
    <div className="music-card">
      <div className="music-card-header">
        <h3>🎵 Music Creation</h3>
        <span className={`music-status ${stats?.running ? 'running' : 'stopped'}`}>
          {stats?.running ? '● Always-on' : '○ Paused'}
        </span>
      </div>

      <div className="music-stats">
        <div className="music-stat">
          <span className="stat-value">{stats?.compositions_created || 0}</span>
          <span className="stat-label">Compositions</span>
        </div>
        <div className="music-stat">
          <span className="stat-value">{stats?.tokens_observed || 0}</span>
          <span className="stat-label">Tokens Observed</span>
        </div>
        <div className="music-stat">
          <span className="stat-value">{stats?.current_style || '-'}</span>
          <span className="stat-label">Current Style</span>
        </div>
      </div>

      <div className="music-explanation">
        <p><strong>Control + Spontaneity:</strong> 85% classical music theory, 15% surprise</p>
        <p><strong>PFC Learning:</strong> Symbolic + acoustic tokens observed continuously</p>
      </div>

      {compositions.length > 0 && (
        <div className="music-compositions">
          <h4>Recent Compositions</h4>
          <div className="composition-list">
            {compositions.slice(0, 5).map(comp => (
              <div key={comp.id} className="composition-row">
                <span className="comp-style">{STYLE_EMOJIS[comp.style] || '🎵'}</span>
                <span className="comp-title">{comp.title}</span>
                <span className="comp-meta">
                  {comp.form} · {comp.key.replace('_', ' ')} · {comp.tempo} BPM
                </span>
                <span className="comp-time">{formatTime(comp.created_at)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {compositions.length === 0 && (
        <div className="no-compositions">
          <p>Music creation runs in background every 2 minutes</p>
        </div>
      )}
    </div>
  );
}