import { useState, useEffect, useRef } from 'react';
import './MusicTab.css';

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
  symbolic_tokens?: number;
  spectral_tokens?: number;
}

interface MusicStats {
  running: boolean;
  compositions_created: number;
  tokens_observed: number;
  current_style: string | null;
  creation_interval: number;
  themes_stored: number;
}

interface PFCStats {
  total_concepts: number;
  stored_concepts: number;
  active_concepts: number;
  training_steps: number;
  total_semantic_tokens: number;
}

const STYLE_EMOJIS: Record<string, string> = {
  baroque: '🏰',
  classical: '🎻',
  romantic: '🎹',
  modern: '🎺',
};

const FORM_NAMES: Record<string, string> = {
  binary: 'Binary (AB)',
  ternary: 'Ternary (ABA)',
  sonata: 'Sonata Form',
  rondo: 'Rondo',
  theme_variations: 'Theme & Variations',
  fugue: 'Fugue',
  invention: 'Invention',
  prelude: 'Prelude',
  nocturne: 'Nocturne',
};

const CHARACTER_EMOJIS: Record<string, string> = {
  joyful: '😊',
  melancholic: '😢',
  dramatic: '🎭',
  peaceful: '☮️',
  heroic: '🦸',
  mysterious: '🌙',
  playful: '🎮',
};

interface AudioCache {
  [key: string]: string;  // base64 audio data
}

export function MusicTab() {
  const [compositions, setCompositions] = useState<Composition[]>([]);
  const [stats, setStats] = useState<MusicStats | null>(null);
  const [pfcStats, setPfcStats] = useState<PFCStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [playing, setPlaying] = useState<string | null>(null);
  const [loadingAudio, setLoadingAudio] = useState<string | null>(null);
  const [selectedComp, setSelectedComp] = useState<Composition | null>(null);
  const [audioCache, setAudioCache] = useState<AudioCache>({});
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioSourceRef = useRef<AudioBufferSourceNode | null>(null);

  useEffect(() => {
    fetchMusicData();
    const interval = setInterval(fetchMusicData, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchMusicData = async () => {
    try {
      // Fetch music creation stats
      const flowRes = await fetch('/api/flow/live');
      if (flowRes.ok) {
        const flowData = await flowRes.json();
        if (flowData.stats?.music_creation) {
          setStats(flowData.stats.music_creation);
        }
        if (flowData.stats?.prefrontal_cortex) {
          setPfcStats({
            total_concepts: flowData.stats.prefrontal_cortex.total_concepts || 0,
            stored_concepts: flowData.stats.prefrontal_cortex.stored_concepts || 0,
            active_concepts: flowData.stats.prefrontal_cortex.active_concepts || 0,
            training_steps: flowData.stats.prefrontal_cortex.training_steps || 0,
            total_semantic_tokens: flowData.stats.prefrontal_cortex.total_semantic_tokens || 0,
          });
        }
      }

      // Fetch recent compositions
      const musicRes = await fetch('/api/music/compositions');
      if (musicRes.ok) {
        const musicData = await musicRes.json();
        setCompositions(musicData.compositions || []);
      }
    } catch (error) {
      console.error('Failed to fetch music data:', error);
    } finally {
      setLoading(false);
    }
  };

  const playComposition = async (comp: Composition) => {
    if (playing === comp.id) {
      stopPlayback();
      return;
    }

    try {
      setLoadingAudio(comp.id);

      // Check cache first
      let audioData = audioCache[comp.id];
      
      if (!audioData) {
        // Fetch audio from server
        const audioRes = await fetch(`/api/music/compositions/${comp.id}/audio`);
        if (!audioRes.ok) {
          throw new Error('Failed to fetch audio');
        }
        const audioJson = await audioRes.json();
        if (!audioJson.success) {
          throw new Error(audioJson.error || 'Failed to generate audio');
        }
        audioData = audioJson.audio;
        
        // Cache for future use
        setAudioCache(prev => ({ ...prev, [comp.id]: audioData }));
      }

      // Decode and play
      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
      }
      const ctx = audioContextRef.current;

      // Decode base64 to ArrayBuffer
      const binaryString = atob(audioData);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }

      // Decode audio data
      const audioBuffer = await ctx.decodeAudioData(bytes.buffer);

      // Stop any existing playback
      if (audioSourceRef.current) {
        try {
          audioSourceRef.current.stop();
        } catch (e) {}
      }

      // Create and play source
      const source = ctx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(ctx.destination);
      source.start(0);
      
      audioSourceRef.current = source;
      setLoadingAudio(null);
      setPlaying(comp.id);

      // Clear playing state when audio ends
      source.onended = () => {
        if (playing === comp.id) {
          setPlaying(null);
        }
      };

    } catch (error) {
      console.error('Playback error:', error);
      setLoadingAudio(null);
      setPlaying(null);
    }
  };

  const stopPlayback = () => {
    if (audioSourceRef.current) {
      try {
        audioSourceRef.current.stop();
      } catch (e) {}
    }
    audioSourceRef.current = null;
    setPlaying(null);
  };

  const formatTime = (ts: string) => {
    if (!ts) return '';
    try {
      const date = new Date(ts);
      return date.toLocaleString();
    } catch {
      return '';
    }
  };

  return (
    <div className="music-tab">
      <div className="music-header">
        <h2>🎵 Music Creation</h2>
        <div className="music-status">
          {stats?.running ? (
            <span className="status-running">● Always-on Background Composition</span>
          ) : (
            <span className="status-stopped">○ Paused</span>
          )}
        </div>
      </div>

      <div className="music-stats-grid">
        <div className="stat-card">
          <div className="stat-value">{stats?.compositions_created || 0}</div>
          <div className="stat-label">Compositions</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{stats?.tokens_observed || 0}</div>
          <div className="stat-label">Tokens Observed</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{stats?.themes_stored || 0}</div>
          <div className="stat-label">Themes</div>
        </div>
        <div className="stat-card pfc-card">
          <div className="stat-value">{pfcStats?.stored_concepts || 0}</div>
          <div className="stat-label">PFC Concepts</div>
          <div className="stat-sublabel">{pfcStats?.total_semantic_tokens?.toLocaleString() || 0} semantic tokens</div>
        </div>
      </div>

      <div className="music-creation-info">
        <h3>Continuous Composition Process</h3>
        <div className="process-flow">
          <div className="process-step">
            <span className="step-icon">🎼</span>
            <span className="step-label">Generate</span>
            <span className="step-desc">Create composition with control + spontaneity</span>
          </div>
          <div className="process-arrow">→</div>
          <div className="process-step">
            <span className="step-icon">🔊</span>
            <span className="step-label">Synthesize</span>
            <span className="step-desc">Acoustic tokens (spectral features)</span>
          </div>
          <div className="process-arrow">→</div>
          <div className="process-step">
            <span className="step-icon">🧠</span>
            <span className="step-label">Observe</span>
            <span className="step-desc">PFC learns symbolic + acoustic patterns</span>
          </div>
          <div className="process-arrow">→</div>
          <div className="process-step">
            <span className="step-icon">💡</span>
            <span className="step-label">Insight</span>
            <span className="step-desc">Feed compositional insights to Knowledge Graph</span>
          </div>
        </div>
      </div>

      <div className="music-section">
        <h3>Recent Compositions</h3>
        {loading ? (
          <div className="music-loading">Loading compositions...</div>
        ) : compositions.length === 0 ? (
          <div className="music-empty">
            <div className="empty-icon">🎵</div>
            <div className="empty-text">No compositions yet</div>
            <div className="empty-hint">Music creation runs in background every 2 minutes</div>
          </div>
        ) : (
          <div className="compositions-list">
            {compositions.slice(0, 20).map(comp => (
              <div 
                key={comp.id} 
                className={`composition-item ${selectedComp?.id === comp.id ? 'selected' : ''}`}
                onClick={() => setSelectedComp(comp)}
              >
                <div className="comp-header">
                  <span className="comp-style">{STYLE_EMOJIS[comp.style] || '🎵'}</span>
                  <span className="comp-title">{comp.title}</span>
                  <span className="comp-character">{CHARACTER_EMOJIS[comp.character] || '🎵'}</span>
                </div>
                <div className="comp-details">
                  <span className="comp-form">{FORM_NAMES[comp.form] || comp.form}</span>
                  <span className="comp-tempo">{comp.tempo} BPM</span>
                  <span className="comp-key">{comp.key.replace('_', ' ')}</span>
                </div>
                <div className="comp-meta">
                  <span className="comp-tokens">
                    {comp.symbolic_tokens || 0} symbolic · {comp.spectral_tokens || 0} acoustic
                  </span>
                  <span className="comp-time">{formatTime(comp.created_at)}</span>
                </div>
                {selectedComp?.id === comp.id && (
                  <div className="comp-expanded">
                    <div className="comp-stat">
                      <span className="label">PFC Observations:</span>
                      <span className="value">{comp.observations}</span>
                    </div>
                    <button 
                      className={`play-btn ${playing === comp.id ? 'playing' : ''} ${loadingAudio === comp.id ? 'loading' : ''}`}
                      onClick={(e) => { e.stopPropagation(); playComposition(comp); }}
                      disabled={loadingAudio !== null && loadingAudio !== comp.id}
                    >
                      {loadingAudio === comp.id ? '⏳ Loading...' : 
                       playing === comp.id ? '⏹ Stop' : '▶ Play Full'}
                    </button>
                    {playing === comp.id && (
                      <span className="playing-indicator">🎵 Playing...</span>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="music-theory">
        <h3>Classical Music Theory Foundation</h3>
        <div className="theory-grid">
          <div className="theory-item">
            <h4>🎵 Styles</h4>
            <ul>
              <li><strong>Baroque</strong> (1600-1750): High ornamentation, counterpoint, terraced dynamics</li>
              <li><strong>Classical</strong> (1750-1820): Clear phrases, balance, symmetry</li>
              <li><strong>Romantic</strong> (1820-1900): Expression, chromaticism, rubato</li>
            </ul>
          </div>
          <div className="theory-item">
            <h4>📐 Forms</h4>
            <ul>
              <li><strong>Sonata</strong>: Exposition → Development → Recapitulation</li>
              <li><strong>Rondo</strong>: ABACABA alternation</li>
              <li><strong>Theme & Variations</strong>: Systematic transformations</li>
            </ul>
          </div>
          <div className="theory-item">
            <h4>🎛️ Control vs Spontaneity</h4>
            <ul>
              <li><strong>Control (85%)</strong>: Functional harmony, voice leading, form</li>
              <li><strong>Spontaneity (15%)</strong>: Surprise modulations, cross-style ornaments</li>
            </ul>
          </div>
          <div className="theory-item">
            <h4>🧠 PFC Learning</h4>
            <ul>
              <li><strong>Symbolic Tokens</strong>: Style, form, key, harmony progressions</li>
              <li><strong>Acoustic Tokens</strong>: Spectral centroid, RMS, frequencies (10 fps)</li>
              <li><strong>Pattern Mining</strong>: Recurring motifs, harmonic sequences</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}