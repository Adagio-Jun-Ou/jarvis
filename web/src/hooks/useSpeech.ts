import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * Speech with graceful degradation:
 * 1. Server-side neural TTS (/api/tts — edge-tts or piper) — best quality,
 *    consistent voice across browsers/OSes.
 * 2. Browser speechSynthesis — always available fallback.
 *
 * The server is probed once (cached in module scope) so we never waste a
 * round trip per utterance.
 */

let serverTtsAvailable: boolean | null = null;
let serverVoice: string | null = null;

async function probeServerTts(): Promise<boolean> {
  if (serverTtsAvailable !== null) return serverTtsAvailable;
  try {
    const res = await fetch('/api/tts/state');
    if (!res.ok) {
      serverTtsAvailable = false;
      return false;
    }
    const data = await res.json();
    serverTtsAvailable = !!(data.success && data.backends && data.backends.length > 0);
    serverVoice = data.active_voice || null;
  } catch {
    serverTtsAvailable = false;
  }
  return serverTtsAvailable;
}

export interface SpeechOptions {
  /** Strip markdown/HTML and cap length before synthesis. */
  clean?: boolean;
  maxLength?: number;
}

export function useSpeech(options: SpeechOptions = {}) {
  const { clean = true, maxLength = 500 } = options;
  const [isSpeaking, setIsSpeaking] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const stoppedRef = useRef(false);

  const stop = useCallback(() => {
    stoppedRef.current = true;
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.src = '';
      audioRef.current = null;
    }
    if (typeof speechSynthesis !== 'undefined') {
      speechSynthesis.cancel();
    }
    setIsSpeaking(false);
  }, []);

  const speakWithServer = useCallback(async (utterance: string): Promise<boolean> => {
    try {
      const res = await fetch('/api/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: utterance }),
      });
      if (!res.ok) {
        serverTtsAvailable = false; // re-probe next time
        return false;
      }
      const blob = await res.blob();
      if (!blob.size) return false;
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audioRef.current = audio;
      audio.onended = () => {
        URL.revokeObjectURL(url);
        audioRef.current = null;
        setIsSpeaking(false);
      };
      audio.onerror = () => {
        URL.revokeObjectURL(url);
        audioRef.current = null;
        setIsSpeaking(false);
      };
      stoppedRef.current = false;
      setIsSpeaking(true);
      await audio.play();
      return true;
    } catch {
      return false;
    }
  }, []);

  const speakWithBrowser = useCallback((utterance: string) => {
    if (typeof speechSynthesis === 'undefined') return;
    speechSynthesis.cancel();
    const synth = new SpeechSynthesisUtterance(utterance);
    const voices = speechSynthesis.getVoices();
    const preferred =
      voices.find(v => v.name.includes('Google') && v.lang.startsWith('en')) ||
      voices.find(v => v.lang.startsWith('en') && v.default) ||
      voices.find(v => v.default);
    if (preferred) synth.voice = preferred;
    synth.rate = 1.0;
    synth.pitch = 1.0;
    synth.onstart = () => setIsSpeaking(true);
    synth.onend = () => setIsSpeaking(false);
    synth.onerror = () => setIsSpeaking(false);
    stoppedRef.current = false;
    speechSynthesis.speak(synth);
  }, []);

  const speak = useCallback(
    async (rawText: string) => {
      let text = rawText;
      if (clean) {
        text = text
          .replace(/```[\s\S]*?```/g, ' code block. ')
          .replace(/<[^>]*>/g, '')
          .replace(/[#*_`~|>-]{1,4}/g, ' ')
          .replace(/\s+/g, ' ')
          .trim();
      }
      if (!text) return;
      if (text.length > maxLength) text = text.slice(0, maxLength);

      const available = await probeServerTts();
      if (available) {
        const ok = await speakWithServer(text);
        if (ok) return;
      }
      speakWithBrowser(text);
    },
    [clean, maxLength, speakWithServer, speakWithBrowser],
  );

  useEffect(() => {
    // Warm the availability probe and browser voice list on mount.
    probeServerTts();
    if (typeof speechSynthesis !== 'undefined') {
      speechSynthesis.getVoices();
    }
    return () => stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return { speak, stop, isSpeaking, serverVoice };
}