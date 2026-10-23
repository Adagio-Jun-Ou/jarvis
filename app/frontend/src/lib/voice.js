// Voice — push-to-talk in, spoken replies out.
//
// The browser owns the microphone (MediaRecorder) so listening is
// always something the person can see; the server owns the models
// (whisper on the GPU, neural TTS). Every function degrades to
// text: no mic, no backend, no audio element — the chat still works.

/** Record from the microphone until stop() is called; resolves the blob. */
export function startRecording() {
  if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
    return Promise.reject(new Error('microphone not available'));
  }
  return navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
    const mime = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg']
      .find((m) => typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported?.(m)) || '';
    const rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
    const chunks = [];
    rec.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
    const done = new Promise((resolve) => {
      rec.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        resolve(new Blob(chunks, { type: rec.mimeType || 'audio/webm' }));
      };
    });
    rec.start();
    return { stop: () => { if (rec.state !== 'inactive') rec.stop(); return done; } };
  });
}

/** Send a recorded clip to the server; resolves the transcript text. */
export async function transcribe(blob, authHeaders) {
  const fd = new FormData();
  const ext = (blob.type || 'audio/webm').includes('ogg') ? 'ogg' : 'webm';
  fd.append('audio', blob, `clip.${ext}`);
  const r = await fetch('/api/voice/transcribe', { method: 'POST', headers: authHeaders(), body: fd });
  if (!r.ok) {
    const d = await r.json().catch(() => ({}));
    throw new Error(d.error || `transcription failed (${r.status})`);
  }
  const d = await r.json();
  return String(d.text || '');
}

/** Strip markdown so the voice reads prose, not asterisks. */
export function speakable(markdown) {
  return String(markdown || '')
    .replace(/```[\s\S]*?```/g, ' code block omitted. ')
    .replace(/`([^`]*)`/g, '$1')
    .replace(/!\[[^\]]*\]\([^)]*\)/g, '')
    .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/^#{1,6}\s+/gm, '')
    .replace(/[*_~>]+/g, '')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 3000);
}

let current = null;

/** Speak text through the server's TTS; resolves false when it can't. */
export async function speak(text, authHeaders) {
  const clean = speakable(text);
  if (!clean) return false;
  const r = await fetch('/api/voice/speak', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ text: clean }),
  });
  if (r.status !== 200) return false;
  const blob = await r.blob();
  if (typeof Audio === 'undefined' || typeof URL?.createObjectURL !== 'function') return false;
  stopSpeaking();
  const url = URL.createObjectURL(blob);
  const audio = new Audio(url);
  current = audio;
  audio.onended = () => { URL.revokeObjectURL(url); if (current === audio) current = null; };
  try { await audio.play(); } catch { return false; }
  return true;
}

export function stopSpeaking() {
  if (current) { try { current.pause(); } catch { /* noop */ } current = null; }
}
