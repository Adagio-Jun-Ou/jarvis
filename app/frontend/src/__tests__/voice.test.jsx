// Voice helpers: markdown becomes prose, TTS degrades to silence
// honestly, transcription posts the clip under the session's identity.

import { it, expect, vi, afterEach } from 'vitest';
import { speakable, speak, transcribe } from '../lib/voice.js';

afterEach(() => vi.restoreAllMocks());

const auth = () => ({ Authorization: 'Bearer t' });

it('speakable strips markdown to prose', () => {
  expect(speakable('# Title\n\n**bold** and `code` and [link](http://x)\n```js\nx\n```'))
    .toBe('Title bold and code and link code block omitted.');
});

it('speak returns false on 204 (no backend) without touching Audio', async () => {
  vi.stubGlobal('fetch', vi.fn(async () => ({ status: 204 })));
  vi.stubGlobal('Audio', vi.fn());
  expect(await speak('hello', auth)).toBe(false);
  expect(Audio).not.toHaveBeenCalled();
});

it('transcribe posts the clip as multipart with auth and returns text', async () => {
  const calls = [];
  vi.stubGlobal('fetch', vi.fn(async (url, opts) => {
    calls.push({ url, opts });
    return { ok: true, json: async () => ({ text: 'turn the lights off' }) };
  }));
  const blob = new Blob([new Uint8Array([1, 2])], { type: 'audio/webm' });
  expect(await transcribe(blob, auth)).toBe('turn the lights off');
  expect(calls[0].url).toBe('/api/voice/transcribe');
  expect(calls[0].opts.headers.Authorization).toBe('Bearer t');
  expect(calls[0].opts.body).toBeInstanceOf(FormData);
});
