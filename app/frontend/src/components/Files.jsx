import React, { useRef, useState } from 'react';
import { useResource } from '../lib/swr.js';
import { toast } from '../lib/toast.jsx';
import { downloadArtifact } from '../lib/download.js';
import { IconFiles, IconDownload } from './icons.jsx';

const GLYPH = { pdf: '📄', csv: '📊', xlsx: '📊', png: '🖼', jpg: '🖼', jpeg: '🖼', json: '{}', txt: '📄', md: '📝', py: '🐍', zip: '🗜' };
const ext = (n) => (n.split('.').pop() || '').toLowerCase();

export default function Files({ session }) {
  const authHeaders = () => session?.authHeaders?.() || {};
  const [drag, setDrag] = useState(false);
  const fileInput = useRef(null);
  const { data, loading, error, retry } = useResource(
    session?.email ? `files:${session.email}` : null,
    (signal) => fetch('/api/files/list', { signal, credentials: 'include', headers: authHeaders() })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status}`)))),
  );
  const folders = data || {};

  const download = (folder, name) => downloadArtifact(`${folder}/${name}`, authHeaders);

  const upload = async (fileList) => {
    for (const f of Array.from(fileList || [])) {
      try {
        const fd = new FormData(); fd.append('file', f);
        const r = await fetch('/api/files/upload', { method: 'POST', credentials: 'include', headers: authHeaders(), body: fd });
        if (!r.ok) throw new Error(`${r.status}`);
        toast(`Uploaded ${f.name}`, 'success');
      } catch { toast(`Upload failed: ${f.name}`); }
    }
    retry();
  };

  const sections = ['uploads', 'work', 'outputs'].filter((k) => (folders[k] || []).length || k === 'uploads');
  const total = Object.values(folders).reduce((n, arr) => n + (arr?.length || 0), 0);

  return (
    <div className="page">
      <div className="page-head"><h2>Files</h2><p>Everything you’ve shared with Alex and everything it has produced.</p></div>

      <div
        className={`dropzone${drag ? ' drag' : ''}`}
        onClick={() => fileInput.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); upload(e.dataTransfer.files); }}
      >
        <input type="file" hidden multiple ref={fileInput} onChange={(e) => upload(e.target.files)} />
        Drop files here or click to upload
      </div>

      {loading && !total ? (
        <div className="files-grid">{Array.from({ length: 6 }).map((_, i) => <div className="card file-card" key={i}><div className="skeleton" style={{ width: 40, height: 40 }} /><div className="skeleton skeleton-row" style={{ width: '80%' }} /></div>)}</div>
      ) : error ? (
        <div className="error-inline">Couldn’t load files. <button className="btn ghost" onClick={retry}>Retry</button></div>
      ) : !total ? (
        <div className="empty-state"><div className="glyph"><IconFiles width={26} height={26} /></div><h3>No files yet</h3><p>Upload a document or let Alex generate one — it’ll show up here.</p></div>
      ) : (
        sections.map((folder) => (
          (folders[folder] || []).length ? (
            <div className="files-section" key={folder}>
              <span className="section-label">{folder}</span>
              <div className="files-grid">
                {folders[folder].map((name) => (
                  <button type="button" className="card file-card" key={name} onClick={() => download(folder, name)} title={`Download ${name}`}>
                    <div className="file-glyph">{GLYPH[ext(name)] || <IconFiles width={20} height={20} />}</div>
                    <div className="fname">{name}</div>
                    <div className="fmeta"><IconDownload width={12} height={12} /> Download</div>
                  </button>
                ))}
              </div>
            </div>
          ) : null
        ))
      )}
    </div>
  );
}
