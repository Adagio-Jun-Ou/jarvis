import { useState, useRef, useCallback } from 'react';
import './MediaInput.css';

interface MediaInputProps {
  onVoiceRecord: (audioBlob: Blob) => void;
  onImageUpload: (imageData: string) => void;
  onImageGenerate: (prompt: string) => void;
  isRecording: boolean;
  isProcessing: boolean;
  disabled?: boolean;
}

export function MediaInput({
  onVoiceRecord,
  onImageUpload,
  onImageGenerate,
  isRecording,
  isProcessing,
  disabled,
}: MediaInputProps) {
  const [showImagePrompt, setShowImagePrompt] = useState(false);
  const [imagePrompt, setImagePrompt] = useState('');
  const [recordingTime, setRecordingTime] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        onVoiceRecord(blob);
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop());
          streamRef.current = null;
        }
        if (timerRef.current) {
          clearInterval(timerRef.current);
          timerRef.current = null;
        }
        setRecordingTime(0);
      };

      mediaRecorder.start();
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
    } catch (error) {
      console.error('Failed to start recording:', error);
      alert('Could not access microphone. Please check permissions.');
    }
  }, [onVoiceRecord]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
  }, []);

  const handleImageUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const dataUrl = e.target?.result as string;
        onImageUpload(dataUrl);
      };
      reader.readAsDataURL(file);
    }
    event.target.value = '';
  }, [onImageUpload]);

  const handleGenerateImage = useCallback(() => {
    if (imagePrompt.trim()) {
      onImageGenerate(imagePrompt.trim());
      setImagePrompt('');
      setShowImagePrompt(false);
    }
  }, [imagePrompt, onImageGenerate]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="media-input">
      <div className="media-buttons">
        <button
          className={`media-btn voice-btn ${isRecording ? 'recording' : ''}`}
          onClick={isRecording ? stopRecording : startRecording}
          disabled={disabled || isProcessing}
          title={isRecording ? 'Stop recording' : 'Start voice input'}
        >
          {isRecording ? (
            <>
              <span className="recording-indicator"></span>
              <span className="recording-time">{formatTime(recordingTime)}</span>
            </>
          ) : (
            '🎤'
          )}
        </button>

        <button
          className="media-btn"
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled || isProcessing}
          title="Upload image"
        >
          📷
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleImageUpload}
          style={{ display: 'none' }}
        />

        <button
          className="media-btn"
          onClick={() => setShowImagePrompt(!showImagePrompt)}
          disabled={disabled || isProcessing}
          title="Generate image"
        >
          🎨
        </button>
      </div>

      {showImagePrompt && (
        <div className="image-prompt-dialog">
          <div className="image-prompt-header">
            <span>Generate Image</span>
            <button 
              className="close-btn"
              onClick={() => setShowImagePrompt(false)}
            >
              ×
            </button>
          </div>
          <textarea
            value={imagePrompt}
            onChange={(e) => setImagePrompt(e.target.value)}
            placeholder="Describe the image you want to generate..."
            rows={3}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleGenerateImage();
              }
            }}
          />
          <div className="image-prompt-actions">
            <select 
              className="style-select"
              defaultValue="realistic"
              onChange={(e) => setImagePrompt(prev => prev + ` [style: ${e.target.value}]`)}
            >
              <option value="realistic">Realistic</option>
              <option value="artistic">Artistic</option>
              <option value="cartoon">Cartoon</option>
              <option value="professional">Professional</option>
            </select>
            <button
              className="generate-btn"
              onClick={handleGenerateImage}
              disabled={!imagePrompt.trim() || isProcessing}
            >
              {isProcessing ? 'Generating...' : 'Generate'}
            </button>
          </div>
        </div>
      )}

      {isProcessing && (
        <div className="processing-indicator">
          <span className="processing-spinner"></span>
          <span>Processing...</span>
        </div>
      )}
    </div>
  );
}