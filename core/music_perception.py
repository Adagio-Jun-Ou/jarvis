"""
Music Perception System - Let Jarvis "Hear" His Compositions.

Enables:
1. Audio analysis (spectral, harmonic, rhythmic features)
2. Structure detection (sections, themes, development)
3. Quality evaluation (coherence, development, resolution)
4. Iterative refinement based on listening

This gives Jarvis actual feedback on what he creates.
"""

import numpy as np
import logging
import threading
from dataclasses import dataclass, field
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)

_LIBROSA_AVAILABLE = False
try:
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    pass


@dataclass
class MusicalFeatures:
    """Features extracted from audio that represent what was "heard"."""
    
    # Structural features
    sections_detected: int = 0
    section_boundaries: list = field(default_factory=list)
    theme_occurrences: list = field(default_factory=list)
    
    # Harmonic features
    key: str = "unknown"
    key_confidence: float = 0.0
    tempo_bpm: float = 120.0
    tempo_stability: float = 0.0
    
    # Melodic features
    melodic_range: tuple = (0, 0)  # (min_note, max_note) in MIDI
    melodic_contour: list = field(default_factory=list)  # Up/down/flat
    repeated_phrases: int = 0
    
    # Dynamic features
    dynamic_range: float = 0.0  # RMS variance
    climax_moment: float = 0.0  # When loudest (0-1)
    
    # Quality metrics
    coherence_score: float = 0.0  # Does it sound unified?
    development_score: float = 0.0  # Is there growth/change?
    resolution_score: float = 0.0  # Does it resolve properly?
    structure_score: float = 0.0  # Overall structural quality
    
    # Raw analysis data
    spectral_tokens: list = field(default_factory=list)
    chroma_tokens: list = field(default_factory=list)
    energy_envelope: list = field(default_factory=list)


@dataclass
class CompositionFeedback:
    """Feedback for iterative improvement."""
    
    overall_score: float = 0.0
    strengths: list = field(default_factory=list)
    weaknesses: list = field(default_factory=list)
    
    # Specific issues to address
    issues: list = field(default_factory=list)
    
    # Suggested improvements
    suggestions: list = field(default_factory=list)
    
    # Should retry generation?
    should_regenerate: bool = False
    retry_prompt_hints: list = field(default_factory=list)


class MusicPerceptionEngine:
    """Analyze generated music to provide feedback for improvement.
    
    This gives Jarvis the ability to "hear" his compositions and
    understand what works and what doesn't.
    """
    
    # Quality thresholds
    MIN_COHERENCE_THRESHOLD = 0.4
    MIN_STRUCTURE_THRESHOLD = 0.3
    MIN_DEVELOPMENT_THRESHOLD = 0.3
    
    def analyze_composition(
        self,
        audio: np.ndarray,
        sample_rate: int,
        expected_form: str = "ternary",
        expected_key: str = "C_major",
        expected_tempo: int = 120,
    ) -> MusicalFeatures:
        """Analyze generated audio and extract musical features.
        
        This is how Jarvis "hears" his composition.
        
        Args:
            audio: Generated audio array
            sample_rate: Sample rate
            expected_form: What form was requested
            expected_key: What key was requested
            expected_tempo: What tempo was requested
            
        Returns:
            MusicalFeatures with all detected characteristics
        """
        if not isinstance(audio, np.ndarray):
            audio = np.asarray(audio, dtype=np.float64)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if len(audio) == 0:
            return MusicalFeatures()
        if sample_rate <= 0:
            sample_rate = 22050

        features = MusicalFeatures()
        
        # Enhanced analysis with librosa when available
        librosa_features = self._librosa_analysis(audio, sample_rate) if _LIBROSA_AVAILABLE else None
        
        duration = len(audio) / sample_rate
        logger.info(f"[MusicPerception] Analyzing {duration:.1f}s composition")
        
        # Extract features across time
        frame_duration = 0.5  # 500ms frames for macro analysis
        frame_samples = max(1, int(frame_duration * sample_rate))
        num_frames = max(1, len(audio) // frame_samples)
        
        all_chroma = []
        all_energy = []
        all_centroids = []
        all_pitches = []
        
        for i in range(num_frames):
            start = i * frame_samples
            end = start + frame_samples
            frame = audio[start:end]
            
            if len(frame) == 0:
                continue
            
            # Energy (loudness)
            frame_float = frame.astype(np.float64)
            rms = np.sqrt(np.mean(frame_float ** 2))
            all_energy.append(rms)
            
            # Spectral centroid (brightness)
            fft = np.fft.rfft(frame_float)
            magnitude = np.abs(fft)
            frequencies = np.fft.rfftfreq(len(frame), 1.0 / sample_rate)
            
            if np.sum(magnitude) > 0:
                centroid = np.sum(frequencies * magnitude) / np.sum(magnitude)
                all_centroids.append(centroid)
            
            # Pitch estimation via autocorrelation
            autocorr = np.correlate(frame_float, frame_float, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            if len(autocorr) > 10:
                peak_idx = np.argmax(autocorr[10:]) + 10
                if peak_idx > 0 and peak_idx < len(autocorr):
                    f0 = sample_rate / peak_idx
                    if 50 < f0 < 2000:
                        midi_note = 12 * np.log2(f0 / 440.0) + 69
                        all_pitches.append(midi_note)
            
            # Chroma (pitch class distribution) for key detection
            chroma = np.zeros(12)
            peak_bins = {}
            for freq_idx in range(1, len(magnitude) - 1):
                freq = frequencies[freq_idx]
                if 50 < freq < 4000 and magnitude[freq_idx] > magnitude[freq_idx-1] and magnitude[freq_idx] > magnitude[freq_idx+1]:
                    peak_bins[freq_idx] = magnitude[freq_idx]

            for freq_idx, mag in peak_bins.items():
                freq = frequencies[freq_idx]
                pitch_class = int(round(12 * np.log2(freq / 440.0) + 69)) % 12
                log_mag = np.log1p(mag)
                weight = 1.0 / (1.0 + freq / 2000.0)
                chroma[pitch_class] += log_mag * weight
                for h in [2, 3]:
                    fund_freq = freq / h
                    if fund_freq >= 50:
                        fund_pc = int(round(12 * np.log2(fund_freq / 440.0) + 69)) % 12
                        chroma[fund_pc] += log_mag * weight * 0.25 / h

            if np.sum(chroma) > 0:
                chroma /= np.sum(chroma)
            all_chroma.append(chroma)
        
        # Store energy envelope
        features.energy_envelope = all_energy
        
        # Detect key from chroma (weighted by energy for better accuracy)
        if all_chroma:
            weights = np.array(all_energy[:len(all_chroma)]) if len(all_energy) >= len(all_chroma) else None
            if weights is not None and np.sum(weights) > 0:
                weights = weights / np.sum(weights)
                avg_chroma = np.average(all_chroma, axis=0, weights=weights)
            else:
                avg_chroma = np.mean(all_chroma, axis=0)
            features.key, features.key_confidence = self._detect_key(avg_chroma)
        if not np.isfinite(features.key_confidence):
            features.key_confidence = 0.0
        
        # Override with librosa results if available (more accurate)
        if librosa_features:
            if librosa_features.get("key"):
                features.key = librosa_features["key"]
            lib_conf = librosa_features.get("key_confidence")
            if lib_conf is not None and np.isfinite(lib_conf):
                features.key_confidence = lib_conf
            if librosa_features.get("tempo"):
                features.tempo_bpm = librosa_features["tempo"]
            lib_stab = librosa_features.get("tempo_stability")
            if lib_stab is not None and np.isfinite(lib_stab):
                features.tempo_stability = lib_stab
        
        # Estimate tempo from energy variations
        if len(all_energy) > 4:
            features.tempo_bpm, features.tempo_stability = self._estimate_tempo(all_energy, frame_duration)
        
        # Detect structure (sections)
        features.sections_detected, features.section_boundaries = self._detect_sections(all_energy, all_centroids)
        
        # Analyze melodic features
        if all_pitches:
            features.melodic_range = (min(all_pitches), max(all_pitches))
            features.melodic_contour = self._compute_contour(all_pitches)
            features.repeated_phrases = self._count_repeated_phrases(all_pitches)
        
        # Dynamic analysis
        if all_energy:
            e_valid = [e for e in all_energy if np.isfinite(e)]
            features.dynamic_range = max(e_valid) - min(e_valid) if e_valid else 0
            max_energy_idx = np.argmax(e_valid) if e_valid else 0
            features.climax_moment = max_energy_idx / len(e_valid) if len(e_valid) > 1 else 0.5
        
        # Quality evaluation
        features.coherence_score = self._evaluate_coherence(features, expected_form)
        features.development_score = self._evaluate_development(features, duration)
        features.resolution_score = self._evaluate_resolution(all_energy, features.key)
        features.structure_score = self._evaluate_structure(features, expected_form)
        
        # Store analysis tokens (only frames where centroid was computed)
        min_len = min(len(all_centroids), len(all_energy))
        features.spectral_tokens = [{"centroid": all_centroids[i], "energy": all_energy[i]} for i in range(min_len)]
        features.chroma_tokens = all_chroma
        
        # Log findings
        logger.info(f"[MusicPerception] Key: {features.key} ({features.key_confidence:.2f})")
        logger.info(f"[MusicPerception] Tempo: {features.tempo_bpm:.0f} BPM (stability {features.tempo_stability:.2f})")
        logger.info(f"[MusicPerception] Sections detected: {features.sections_detected}")
        logger.info(f"[MusicPerception] Coherence: {features.coherence_score:.2f}, Development: {features.development_score:.2f}")
        
        return features
    
    def _librosa_analysis(self, audio: np.ndarray, sr: int) -> Optional[dict]:
        """Use librosa for enhanced audio analysis when available."""
        if not _LIBROSA_AVAILABLE:
            return None
        try:
            import librosa
            
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio_float = audio.astype(np.float32)
            
            results = {}
            
            try:
                tempo, _ = librosa.feature.rhythm.tempo(y=audio_float, sr=sr)
                results["tempo"] = float(tempo[0]) if len(tempo) > 0 else 120.0
                results["tempo_stability"] = 0.8
            except Exception as e:
                logger.debug(f"Tempo detection failed: {e}")
            
            try:
                chroma = librosa.feature.chroma_cqt(y=audio_float, sr=sr)
                chroma_avg = chroma.mean(axis=1)
                major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.08])
                minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
                key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                
                best_key = "C major"
                best_conf = 0.0
                chroma_norm = chroma_avg - np.mean(chroma_avg)
                chroma_std = np.sqrt(np.sum(chroma_norm**2))
                
                for shift in range(12):
                    chroma_rotated = np.roll(chroma_norm, -shift)
                    for profile, mode in [(major_profile, "major"), (minor_profile, "minor")]:
                        prof_mean = profile - np.mean(profile)
                        prof_std = np.sqrt(np.sum(prof_mean**2))
                        if prof_std > 0 and chroma_std > 0:
                            corr = np.sum(prof_mean * chroma_rotated) / (prof_std * chroma_std)
                        else:
                            corr = 0.0
                        if corr > best_conf:
                            best_conf = corr
                            best_key = f"{key_names[shift]} {mode}"
                
                results["key"] = best_key
                results["key_confidence"] = float(abs(best_conf))
            except Exception as e:
                logger.debug(f"Key detection failed: {e}")
            
            return results if results else None
            
        except Exception as e:
            logger.debug(f"[MusicPerception] Librosa analysis failed: {e}")
            return None

    def _detect_key(self, chroma: np.ndarray) -> tuple:
        """Detect key from chroma profile."""
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.08])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        chroma_norm = chroma - np.mean(chroma)
        chroma_std = np.sqrt(np.sum(chroma_norm**2))
        
        best_key = "C major"
        best_correlation = -2.0
        
        for shift in range(12):
            chroma_rotated = np.roll(chroma_norm, -shift)
            for profile, mode in [(major_profile, "major"), (minor_profile, "minor")]:
                prof_mean = profile - np.mean(profile)
                prof_std = np.sqrt(np.sum(prof_mean**2))
                if prof_std > 0 and chroma_std > 0:
                    corr = np.sum(prof_mean * chroma_rotated) / (prof_std * chroma_std)
                else:
                    corr = 0.0
                if corr > best_correlation:
                    best_correlation = corr
                    best_key = f"{note_names[shift]} {mode}"
        
        return best_key, abs(best_correlation)
    
    def _estimate_tempo(self, energy: list, frame_duration: float) -> tuple:
        """Estimate tempo from energy envelope."""
        if len(energy) < 4:
            return 120.0, 0.0
        
        # Find peaks in energy (beats)
        energy_arr = np.array(energy)
        
        # Autocorrelation for rhythmic periodicity
        autocorr = np.correlate(energy_arr - np.mean(energy_arr), energy_arr - np.mean(energy_arr), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find peaks (beat intervals)
        # Assume tempo between 60-180 BPM
        min_beat_frames = max(1, round(60 / 180 / frame_duration))
        max_beat_frames = max(2, round(60 / 60 / frame_duration))
        
        # Find first significant peak in range
        if len(autocorr) > min_beat_frames + 2:
            search_range = autocorr[min_beat_frames:min_beat_frames + max_beat_frames]
            if len(search_range) > 0:
                peak_idx = np.argmax(search_range) + min_beat_frames
                beat_frames = peak_idx
                beat_duration = beat_frames * frame_duration
                bpm = 60.0 / beat_duration if beat_duration > 0 else 120.0
                
                # Stability = how clear the peak is
                stability = search_range.max() / (autocorr[0] + 1e-10)
                stability = float(stability) if np.isfinite(stability) else 0.0
                
                return min(200, max(60, bpm)), stability
        
        return 120.0, 0.0
    
    def _detect_sections(self, energy: list, centroids: list) -> tuple:
        """Detect section boundaries from energy and spectral changes."""
        if len(energy) < 4:
            return 1, []

        e_arr = np.array(energy)
        c_arr = np.array(centroids) if len(centroids) >= len(energy) else np.array(
            centroids + [centroids[-1]] * (len(energy) - len(centroids))) if centroids else np.zeros(len(energy))

        # Pass 1: Novelty function with raised-cosine kernel
        kernel_len = max(2, len(energy) // 12)
        kernel = np.zeros(2 * kernel_len)
        for k in range(kernel_len):
            kernel[k] = -np.cos(np.pi * k / kernel_len) * 0.5 + 0.5
            kernel[kernel_len + k] = np.cos(np.pi * k / kernel_len) * 0.5 - 0.5
        kernel /= np.sum(np.abs(kernel)) + 1e-10

        e_norm = np.log1p(e_arr)
        c_norm = c_arr / (np.mean(c_arr) + 1e-10) if np.mean(c_arr) > 0 else c_arr
        combined = np.column_stack([e_norm, c_norm * 0.5])
        novelty = np.zeros(len(energy))
        for i in range(kernel_len, len(energy) - kernel_len):
            left = combined[max(0, i - kernel_len):i].flatten()
            right = combined[i:min(len(energy), i + kernel_len)].flatten()
            n = min(len(left), len(right))
            if n > 0:
                novelty[i] = np.sum((right[:n] - left[:n]) ** 2)

        if np.max(novelty) > 0:
            novelty /= np.max(novelty)
        nov_median = np.median(novelty)
        nov_mad = np.median(np.abs(novelty - nov_median))
        nov_threshold = max(nov_median + 1.5 * nov_mad, 0.12)

        boundaries = set()
        for i in range(len(novelty)):
            if novelty[i] > nov_threshold:
                boundaries.add(i)

        # Pass 2: Sustained level shifts
        window = max(2, len(energy) // 8)
        e_mean = np.mean(energy) + 1e-10
        for i in range(window, len(energy) - window):
            before_e = np.mean(energy[max(0, i - window):i])
            after_e = np.mean(energy[i:min(len(energy), i + window)])
            shift_e = abs(after_e - before_e) / e_mean
            before_c = np.mean(c_arr[max(0, i - window):i]) if len(c_arr) >= i else 0
            after_c = np.mean(c_arr[i:min(len(c_arr), i + window)]) if len(c_arr) >= i + window else 0
            c_mean_val = np.mean(c_arr) + 1e-10
            shift_c = abs(after_c - before_c) / c_mean_val if c_mean_val > 0 else 0
            if shift_e > 0.3 or shift_c > 0.3:
                boundaries.add(i)

        # Merge nearby boundaries
        min_gap = max(2, len(energy) // 16)
        merged_boundaries = []
        for b in sorted(boundaries):
            if not merged_boundaries or b - merged_boundaries[-1] > min_gap:
                merged_boundaries.append(b)

        sections = len(merged_boundaries) + 1
        return sections, merged_boundaries
    
    def _compute_contour(self, pitches: list) -> list:
        """Compute melodic contour (up/down/flat)."""
        if len(pitches) < 2:
            return []
        
        contour = []
        for i in range(1, len(pitches)):
            diff = pitches[i] - pitches[i-1]
            if abs(diff) < 0.5:
                contour.append(0)  # Flat
            elif diff > 0:
                contour.append(1)  # Up
            else:
                contour.append(-1)  # Down
        
        return contour
    
    def _count_repeated_phrases(self, pitches: list) -> int:
        """Count repeated melodic phrases (theme development)."""
        if len(pitches) < 6:
            return 0
        
        quantized = [int(round(p)) for p in pitches]
        
        phrases_seen = Counter()
        for phrase_len in (3, 4, 5, 6):
            if len(quantized) < phrase_len:
                continue
            for i in range(len(quantized) - phrase_len):
                phrase = tuple(quantized[i:i+phrase_len])
                phrases_seen[phrase] += 1
        
        repeated = sum(1 for count in phrases_seen.values() if count > 1)
        
        return repeated
    
    def _evaluate_coherence(self, features: MusicalFeatures, expected_form: str) -> float:
        """Evaluate if composition sounds unified and coherent."""
        score = 0.0

        # Key confidence is the primary coherence indicator
        score += features.key_confidence * 0.25

        # Tempo stability contributes to perceived coherence
        score += min(features.tempo_stability, 1.0) * 0.12

        # Chroma stability: consistent pitch class distribution = more coherent
        if features.chroma_tokens and len(features.chroma_tokens) > 2:
            chroma_arr = np.array(features.chroma_tokens)
            chroma_var = np.mean(np.var(chroma_arr, axis=0))
            chroma_stability = 1.0 / (1.0 + chroma_var * 10)
            score += chroma_stability * 0.10

        # Melodic range: moderate range is more coherent
        if features.melodic_range and features.melodic_range[1] > features.melodic_range[0]:
            range_semitones = features.melodic_range[1] - features.melodic_range[0]
            if 6 <= range_semitones <= 24:
                score += 0.15
            elif 3 <= range_semitones <= 30:
                score += 0.10
            elif range_semitones > 0:
                score += 0.05

        # Repeated phrases indicate coherence
        score += min(features.repeated_phrases * 0.05, 0.10)

        # Smooth melodic contour indicates coherence
        if features.melodic_contour:
            contour_arr = np.array(features.melodic_contour)
            smoothness = 1.0 - min(np.mean(np.abs(contour_arr)), 1.0)
            score += smoothness * 0.08

        # Bonus for having reasonable number of sections
        if 2 <= features.sections_detected <= 6:
            score += 0.10

        # Energy consistency: moderate variation is coherent
        if features.energy_envelope and len(features.energy_envelope) > 4:
            e_arr = np.array(features.energy_envelope)
            e_cv = np.std(e_arr) / (np.mean(e_arr) + 1e-10)
            if 0.2 < e_cv < 1.5:
                score += 0.08
            elif e_cv > 0:
                score += 0.03

        return min(1.0, score)
    
    def _evaluate_development(self, features: MusicalFeatures, duration: float) -> float:
        """Evaluate thematic development and growth."""
        score = 0.0

        if features.dynamic_range > 0:
            norm_range = min(features.dynamic_range / (np.mean(features.energy_envelope) + 1e-6), 3.0) if features.energy_envelope else 0
            score += min(norm_range * 0.15, 0.25)
        else:
            score += 0.05

        if features.sections_detected >= 4:
            score += 0.25
        elif features.sections_detected >= 3:
            score += 0.20
        elif features.sections_detected >= 2:
            score += 0.12
        else:
            score += 0.05

        if 0.45 <= features.climax_moment <= 0.7:
            score += 0.2
        elif 0.3 <= features.climax_moment <= 0.8:
            score += 0.1
        else:
            score += 0.02

        if features.repeated_phrases >= 3:
            score += 0.15
        elif features.repeated_phrases >= 1:
            score += 0.08

        # Energy trajectory bonus: ascending then descending = good development
        if features.energy_envelope and len(features.energy_envelope) > 6:
            e_arr = np.array(features.energy_envelope)
            mid = len(e_arr) // 2
            first_half = np.mean(e_arr[:mid])
            peak_idx = np.argmax(e_arr)
            if peak_idx >= mid * 0.7 and first_half > 0:
                rise_ratio = (np.max(e_arr[:mid+1]) - e_arr[0]) / (first_half + 1e-10)
                score += min(rise_ratio * 0.05, 0.08)

        return min(1.0, score)
    
    def _evaluate_resolution(self, energy: list, key: str) -> float:
        """Evaluate if composition resolves properly."""
        if len(energy) < 4:
            return 0.5

        end_energy = np.mean(energy[-4:])
        overall_energy = np.mean(energy)

        energy_ratio = end_energy / (overall_energy + 1e-6)

        if energy_ratio < 0.5:
            score = 0.9
        elif energy_ratio < 0.7:
            score = 0.8
        elif energy_ratio < 0.9:
            score = 0.6
        elif energy_ratio < 1.1:
            score = 0.4
        else:
            score = 0.2

        peak_idx = np.argmax(energy)
        peak_position = peak_idx / len(energy)
        if 0.4 <= peak_position <= 0.75:
            score = min(1.0, score + 0.1)

        tail = energy[-min(4, len(energy)):]
        if all(tail[i] >= tail[i+1] for i in range(len(tail)-1)):
            score = min(1.0, score + 0.05)

        return min(1.0, score)
    
    def _evaluate_structure(self, features: MusicalFeatures, expected_form: str) -> float:
        """Evaluate structural coherence."""
        score = 0.0

        expected_sections = {
            "binary": 2, "ternary": 3, "sonata": 3, "rondo": 5,
            "fugue": 2, "theme_variations": 4, "prelude": 1, "nocturne": 2,
            "aba": 3, "song_form": 2, "strophic": 2, "through_composed": 1,
            "minuet": 3, "scherzo": 3, "march": 3, "waltz": 2,
        }

        expected = expected_sections.get(expected_form, 3)

        diff = abs(features.sections_detected - expected)
        if diff == 0:
            score += 0.45
        elif diff == 1:
            score += 0.35
        elif diff == 2:
            score += 0.20
        else:
            score += 0.05

        # Coherence contributes to structural quality
        score += features.coherence_score * 0.35

        # Resolution contributes to structural completeness
        score += features.resolution_score * 0.20

        return min(1.0, score)
    
    def generate_feedback(
        self,
        features: MusicalFeatures,
        expected_form: str,
        expected_key: str,
        expected_tempo: int,
    ) -> CompositionFeedback:
        """Generate feedback for improving the composition."""
        feedback = CompositionFeedback()
        
        # Overall score
        weights = {"coherence": 0.3, "development": 0.25, "resolution": 0.2, "structure": 0.25}
        feedback.overall_score = (
            features.coherence_score * weights["coherence"] +
            features.development_score * weights["development"] +
            features.resolution_score * weights["resolution"] +
            features.structure_score * weights["structure"]
        )
        
        # Identify strengths
        if features.coherence_score > 0.5:
            feedback.strengths.append("Coherent harmonic structure")
        if features.coherence_score > 0.75:
            feedback.strengths.append("Strong tonal center")
        if features.development_score > 0.5:
            feedback.strengths.append("Good thematic development")
        if features.development_score > 0.75:
            feedback.strengths.append("Excellent dynamic contrast")
        if features.resolution_score > 0.5:
            feedback.strengths.append("Satisfying resolution")
        if features.resolution_score > 0.75:
            feedback.strengths.append("Strong musical resolution")
        if features.structure_score > 0.5:
            feedback.strengths.append(f"Clear {expected_form} structure")
        if features.key_confidence > 0.6:
            feedback.strengths.append(f"Clear key ({features.key})")
        if features.tempo_stability > 0.4:
            feedback.strengths.append(f"Steady tempo ({features.tempo_bpm:.0f} BPM)")
        if features.repeated_phrases > 2:
            feedback.strengths.append("Good melodic repetition")
        if features.dynamic_range > 0 and features.energy_envelope:
            norm_dr = features.dynamic_range / (np.mean(features.energy_envelope) + 1e-6)
            if 0.3 < norm_dr < 2.5:
                feedback.strengths.append("Appropriate dynamic contrast")
        if features.melodic_range and features.melodic_range[1] > features.melodic_range[0]:
            r = features.melodic_range[1] - features.melodic_range[0]
            if 5 <= r <= 24:
                feedback.strengths.append("Good melodic range")

        # Identify weaknesses
        if features.coherence_score < self.MIN_COHERENCE_THRESHOLD:
            feedback.weaknesses.append("Incoherent - sounds disjointed")
            feedback.issues.append("coherence_low")
            feedback.suggestions.append("Add repeating melodic phrases for unity")
        elif features.coherence_score < 0.5:
            feedback.weaknesses.append("Weak harmonic coherence")
            feedback.issues.append("coherence_low")
            feedback.suggestions.append("Strengthen tonal center with clearer key")

        if features.structure_score < self.MIN_STRUCTURE_THRESHOLD:
            feedback.weaknesses.append(f"Structure unclear (expected {expected_form})")
            feedback.issues.append("structure_weak")
            feedback.suggestions.append(f"Clarify {expected_form} form with distinct sections")
        elif features.structure_score < 0.45:
            feedback.weaknesses.append(f"Structure could be clearer for {expected_form}")
            feedback.suggestions.append(f"Emphasize section contrasts for {expected_form} form")

        if features.development_score < self.MIN_DEVELOPMENT_THRESHOLD:
            feedback.weaknesses.append("Lacks thematic development")
            feedback.issues.append("development_weak")
            feedback.suggestions.append("Add contrast: develop theme in different keys/loudness")
        elif features.development_score < 0.45:
            feedback.weaknesses.append("Limited thematic development")
            feedback.suggestions.append("Add more dynamic variation and climax")

        if features.resolution_score < 0.5:
            feedback.weaknesses.append("Abrupt or unresolved ending")
            feedback.issues.append("resolution_weak")
            feedback.suggestions.append("End more gently - reduce energy at conclusion")

        # Key mismatch
        detected_key_norm = features.key.lower().replace(" ", "_")
        expected_key_norm = expected_key.lower().replace(" ", "_")
        detected_root = features.key.lower().split()[0] if features.key and features.key != "unknown" else ""
        expected_root = expected_key.lower().split("_")[0] if expected_key else ""
        detected_mode = features.key.lower().split()[-1] if features.key and features.key != "unknown" else ""
        expected_mode = expected_key.lower().split("_")[-1] if expected_key else ""

        key_match = (detected_key_norm == expected_key_norm or
                     (detected_root == expected_root and detected_mode == expected_mode))
        if not key_match and features.key_confidence > 0.5 and features.key != "unknown":
            feedback.weaknesses.append(f"Key mismatch: expected {expected_key}, got {features.key}")
            feedback.issues.append("key_mismatch")
            feedback.suggestions.append(f"Adjust composition to emphasize {expected_key} tonality")

        # Tempo mismatch
        tempo_diff = abs(features.tempo_bpm - expected_tempo)
        if tempo_diff > 30 and features.tempo_stability > 0.3:
            feedback.weaknesses.append(f"Tempo mismatch: expected {expected_tempo}, got {features.tempo_bpm:.0f}")
            feedback.issues.append("tempo_mismatch")
            feedback.suggestions.append(f"Adjust tempo closer to {expected_tempo} BPM")
        
        # Should regenerate?
        feedback.should_regenerate = (
            features.coherence_score < self.MIN_COHERENCE_THRESHOLD or
            features.structure_score < self.MIN_STRUCTURE_THRESHOLD or
            features.development_score < self.MIN_DEVELOPMENT_THRESHOLD
        )
        
        # Generate prompt hints for regeneration
        if feedback.should_regenerate:
            if "coherence_low" in feedback.issues:
                feedback.retry_prompt_hints.append("clear thematic unity with repeating motifs")
            if "structure_weak" in feedback.issues:
                feedback.retry_prompt_hints.append(f"distinct {expected_form} sections with clear transitions")
            if "development_weak" in feedback.issues:
                feedback.retry_prompt_hints.append("thematic development with increasing intensity")
            if "resolution_weak" in feedback.issues:
                feedback.retry_prompt_hints.append("gradual resolution with decreasing energy at end")
        
        return feedback


# Singleton
_perception_engine: MusicPerceptionEngine | None = None
_perception_lock = threading.Lock()


def get_music_perception() -> MusicPerceptionEngine:
    """Get singleton perception engine."""
    global _perception_engine
    with _perception_lock:
        if _perception_engine is None:
            _perception_engine = MusicPerceptionEngine()
        return _perception_engine
