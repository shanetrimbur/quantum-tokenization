"""Real-recording test: recover the circle of fifths from the Well-Tempered
Clavier, and detect modulation within pieces.

Corpus: Kimiko Ishizaka's public-domain recording of Bach's WTC Book 1
(archive.org: bach-well-tempered-clavier-book-1). The 12 major-key preludes
traverse every major key, giving a strong falsifiable prediction:

    The estimated tonal center of each prelude — computed from raw audio
    with the chroma -> BPE -> chord-centroid pipeline, no score information —
    should lay the 12 preludes out as the circle of fifths.

A constant rotation between estimated and true angles is expected and allowed:
the circular mean of a diatonic set on the fifths circle sits at the center of
its 6-fifth chain (2 fifths sharp of the tonic — D for C major), so we fit one
global offset and score the residuals.

Also: sliding-window tonal drift gives a modulation timeline per piece.

Run: python3 real_audio_analysis.py   (expects data/wtc/prelude_*.wav)
"""

import glob
import math
import os
from collections import Counter

import numpy as np
from scipy.io import wavfile

from harmonic_placement import byte_pair_encoding
from audio_tokenizer import (SR, FRAME, HOP, SYMBOLS, NOTE,
                             audio_chord_placement, note_phi)

TWO_PI = 2 * math.pi

# ---------------------------------------------------------------------------
# Audio -> pitch-class symbol stream (real recordings)
# ---------------------------------------------------------------------------

def load_wav(path):
    sr, x = wavfile.read(path)
    x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x /= max(np.abs(x).max(), 1e-9)
    if sr != SR:
        from scipy.signal import resample_poly
        g = math.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g)
    return x

def wav_to_symbols(path, gate_db=-45):
    """Windowed FFT -> 12-bin chroma -> dominant pitch class per frame.
    Frames quieter than gate_db (relative to peak frame) are skipped."""
    x = load_wav(path)
    window = np.hanning(FRAME)
    freqs = np.fft.rfftfreq(FRAME, 1 / SR)
    valid = (freqs > 55) & (freqs < 4200)
    pcs_of_bin = np.zeros(len(freqs), dtype=int)
    pcs_of_bin[valid] = np.round(
        12 * np.log2(freqs[valid] / 440.0) + 69).astype(int) % 12
    starts = range(0, len(x) - FRAME, HOP)
    energies = []
    chromas = []
    for s in starts:
        frame = x[s:s + FRAME] * window
        spec = np.abs(np.fft.rfft(frame))
        energies.append(float(spec[valid].sum()))
        chroma = np.zeros(12)
        np.add.at(chroma, pcs_of_bin[valid], spec[valid])
        chromas.append(chroma)
    energies = np.array(energies)
    gate = energies.max() * 10 ** (gate_db / 20)
    out = []
    for e, chroma in zip(energies, chromas):
        if e < gate or chroma.sum() < 1e-9:
            continue
        out.append(SYMBOLS[int(np.argmax(chroma))])
    return ''.join(out)

# ---------------------------------------------------------------------------
# Tonal center estimation from token geometry
# ---------------------------------------------------------------------------

def analyze_piece(symbols, num_merges=80):
    """BPE + chord placement; returns (vocab, counts, est_angle, coherence).
    est_angle = count-weighted circular mean longitude of the token cloud;
    coherence = resultant length in [0, 1] (1 = tonally focused)."""
    stream, _ = byte_pair_encoding(symbols, num_merges)
    counts = Counter(stream)
    vocab = audio_chord_placement(counts)
    x = sum(c * math.cos(vocab[t][1]) for t, c in counts.items())
    y = sum(c * math.sin(vocab[t][1]) for t, c in counts.items())
    n = sum(counts.values())
    return vocab, counts, math.atan2(y, x) % TWO_PI, math.hypot(x, y) / n

def key_angle(key_name):
    """True longitude of a key's tonic on the circle of fifths."""
    return TWO_PI * ((7 * NOTE[key_name]) % 12) / 12

def circ_diff(a, b):
    d = (a - b) % TWO_PI
    return d - TWO_PI if d > math.pi else d

def fit_offset(est, true):
    """Best single rotation aligning estimated to true angles."""
    x = sum(math.cos(e - t) for e, t in zip(est, true))
    y = sum(math.sin(e - t) for e, t in zip(est, true))
    return math.atan2(y, x)

def modulation_timeline(symbols, window_frames=200, hop_frames=25):
    """Sliding circular mean of note angles -> tonal drift over time.
    Returns (times_sec, angles_deg, coherence)."""
    angles = np.array([note_phi(c) for c in symbols])
    times, drift, coh = [], [], []
    for s in range(0, len(angles) - window_frames, hop_frames):
        w = angles[s:s + window_frames]
        x, y = np.cos(w).mean(), np.sin(w).mean()
        times.append((s + window_frames / 2) * HOP / SR)
        drift.append(math.degrees(math.atan2(y, x) % TWO_PI))
        coh.append(math.hypot(x, y))
    return np.array(times), np.array(drift), np.array(coh)

# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

FIFTHS_ORDER = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'Ab', 'Eb', 'Bb', 'F']

def run_experiment(wav_glob='data/wtc/prelude_*.wav', viz=True):
    paths = sorted(glob.glob(wav_glob))
    if not paths:
        raise SystemExit(f'no files match {wav_glob}')
    results = {}
    for p in paths:
        key = os.path.basename(p).replace('prelude_', '').replace('.wav', '')
        symbols = wav_to_symbols(p)
        vocab, counts, est, coh = analyze_piece(symbols)
        results[key] = dict(symbols=symbols, est=est, coherence=coh,
                            vocab=vocab, counts=counts)
        print(f'{key:<3} {len(symbols):>5} frames  vocab {len(counts):>3}  '
              f'est angle {math.degrees(est):6.1f} deg  coherence {coh:.2f}')

    keys = list(results)
    est = [results[k]['est'] for k in keys]
    true = [key_angle(k) for k in keys]
    off = fit_offset(est, true)
    print(f'\nFitted global offset: {math.degrees(off):.1f} deg '
          f'(theory predicts ~+60, i.e. 2 fifths, for diatonic sets)')

    errors, correct = [], 0
    print(f'\n{"key":<4}{"true":>7}{"est-off":>9}{"err":>7}   nearest key')
    for k, e, t in zip(keys, est, true):
        adj = (e - off) % TWO_PI
        err = math.degrees(circ_diff(adj, t))
        near = min(FIFTHS_ORDER, key=lambda q: abs(circ_diff(adj, key_angle(q))))
        hit = near == k
        correct += hit
        errors.append(abs(err))
        print(f'{k:<4}{math.degrees(t):>7.0f}{math.degrees(adj):>9.1f}'
              f'{err:>7.1f}   {near} {"OK" if hit else "MISS"}')
    print(f'\nMean |error|: {np.mean(errors):.1f} deg '
          f'(chance ~90)   nearest-key accuracy: {correct}/{len(keys)}')

    if viz:
        visualize(results, off, 'docs/images/fifths_placement/wtc_circle.html')
    return results, off


def visualize(results, off, out_path):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'polar'}, {'type': 'xy'}]],
        subplot_titles=['Estimated tonal centers (offset removed) vs '
                        'circle of fifths',
                        'Modulation timeline: Prelude in C major'])

    # polar: true positions ring + estimated
    keys = list(results)
    fig.add_trace(go.Scatterpolar(
        r=[1.0] * len(FIFTHS_ORDER),
        theta=[math.degrees(key_angle(k)) for k in FIFTHS_ORDER],
        mode='markers+text', text=FIFTHS_ORDER, textposition='top center',
        marker=dict(size=10, color='#bbbbbb'), name='true position'),
        row=1, col=1)
    fig.add_trace(go.Scatterpolar(
        r=[results[k]['coherence'] for k in keys],
        theta=[math.degrees((results[k]['est'] - off) % TWO_PI) for k in keys],
        mode='markers+text', text=keys, textposition='bottom center',
        marker=dict(size=12, color='#e45756'),
        name='estimated (radius = coherence)'),
        row=1, col=1)

    # timeline for the C major prelude
    t, drift, coh = modulation_timeline(results['C']['symbols'])
    fig.add_trace(go.Scatter(
        x=t, y=drift, mode='markers',
        marker=dict(size=5, color=coh, colorscale='Viridis',
                    colorbar=dict(title='coherence', x=1.02)),
        name='tonal drift'), row=1, col=2)
    for k in ['C', 'G', 'F']:
        yk = math.degrees((key_angle(k) + math.radians(60)) % TWO_PI)
        fig.add_trace(go.Scatter(
            x=[t[0], t[-1]], y=[yk, yk], mode='lines+text',
            text=[f'{k} region', ''], textposition='top right',
            line=dict(dash='dot', color='#999'), showlegend=False),
            row=1, col=2)
    fig.update_yaxes(title='window angle (deg on fifths circle)', row=1, col=2)
    fig.update_xaxes(title='time (s)', row=1, col=2)
    fig.update_layout(
        title='Bach, Well-Tempered Clavier Book 1 (Ishizaka, public domain): '
              'the circle of fifths recovered from raw audio',
        height=600, width=1300)
    fig.write_html(out_path)
    print(f'\nVisualization written to {out_path}')


if __name__ == '__main__':
    run_experiment()
