"""Audio-derived tokens on the Bloch sphere via the circle of fifths.

Pipeline (all honest signal processing — no ground-truth leakage):

  1. SYNTHESIZE the Giant Steps changes (Coltrane) as raw audio. The form
     cycles through three tonal centers a major third apart (B, G, Eb) —
     the 'Coltrane matrix'. This gives a falsifiable prediction: token
     geometry should recover three key clusters ~120 degrees apart on the
     circle of fifths.
  2. EXTRACT chroma per frame with a windowed FFT, fold magnitudes into 12
     pitch classes, take the dominant pitch class as the frame's symbol
     (alphabet '0'..'B', base-12).
  3. TOKENIZE the symbol stream with the same greedy BPE used for text —
     merged tokens are recurring pitch-class motifs.
  4. PLACE each token on the Bloch sphere: each symbol is a *note* whose
     longitude is its position on the circle of fifths
     (phi_note = 2*pi * (7*pc mod 12) / 12); a token is the CHORD of its
     notes, placed at their circular mean (Chew's 'center of effect').
     Latitude = surprisal, as for text.
  5. TEST: (a) geodesic distance vs pitch-content similarity,
     (b) within-key vs across-key separation against ground-truth tonal
     centers (labels used ONLY for evaluation, never for placement),
     with a permutation test.

Run: python3 audio_tokenizer.py
Outputs: data/giant_steps.wav, docs/images/fifths_placement/audio_tokens.html
"""

import math
import os
import wave
from collections import Counter

import numpy as np
from scipy.stats import spearmanr

from harmonic_placement import (byte_pair_encoding, bloch_xyz,
                                geodesic_matrix, upper_triangle,
                                content_similarity_matrix)

TWO_PI = 2 * math.pi
SR = 22050
FRAME = 2048
HOP = 1024
SYMBOLS = '0123456789AB'  # pitch classes C..B in base-12

# ---------------------------------------------------------------------------
# 1. Synthesis: Giant Steps changes
# ---------------------------------------------------------------------------

NOTE = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'Eb': 3, 'E': 4, 'F': 5, 'F#': 6,
        'Gb': 6, 'G': 7, 'Ab': 8, 'A': 9, 'Bb': 10, 'B': 11}

def chord_pcs(root, quality):
    r = NOTE[root]
    intervals = {'maj7': (0, 4, 7, 11), '7': (0, 4, 7, 10), 'm7': (0, 3, 7, 10)}
    return [(r + i) % 12 for i in intervals[quality]]

# (chord, quality, tonal center) — centers follow the standard analysis:
# each dominant resolves down a fifth; ii-V pairs belong to their target key.
GIANT_STEPS = [
    ('B', 'maj7', 'B'), ('D', '7', 'G'),
    ('G', 'maj7', 'G'), ('Bb', '7', 'Eb'),
    ('Eb', 'maj7', 'Eb'), ('Eb', 'maj7', 'Eb'),
    ('A', 'm7', 'G'), ('D', '7', 'G'),
    ('G', 'maj7', 'G'), ('Bb', '7', 'Eb'),
    ('Eb', 'maj7', 'Eb'), ('F#', '7', 'B'),
    ('B', 'maj7', 'B'), ('B', 'maj7', 'B'),
    ('F', 'm7', 'Eb'), ('Bb', '7', 'Eb'),
    ('Eb', 'maj7', 'Eb'), ('Eb', 'maj7', 'Eb'),
    ('A', 'm7', 'G'), ('D', '7', 'G'),
    ('G', 'maj7', 'G'), ('G', 'maj7', 'G'),
    ('C#', 'm7', 'B'), ('F#', '7', 'B'),
    ('B', 'maj7', 'B'), ('B', 'maj7', 'B'),
    ('F', 'm7', 'Eb'), ('Bb', '7', 'Eb'),
    ('Eb', 'maj7', 'Eb'), ('Eb', 'maj7', 'Eb'),
    ('C#', 'm7', 'B'), ('F#', '7', 'B'),
]

def pc_to_freq(pc, octave=4):
    midi = 12 * (octave + 1) + pc
    return 440.0 * 2 ** ((midi - 69) / 12)

def synthesize(choruses=4, chord_dur=0.4, seed=7):
    """Render the changes as summed sines with light voicing variation."""
    rng = np.random.default_rng(seed)
    n = int(SR * chord_dur)
    t = np.arange(n) / SR
    env = np.minimum(1, 10 * t) * np.minimum(1, 10 * (chord_dur - t))
    audio, labels = [], []
    for _ in range(choruses):
        for root, quality, center in GIANT_STEPS:
            sig = np.zeros(n)
            for pc in chord_pcs(root, quality):
                octv = int(rng.choice([3, 4, 4, 5]))
                f = pc_to_freq(pc, octv)
                # fundamental + a couple of partials for realism
                sig += np.sin(TWO_PI * f * t)
                sig += 0.4 * np.sin(TWO_PI * 2 * f * t)
                sig += 0.15 * np.sin(TWO_PI * 3 * f * t)
            audio.append(sig * env / 8)
            labels.append(center)
    return np.concatenate(audio), labels, n

def write_wav(signal, path):
    x = np.clip(signal, -1, 1)
    pcm = (x * 32767).astype('<i2')
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())

# ---------------------------------------------------------------------------
# 2. Chroma extraction -> symbol stream
# ---------------------------------------------------------------------------

def chroma_symbols(signal, chord_len, labels):
    """Windowed FFT -> 12-bin chroma -> dominant pitch class per frame.
    Returns (symbol string, per-frame tonal-center labels)."""
    window = np.hanning(FRAME)
    freqs = np.fft.rfftfreq(FRAME, 1 / SR)
    valid = (freqs > 60) & (freqs < 5000)
    pcs_of_bin = np.zeros(len(freqs), dtype=int)
    pcs_of_bin[valid] = np.round(
        12 * np.log2(freqs[valid] / 440.0) + 69).astype(int) % 12
    symbols, frame_labels = [], []
    for start in range(0, len(signal) - FRAME, HOP):
        spec = np.abs(np.fft.rfft(signal[start:start + FRAME] * window))
        chroma = np.zeros(12)
        np.add.at(chroma, pcs_of_bin[valid], spec[valid])
        if chroma.sum() < 1e-9:
            continue
        symbols.append(SYMBOLS[int(np.argmax(chroma))])
        frame_labels.append(labels[min(start // chord_len, len(labels) - 1)])
    return ''.join(symbols), frame_labels

# ---------------------------------------------------------------------------
# 3. Placement: notes on the circle of fifths, tokens as chord centroids
# ---------------------------------------------------------------------------

def note_phi(symbol):
    """Pitch class -> longitude on the circle of fifths (x7 mod 12)."""
    pc = int(symbol, 12)
    return TWO_PI * ((7 * pc) % 12) / 12

def audio_chord_placement(token_counts):
    """Same recipe as text chord_placement, but notes are real pitches."""
    total = sum(token_counts.values())
    surprisal = {t: -math.log2(c / total) for t, c in token_counts.items()}
    max_s = max(surprisal.values())
    eps = 0.02 * math.pi
    vocab = {}
    for t in token_counts:
        angles = [note_phi(c) for c in t]
        x = sum(math.cos(a) for a in angles)
        y = sum(math.sin(a) for a in angles)
        phi = math.atan2(y, x) % TWO_PI if x * x + y * y > 1e-12 else 0.0
        theta = eps + (math.pi - 2 * eps) * (surprisal[t] / max_s)
        vocab[t] = (theta, phi)
    return vocab

def random_placement(token_counts, seed=42):
    rng = np.random.default_rng(seed)
    return {t: (math.acos(1 - 2 * rng.uniform()), TWO_PI * rng.uniform())
            for t in token_counts}

# ---------------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------------

def token_key_labels(stream, frame_labels):
    """Majority ground-truth tonal center per token (evaluation only)."""
    votes = {}
    pos = 0
    for tok in stream:
        span = frame_labels[pos:pos + len(tok)]
        pos += len(tok)
        votes.setdefault(tok, Counter()).update(span)
    return {t: v.most_common(1)[0][0] for t, v in votes.items()}

def key_separation(vocab, tokens, key_of, n_perm=2000, seed=0):
    """Across-key / within-key mean geodesic distance ratio (+ permutation p).
    Ratio > 1 means same-key tokens sit closer together than cross-key."""
    D = geodesic_matrix(vocab, tokens)
    labels = np.array([key_of[t] for t in tokens])
    iu = np.triu_indices(len(tokens), k=1)
    same = labels[iu[0]] == labels[iu[1]]
    d = D[iu]
    ratio = d[~same].mean() / d[same].mean()
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(labels)
        s = perm[iu[0]] == perm[iu[1]]
        if d[~s].mean() / d[s].mean() >= ratio:
            count += 1
    return ratio, (count + 1) / (n_perm + 1)

def key_cluster_angles(vocab, tokens, key_of):
    """Circular mean longitude of each key's tokens."""
    out = {}
    for key in sorted(set(key_of.values())):
        phis = [vocab[t][1] for t in tokens if key_of[t] == key]
        x = np.mean(np.cos(phis))
        y = np.mean(np.sin(phis))
        out[key] = math.degrees(math.atan2(y, x) % TWO_PI)
    return out

# ---------------------------------------------------------------------------
# 5. Experiment
# ---------------------------------------------------------------------------

def run_experiment(num_merges=80, viz=True):
    os.makedirs('data', exist_ok=True)
    signal, chord_labels, chord_len = synthesize()
    write_wav(signal, 'data/giant_steps.wav')
    print(f'Synthesized {len(signal)/SR:.1f}s of Giant Steps changes '
          f'-> data/giant_steps.wav')

    symbols, frame_labels = chroma_symbols(signal, chord_len, chord_labels)
    print(f'Chroma frames: {len(symbols)}   '
          f'(pitch-class alphabet, e.g. {symbols[:40]}...)')

    stream, merges = byte_pair_encoding(symbols, num_merges)
    counts = Counter(stream)
    tokens = sorted(counts)
    print(f'BPE merges: {num_merges}   token stream: {len(stream)}   '
          f'vocab: {len(tokens)}')

    placed = audio_chord_placement(counts)
    rand = random_placement(counts)
    key_of = token_key_labels(stream, frame_labels)

    # (a) geometry vs pitch content
    content = upper_triangle(content_similarity_matrix(tokens))
    print('\nGeodesic distance vs pitch-class content similarity (Spearman):')
    for name, vocab in (('chord', placed), ('random', rand)):
        rho, p = spearmanr(upper_triangle(geodesic_matrix(vocab, tokens)),
                           content)
        print(f'  {name:<8} rho = {rho:+.3f}  (p = {p:.1e})')

    # (b) do the three Coltrane tonal centers separate?
    print('\nTonal-center separation (across-key / within-key distance, '
          'permutation test):')
    for name, vocab in (('chord', placed), ('random', rand)):
        ratio, p = key_separation(vocab, tokens, key_of)
        print(f'  {name:<8} ratio = {ratio:.3f}  (p = {p:.4f})')

    angles = key_cluster_angles(placed, tokens, key_of)
    print('\nKey-cluster mean longitudes (prediction: ~120 degrees apart):')
    keys = list(angles)
    for k in keys:
        print(f'  {k:<3} {angles[k]:7.1f} deg')
    for i in range(len(keys)):
        a, b = keys[i], keys[(i + 1) % len(keys)]
        sep = abs(angles[a] - angles[b]) % 360
        print(f'  |{a} - {b}| = {min(sep, 360 - sep):.1f} deg')

    if viz:
        visualize(placed, rand, counts, key_of,
                  'docs/images/fifths_placement/audio_tokens.html')


def visualize(placed, rand, counts, key_of, out_path):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tokens = sorted(counts)
    palette = {'B': '#e45756', 'G': '#4c78a8', 'Eb': '#f2a900'}
    colors = [palette[key_of[t]] for t in tokens]

    fig = make_subplots(
        rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=['Audio Tokens: Chord-Centroid Placement',
                        'Random Placement (null model)'])
    u, v = np.mgrid[0:TWO_PI:40j, 0:np.pi:20j]
    wire = dict(x=np.cos(u) * np.sin(v), y=np.sin(u) * np.sin(v),
                z=np.cos(v), opacity=0.12, showscale=False,
                colorscale='Greys')
    for col, vocab in ((1, placed), (2, rand)):
        pts = np.array([bloch_xyz(*vocab[t]) for t in tokens])
        fig.add_trace(go.Surface(**wire), row=1, col=col)
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode='markers',
            marker=dict(size=5, color=colors),
            text=[f'{t!r} key={key_of[t]} n={counts[t]}' for t in tokens],
            hovertemplate='%{text}<extra></extra>', showlegend=False),
            row=1, col=col)
    # legend proxies
    for key, c in palette.items():
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None], mode='markers',
            marker=dict(size=6, color=c), name=f'tonal center {key}'),
            row=1, col=1)
    fig.update_layout(
        title='Giant Steps, tokenized: three Coltrane tonal centers on the '
              'Bloch sphere', height=650, width=1300)
    fig.write_html(out_path)
    print(f'\nVisualization written to {out_path}')


if __name__ == '__main__':
    run_experiment()
