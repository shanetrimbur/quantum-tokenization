"""Harmonic (circle-of-fifths) token placement on the Bloch sphere.

Replaces random (theta, phi) assignment with a placement rule where geometry
encodes token content and statistics:

  phi (longitude) — harmonic content. Each byte value b gets an angle on the
      CONTINUOUS spiral of fifths: phi_char = 2*pi * frac(b * log2(3/2)).
      log2(3/2) is the width of a just perfect fifth in octaves; it is
      irrational, so repeated fifths never close the circle — the familiar
      12-tone Circle of Fifths is its rational approximation (12 fifths ~ 7
      octaves). A multi-character token's phi is the SUM of its characters'
      angles mod 2*pi, i.e. the circle-group operation: BPE merge = interval
      addition = modular arithmetic on the circle. This makes phi an exact
      homomorphism from token concatenation to U(1):
          phi(AB) = phi(A) + phi(B)  (mod 2*pi)
      Consequence: phi encodes the token's character multiset (anagrams
      collide by design — content, not order).

  theta (latitude) — information content. theta = pi * surprisal / max
      surprisal, so frequent tokens sit near the |0> pole and rare tokens
      migrate toward |1>.

The experiment at the bottom tests whether this geometry is *meaningful*:
if it is, geodesic distance on the sphere should track token similarity
(content overlap and distributional similarity), whereas random placement
should show ~zero correlation.
"""

import math
import os
from collections import Counter

import numpy as np
from scipy.stats import spearmanr

# Just perfect fifth, in octaves. Irrational => the spiral of fifths never closes.
FIFTH = math.log2(3 / 2)

TWO_PI = 2 * math.pi


# ---------------------------------------------------------------------------
# BPE (standalone copy of the repo's greedy char-level BPE, no qiskit needed)
# ---------------------------------------------------------------------------

def byte_pair_encoding(text, num_merges):
    """Greedy character-level BPE; returns (token_stream, merge_history)."""
    words = list(text)
    merges = []
    for _ in range(num_merges):
        pairs = Counter(zip(words, words[1:]))
        if not pairs:
            break
        most_common = max(pairs, key=pairs.get)
        merges.append(most_common)
        merged = ''.join(most_common)
        new_words = []
        i = 0
        n = len(words)
        while i < n:
            if i < n - 1 and words[i] == most_common[0] and words[i + 1] == most_common[1]:
                new_words.append(merged)
                i += 2
            else:
                new_words.append(words[i])
                i += 1
        words = new_words
    return words, merges


# ---------------------------------------------------------------------------
# Placement rules
# ---------------------------------------------------------------------------

def fifths_phi(token):
    """Longitude via the continuous spiral of fifths (circle-group sum)."""
    byte_sum = sum(token.encode('utf-8'))
    return TWO_PI * ((byte_sum * FIFTH) % 1.0)


def harmonic_placement(token_counts):
    """token -> (theta, phi). phi = spiral-of-fifths content angle,
    theta = normalized surprisal (frequent -> |0> pole)."""
    total = sum(token_counts.values())
    surprisal = {t: -math.log2(c / total) for t, c in token_counts.items()}
    max_s = max(surprisal.values())
    # keep off the exact poles so phi stays visible/meaningful
    eps = 0.02 * math.pi
    vocab = {}
    for t in token_counts:
        theta = eps + (math.pi - 2 * eps) * (surprisal[t] / max_s)
        vocab[t] = (theta, fifths_phi(t))
    return vocab


def chord_placement(token_counts):
    """token -> (theta, phi), 'chord centroid' variant.

    Each character is a note on the continuous spiral of fifths; a token is
    the CHORD of its characters, placed at the circular mean (resultant
    vector angle) of its notes — Elaine Chew's Spiral Array 'center of
    effect', applied to text. Unlike the sum rule (an equidistributed hash
    that preserves composition but destroys locality), the centroid is
    locality-preserving: tokens sharing characters get nearby phi.
    theta is surprisal, as in harmonic_placement.
    """
    total = sum(token_counts.values())
    surprisal = {t: -math.log2(c / total) for t, c in token_counts.items()}
    max_s = max(surprisal.values())
    eps = 0.02 * math.pi
    vocab = {}
    for t in token_counts:
        angles = [TWO_PI * ((b * FIFTH) % 1.0) for b in t.encode('utf-8')]
        x = sum(math.cos(a) for a in angles)
        y = sum(math.sin(a) for a in angles)
        if x * x + y * y < 1e-12:  # degenerate chord, fall back to sum rule
            phi = fifths_phi(t)
        else:
            phi = math.atan2(y, x) % TWO_PI
        theta = eps + (math.pi - 2 * eps) * (surprisal[t] / max_s)
        vocab[t] = (theta, phi)
    return vocab


def random_placement(token_counts, seed=42):
    """Uniform-on-sphere null model."""
    rng = np.random.default_rng(seed)
    vocab = {}
    for t in token_counts:
        theta = math.acos(1 - 2 * rng.uniform())
        phi = TWO_PI * rng.uniform()
        vocab[t] = (theta, phi)
    return vocab


def bloch_xyz(theta, phi):
    return (math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta))


def geodesic_matrix(vocab, tokens):
    """Pairwise great-circle distances between token states."""
    pts = np.array([bloch_xyz(*vocab[t]) for t in tokens])
    dots = np.clip(pts @ pts.T, -1.0, 1.0)
    return np.arccos(dots)


# ---------------------------------------------------------------------------
# Similarity targets: what the geometry *should* predict
# ---------------------------------------------------------------------------

def content_similarity_matrix(tokens):
    """Cosine similarity of character-count (bag-of-chars) vectors."""
    chars = sorted({c for t in tokens for c in t})
    idx = {c: i for i, c in enumerate(chars)}
    M = np.zeros((len(tokens), len(chars)))
    for r, t in enumerate(tokens):
        for c in t:
            M[r, idx[c]] += 1
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    M = M / np.where(norms == 0, 1, norms)
    return M @ M.T


def ppmi_similarity_matrix(tokens, stream, window=2):
    """Cosine similarity of PPMI co-occurrence vectors (distributional)."""
    idx = {t: i for i, t in enumerate(tokens)}
    V = len(tokens)
    C = np.zeros((V, V))
    ids = np.array([idx[t] for t in stream])
    for off in range(1, window + 1):
        a, b = ids[:-off], ids[off:]
        np.add.at(C, (a, b), 1)
        np.add.at(C, (b, a), 1)
    total = C.sum()
    row = C.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        pmi = np.log((C * total) / (row * row.T))
    ppmi = np.where(np.isfinite(pmi) & (pmi > 0), pmi, 0.0)
    norms = np.linalg.norm(ppmi, axis=1, keepdims=True)
    ppmi = ppmi / np.where(norms == 0, 1, norms)
    return ppmi @ ppmi.T


def upper_triangle(M):
    iu = np.triu_indices(M.shape[0], k=1)
    return M[iu]


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_experiment(corpus_path='data/pride_and_prejudice.txt',
                   corpus_chars=100_000, num_merges=150, viz=True):
    with open(corpus_path, encoding='utf-8') as f:
        raw = f.read()
    # strip Gutenberg header/footer
    start = raw.find('*** START')
    start = raw.find('\n', start) + 1 if start != -1 else 0
    end = raw.find('*** END')
    text = raw[start:end if end != -1 else len(raw)][:corpus_chars]

    print(f'Corpus: {len(text):,} chars   BPE merges: {num_merges}')
    stream, merges = byte_pair_encoding(text, num_merges)
    counts = Counter(stream)
    tokens = sorted(counts)
    print(f'Token stream: {len(stream):,}   vocab: {len(tokens)}')

    harmonic = harmonic_placement(counts)
    chord = chord_placement(counts)
    random_v = random_placement(counts)

    # homomorphism sanity check: phi(AB) == phi(A) + phi(B) mod 2*pi
    worst = 0.0
    for a, b in merges:
        merged = a + b
        if merged in harmonic and a in harmonic and b in harmonic:
            expect = (harmonic[a][1] + harmonic[b][1]) % TWO_PI
            err = abs(expect - harmonic[merged][1])
            worst = max(worst, min(err, TWO_PI - err))
    print(f'Circle-group homomorphism max error over {len(merges)} merges: {worst:.2e} rad')

    content = upper_triangle(content_similarity_matrix(tokens))
    ppmi = upper_triangle(ppmi_similarity_matrix(tokens, stream))

    results = {}
    for name, vocab in (('sum-rule', harmonic), ('chord', chord), ('random', random_v)):
        dist = upper_triangle(geodesic_matrix(vocab, tokens))
        rc, pc = spearmanr(dist, content)
        rp, pp = spearmanr(dist, ppmi)
        results[name] = dict(content=(rc, pc), ppmi=(rp, pp))

    print(f'\n{len(content):,} token pairs. Spearman rho, geodesic distance vs similarity')
    print('(meaningful geometry => NEGATIVE rho: similar tokens closer together)\n')
    print(f'{"placement":<12}{"vs content sim":>22}{"vs PPMI (distributional)":>28}')
    for name, r in results.items():
        rc, pc = r['content']
        rp, pp = r['ppmi']
        print(f'{name:<12}{rc:>14.3f} (p={pc:.1e}){rp:>18.3f} (p={pp:.1e})')

    if viz:
        visualize(chord, random_v, counts,
                  'docs/images/fifths_placement/fifths_vs_random.html')
    return results


def visualize(harmonic, random_v, counts, out_path):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tokens = sorted(counts)
    freqs = np.log10([counts[t] for t in tokens])

    fig = make_subplots(
        rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=['Chord-Centroid Placement (spiral of fifths + surprisal)',
                        'Random Placement (null model)'])

    u, v = np.mgrid[0:TWO_PI:40j, 0:np.pi:20j]
    wire = dict(x=np.cos(u) * np.sin(v), y=np.sin(u) * np.sin(v), z=np.cos(v),
                opacity=0.12, showscale=False, colorscale='Greys')

    for col, vocab in ((1, harmonic), (2, random_v)):
        pts = np.array([bloch_xyz(*vocab[t]) for t in tokens])
        fig.add_trace(go.Surface(**wire), row=1, col=col)
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode='markers',
            marker=dict(size=4, color=freqs, colorscale='Viridis',
                        colorbar=dict(title='log10 freq') if col == 1 else None,
                        showscale=(col == 1)),
            text=[repr(t) for t in tokens],
            hovertemplate='token %{text}<extra></extra>',
            showlegend=False), row=1, col=col)

    fig.update_layout(title='BPE Tokens on the Bloch Sphere: Harmonic vs Random Placement',
                      height=650, width=1300)
    fig.write_html(out_path)
    print(f'\nVisualization written to {out_path}')


if __name__ == '__main__':
    run_experiment()
