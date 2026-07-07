"""Harmonic metadata, taxonomy, and geometric retrieval.

Every analyzed recording gets a compact HARMONIC SIGNATURE derived from the
chroma -> BPE -> chord-centroid pipeline — no score, no tags, no human input:

  key_angle    circular-mean longitude of the token cloud on the circle of
               fifths (the diatonic +60 deg offset removed), i.e. estimated
               tonal center as geometry
  key          nearest named key to that angle
  coherence    resultant length of the token cloud in [0, 1] — how tonally
               focused the piece is (1 = a single point on the circle)
  drift_span   angular spread of the sliding-window tonal center over time —
               how far the piece modulates
  entropy      Shannon entropy of the token distribution (bits) — motif
               diversity
  motifs       most frequent BPE tokens with their sphere coordinates —
               the piece's reusable harmonic vocabulary

Signatures are stored in a JSON catalog (data/harmonic_catalog.json). The
TAXONOMY is read straight off the geometry — tonal center (12 branches) x
coherence class x modulation class — and RETRIEVAL is circular geometry:

  compatible(piece)   pieces whose tonal centers are adjacent on the circle
                      of fifths (dominant / subdominant / relative) — what a
                      composer can segue or overlay without a clash
  motif_search(pcs)   rank every motif in the catalog by geodesic distance
                      to a query pitch-class string — 'find me material that
                      sounds like this' across the whole library

Run: python3 harmonic_metadata.py   (expects data/wtc/*.wav + giant_steps.wav)
"""

import glob
import json
import math
import os
from collections import Counter

import numpy as np

from audio_tokenizer import audio_chord_placement, note_phi, NOTE
from harmonic_placement import byte_pair_encoding, bloch_xyz
from real_audio_analysis import (wav_to_symbols, analyze_piece,
                                 modulation_timeline, key_angle, circ_diff,
                                 FIFTHS_ORDER)

TWO_PI = 2 * math.pi
DIATONIC_OFFSET = math.radians(60)  # circular mean of a major diatonic set
CATALOG_PATH = 'data/harmonic_catalog.json'


# ---------------------------------------------------------------------------
# Signature extraction
# ---------------------------------------------------------------------------

def harmonic_signature(name, wav_path, num_merges=80, top_motifs=12):
    symbols = wav_to_symbols(wav_path)
    vocab, counts, est, coh = analyze_piece(symbols, num_merges)
    adj = (est - DIATONIC_OFFSET) % TWO_PI
    key = min(FIFTHS_ORDER, key=lambda q: abs(circ_diff(adj, key_angle(q))))

    _, drift, _ = modulation_timeline(symbols)
    rad = np.radians(drift)
    span = math.degrees(_circular_std(rad))

    total = sum(counts.values())
    probs = np.array([c / total for c in counts.values()])
    entropy = float(-(probs * np.log2(probs)).sum())

    motifs = [dict(pcs=t, count=c, theta=vocab[t][0], phi=vocab[t][1])
              for t, c in counts.most_common(top_motifs)]
    return dict(name=name, file=wav_path, frames=len(symbols),
                key=key, key_angle_deg=round(math.degrees(adj), 1),
                coherence=round(coh, 3), drift_span_deg=round(span, 1),
                token_entropy_bits=round(entropy, 2),
                vocab_size=len(counts), motifs=motifs)


def _circular_std(angles):
    x, y = np.cos(angles).mean(), np.sin(angles).mean()
    R = min(1.0, math.hypot(x, y))
    return math.sqrt(-2 * math.log(max(R, 1e-12)))


# ---------------------------------------------------------------------------
# Taxonomy: classes read off the geometry
# ---------------------------------------------------------------------------

def coherence_class(sig):
    c = sig['coherence']
    return 'focused' if c >= 0.45 else 'tonal' if c >= 0.30 else 'chromatic'

def modulation_class(sig):
    s = sig['drift_span_deg']
    return 'static' if s < 25 else 'mild' if s < 45 else 'roving'

def taxonomy(catalog):
    """center -> coherence class -> modulation class -> pieces."""
    tree = {}
    for sig in catalog:
        (tree.setdefault(sig['key'], {})
             .setdefault(coherence_class(sig), {})
             .setdefault(modulation_class(sig), [])).append(sig['name'])
    return tree

def print_taxonomy(tree):
    for key in sorted(tree, key=FIFTHS_ORDER.index):
        print(f'  {key}')
        for coh, mods in tree[key].items():
            for mod, names in mods.items():
                print(f'    {coh}/{mod}: {", ".join(names)}')


# ---------------------------------------------------------------------------
# Geometric retrieval
# ---------------------------------------------------------------------------

def compatible(sig, catalog, max_fifths=1):
    """Pieces within max_fifths steps on the circle of fifths — safe to
    segue/overlay. Sorted by angular distance."""
    out = []
    for other in catalog:
        if other['name'] == sig['name']:
            continue
        d = abs(circ_diff(math.radians(other['key_angle_deg']),
                          math.radians(sig['key_angle_deg'])))
        if d <= max_fifths * TWO_PI / 12 + 1e-9:
            out.append((math.degrees(d), other['name'], other['key']))
    return sorted(out)

def pcs_centroid(pcs):
    """Chord-centroid longitude of a pitch-class string like '047' (C E G)."""
    angles = [note_phi(c) for c in pcs]
    x = sum(math.cos(a) for a in angles)
    y = sum(math.sin(a) for a in angles)
    return math.atan2(y, x) % TWO_PI

def motif_search(query_pcs, catalog, k=8):
    """Rank all catalog motifs by circular distance to the query's centroid."""
    q = pcs_centroid(query_pcs)
    hits = []
    for sig in catalog:
        for m in sig['motifs']:
            d = abs(circ_diff(m['phi'], q))
            hits.append((math.degrees(d), sig['name'], m['pcs'], m['count']))
    return sorted(hits)[:k]


# ---------------------------------------------------------------------------
# Build + demo
# ---------------------------------------------------------------------------

def build_catalog():
    jobs = [('giant_steps (synth)', 'data/giant_steps.wav')]
    for p in sorted(glob.glob('data/wtc/prelude_*.wav')):
        key = os.path.basename(p).replace('prelude_', '').replace('.wav', '')
        jobs.append((f'WTC I prelude in {key}', p))
    catalog = []
    for name, path in jobs:
        sig = harmonic_signature(name, path)
        catalog.append(sig)
        print(f"{sig['name']:<24} key={sig['key']:<3} "
              f"angle={sig['key_angle_deg']:6.1f} "
              f"coh={sig['coherence']:.2f} drift={sig['drift_span_deg']:5.1f} "
              f"H={sig['token_entropy_bits']:.2f} bits")
    os.makedirs('data', exist_ok=True)
    with open(CATALOG_PATH, 'w') as f:
        json.dump(catalog, f, indent=1)
    print(f'\nCatalog written to {CATALOG_PATH} ({len(catalog)} records)')
    return catalog


def demo(catalog):
    print('\n=== Taxonomy (tonal center / coherence / modulation) ===')
    print_taxonomy(taxonomy(catalog))

    c_maj = next(s for s in catalog if s['name'].endswith('prelude in C'))
    print(f"\n=== Compatible with '{c_maj['name']}' (within 1 fifth) ===")
    for d, name, key in compatible(c_maj, catalog):
        print(f'  {d:5.1f} deg  {name} ({key})')

    print('\n=== Motif search: query = C E G (pcs 047) ===')
    for d, name, pcs, count in motif_search('047', catalog):
        print(f'  {d:5.1f} deg  {pcs!r:>10} x{count:<3} in {name}')

    print('\n=== Motif search: query = B D# F# (pcs B36) ===')
    for d, name, pcs, count in motif_search('B36', catalog):
        print(f'  {d:5.1f} deg  {pcs!r:>10} x{count:<3} in {name}')


if __name__ == '__main__':
    demo(build_catalog())
