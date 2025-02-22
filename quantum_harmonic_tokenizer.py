import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from collections import Counter
import re

class QuantumHarmonicTokenizer:
    def __init__(self):
        """Initialize quantum tokenizer"""
        os.makedirs('docs/images/quantum_tokenization', exist_ok=True)
        self.backend = Aer.get_backend('statevector_simulator')
        np.random.seed(42)
        
        # Musical note frequencies for harmonic mapping
        self.base_frequencies = {
            'C': 261.63, 'G': 392.00, 'D': 293.66,
            'A': 440.00, 'E': 329.63, 'B': 493.88
        }

    def text_to_quantum_tokens(self, text, modulus=12):
        """Convert text to quantum tokens using harmonic principles"""
        # First pass: identify common substrings
        substrings = [text[i:i+2] for i in range(len(text)-1)]
        freq = Counter(substrings)
        common_pairs = {pair: count for pair, count in freq.items() if count > 1}
        
        # Map frequent pairs to harmonic angles
        quantum_tokens = []
        i = 0
        while i < len(text)-1:
            pair = text[i:i+2]
            if pair in common_pairs:
                # Map to harmonic angles using musical ratios
                freq_ratio = (hash(pair) % 12) / 12
                theta = np.pi * freq_ratio
                phi = 2 * np.pi * freq_ratio
                
                # Create quantum circuit for token
                qc = QuantumCircuit(1)
                qc.ry(theta, 0)
                qc.rz(phi, 0)
                
                # Get quantum state
                state = self.get_statevector(qc)
                quantum_tokens.append((pair, state))
                i += 2
            else:
                # Single character token
                char = text[i]
                freq_ratio = (ord(char) % 12) / 12
                theta = np.pi * freq_ratio
                phi = 2 * np.pi * freq_ratio
                
                qc = QuantumCircuit(1)
                qc.ry(theta, 0)
                qc.rz(phi, 0)
                
                state = self.get_statevector(qc)
                quantum_tokens.append((char, state))
                i += 1
        
        return quantum_tokens

    def get_statevector(self, circuit):
        """Get statevector from quantum circuit"""
        job = self.backend.run(circuit)
        result = job.result()
        return result.get_statevector()

    def classical_bpe_tokenize(self, text, vocab_size=1000):
        """Classical BPE tokenization for comparison"""
        # Simple BPE implementation
        vocab = set(text)
        while len(vocab) < vocab_size:
            pairs = Counter(zip(text[:-1], text[1:]))
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            new_token = ''.join(best_pair)
            vocab.add(new_token)
            text = re.sub(f'{best_pair[0]}{best_pair[1]}', new_token, text)
        
        return list(text)

    def visualize_comparison(self, text):
        """Visualize quantum vs classical tokenization"""
        # Get tokenizations
        quantum_tokens = self.text_to_quantum_tokens(text)
        classical_tokens = self.classical_bpe_tokenize(text)
        
        frames = []
        for t in np.linspace(0, 2*np.pi, 60):
            fig = make_subplots(
                rows=2, cols=2,
                specs=[
                    [{'type': 'scene'}, {'type': 'xy'}],
                    [{'type': 'xy'}, {'type': 'xy'}]
                ],
                subplot_titles=[
                    "Quantum Token States",
                    "Token Distribution",
                    "Compression Ratio",
                    "Information Density"
                ]
            )
            
            # Plot quantum states
            for token, state in quantum_tokens[:10]:  # Show first 10 tokens
                x = np.real(state[0])
                y = np.real(state[1])
                z = np.imag(state[1])
                
                fig.add_trace(
                    go.Scatter3d(
                        x=[0, x], y=[0, y], z=[0, z],
                        mode='lines+markers',
                        name=f'Token: {token}'
                    ),
                    row=1, col=1
                )
            
            # Token distribution
            q_freq = Counter([t[0] for t in quantum_tokens])
            c_freq = Counter(classical_tokens)
            
            fig.add_trace(
                go.Bar(
                    x=list(range(len(q_freq))),
                    y=list(q_freq.values()),
                    name='Quantum Tokens'
                ),
                row=1, col=2
            )
            
            # Compression ratios
            methods = ['Classical', 'Quantum']
            ratios = [len(text)/len(classical_tokens),
                     len(text)/len(quantum_tokens)]
            
            fig.add_trace(
                go.Bar(
                    x=methods,
                    y=ratios,
                    name='Compression Ratio'
                ),
                row=2, col=1
            )
            
            # Information density
            q_density = len(set([t[0] for t in quantum_tokens])) / len(quantum_tokens)
            c_density = len(set(classical_tokens)) / len(classical_tokens)
            
            fig.add_trace(
                go.Bar(
                    x=methods,
                    y=[c_density, q_density],
                    name='Information Density'
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title="Quantum vs Classical Tokenization",
                height=900,
                scene=dict(
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                )
            )
            
            frames.append(go.Frame(data=fig.data))
        
        fig = go.Figure(
            data=frames[0].data,
            layout=go.Layout(
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [{
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': 100}}]
                    }]
                }]
            ),
            frames=frames
        )
        
        fig.write_html('docs/images/quantum_tokenization/comparison.html')
        fig.write_image('docs/images/quantum_tokenization/comparison.png')

# Test the tokenizer
sample_text = """
Quantum computing leverages the principles of quantum mechanics to process 
information in fundamentally new ways. By mapping data onto quantum states 
and using quantum operations, we can achieve more efficient compression and 
processing than classical methods.
"""

tokenizer = QuantumHarmonicTokenizer()
tokenizer.visualize_comparison(sample_text) 