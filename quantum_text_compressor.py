import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import requests
from collections import Counter
from tqdm import tqdm

class QuantumTextCompressor:
    def __init__(self):
        """Initialize quantum compressor"""
        os.makedirs('docs/images/quantum_compression', exist_ok=True)
        self.backend = Aer.get_backend('statevector_simulator')
        np.random.seed(42)
        
        # Pre-compute quantum circuits for common patterns
        self.circuit_cache = {}

    def get_cached_circuit(self, pattern):
        """Get or create quantum circuit for a pattern"""
        if pattern not in self.circuit_cache:
            # Map pattern to angles using hash function
            theta = np.pi * (hash(pattern) % 12) / 12
            phi = 2 * np.pi * (hash(pattern[::-1]) % 12) / 12
            
            qc = QuantumCircuit(1)
            qc.ry(theta, 0)
            qc.rz(phi, 0)
            
            # Cache the statevector directly
            job = self.backend.run(qc)
            self.circuit_cache[pattern] = job.result().get_statevector()
            
        return self.circuit_cache[pattern]

    def compress_text(self, text, window_size=4):
        """Compress text using quantum circuits"""
        # Find common patterns
        patterns = []
        for i in range(len(text) - window_size + 1):
            patterns.append(text[i:i+window_size])
        
        pattern_counts = Counter(patterns)
        common_patterns = {p: c for p, c in pattern_counts.items() if c > 1}
        
        # Create quantum encodings for common patterns
        quantum_tokens = []
        i = 0
        with tqdm(total=len(text), desc="Compressing") as pbar:
            while i < len(text):
                best_pattern = None
                best_len = 0
                
                # Look for longest matching pattern
                for size in range(min(window_size, len(text) - i), 0, -1):
                    pattern = text[i:i+size]
                    if pattern in common_patterns:
                        best_pattern = pattern
                        best_len = size
                        break
                
                if best_pattern:
                    state = self.get_cached_circuit(best_pattern)
                    quantum_tokens.append((best_pattern, state))
                    i += best_len
                else:
                    state = self.get_cached_circuit(text[i])
                    quantum_tokens.append((text[i], state))
                    i += 1
                    
                pbar.update(best_len or 1)
        
        return quantum_tokens

    def analyze_compression(self, original_text, quantum_tokens):
        """Analyze compression performance"""
        original_size = len(original_text)
        compressed_size = len(quantum_tokens)
        unique_tokens = len(set(t[0] for t in quantum_tokens))
        
        # Calculate compression metrics
        compression_ratio = original_size / compressed_size
        token_efficiency = unique_tokens / compressed_size
        
        return {
            'original_size': original_size,
            'compressed_size': compressed_size,
            'unique_tokens': unique_tokens,
            'compression_ratio': compression_ratio,
            'token_efficiency': token_efficiency
        }

    def visualize_results(self, text, quantum_tokens, metrics):
        """Visualize compression results"""
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{'type': 'scene'}, {'type': 'xy'}],
                [{'type': 'xy'}, {'type': 'xy'}]
            ],
            subplot_titles=[
                "Quantum Token States",
                "Token Length Distribution",
                "Compression Metrics",
                "Token Frequency"
            ]
        )
        
        # Plot quantum states for first 20 unique tokens
        unique_tokens = list(set(t[0] for t in quantum_tokens))[:20]
        for token in unique_tokens:
            state = self.get_cached_circuit(token)
            x = np.real(state[0])
            y = np.real(state[1])
            z = np.imag(state[1])
            
            fig.add_trace(
                go.Scatter3d(
                    x=[0, x], y=[0, y], z=[0, z],
                    mode='lines+markers',
                    name=f'Token: {token[:10]}...' if len(token) > 10 else f'Token: {token}'
                ),
                row=1, col=1
            )
        
        # Token length distribution
        token_lengths = [len(t[0]) for t in quantum_tokens]
        length_counts = Counter(token_lengths)
        
        fig.add_trace(
            go.Bar(
                x=list(length_counts.keys()),
                y=list(length_counts.values()),
                name='Token Lengths'
            ),
            row=1, col=2
        )
        
        # Compression metrics
        metrics_labels = ['Original', 'Compressed', 'Unique']
        metrics_values = [metrics['original_size'], 
                         metrics['compressed_size'],
                         metrics['unique_tokens']]
        
        fig.add_trace(
            go.Bar(
                x=metrics_labels,
                y=metrics_values,
                name='Size Metrics'
            ),
            row=2, col=1
        )
        
        # Token frequency
        token_freq = Counter(t[0] for t in quantum_tokens)
        most_common = token_freq.most_common(10)
        
        fig.add_trace(
            go.Bar(
                x=[t[0][:10] + '...' if len(t[0]) > 10 else t[0] for t in most_common],
                y=[t[1] for t in most_common],
                name='Token Frequency'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Quantum Text Compression Analysis",
            height=1000,
            showlegend=False,
            scene=dict(
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            )
        )
        
        fig.write_html('docs/images/quantum_compression/text_analysis.html')
        fig.write_image('docs/images/quantum_compression/text_analysis.png')

def get_sample_text():
    """Get sample text from Project Gutenberg"""
    try:
        # Use a local sample text if the download fails
        sample = """
        It is a truth universally acknowledged, that a single man in possession of a good fortune, 
        must be in want of a wife. However little known the feelings or views of such a man may be 
        on his first entering a neighbourhood, this truth is so well fixed in the minds of the 
        surrounding families, that he is considered the rightful property of some one or other of 
        their daughters.

        "My dear Mr. Bennet," said his lady to him one day, "have you heard that Netherfield Park 
        is let at last?"

        Mr. Bennet replied that he had not.

        "But it is," returned she; "for Mrs. Long has just been here, and she told me all about it."
        Mr. Bennet made no answer.

        "Do you not want to know who has taken it?" cried his wife impatiently.

        "You want to tell me, and I have no objection to hearing it."

        This was invitation enough.
        """
        
        try:
            # Try to get more text from Project Gutenberg
            url = "https://www.gutenberg.org/files/1342/1342-0.txt"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                text = response.text
                # Take first chapter
                chapter_text = text[text.find("Chapter 1"):text.find("Chapter 2")]
                if len(chapter_text) > 100:  # Verify we got meaningful text
                    return chapter_text[:5000]
        except:
            pass  # Fall back to sample text if download fails
        
        return sample

    except Exception as e:
        print(f"Error loading text: {e}")
        return "Sample text for testing quantum compression algorithms."

if __name__ == "__main__":
    try:
        print("Loading sample text...")
        sample_text = get_sample_text()
        
        if not sample_text:
            raise ValueError("Failed to load sample text")
            
        print(f"Loaded text of length: {len(sample_text)} characters")
        
        print("Initializing quantum compressor...")
        compressor = QuantumTextCompressor()
        
        print("Compressing text...")
        quantum_tokens = compressor.compress_text(sample_text)
        
        if not quantum_tokens:
            raise ValueError("Compression produced no tokens")
            
        print("Analyzing results...")
        metrics = compressor.analyze_compression(sample_text, quantum_tokens)
        
        print("\nCompression Results:")
        print(f"Original size: {metrics['original_size']} characters")
        print(f"Compressed size: {metrics['compressed_size']} tokens")
        print(f"Unique tokens: {metrics['unique_tokens']}")
        print(f"Compression ratio: {metrics['compression_ratio']:.2f}x")
        print(f"Token efficiency: {metrics['token_efficiency']:.2f}")
        
        print("\nGenerating visualizations...")
        compressor.visualize_results(sample_text, quantum_tokens, metrics)
        print("Done! Check the docs/images/quantum_compression directory for results.")
        
    except Exception as e:
        print(f"Error during compression: {e}") 