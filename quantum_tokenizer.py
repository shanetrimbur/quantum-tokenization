import numpy as np
from collections import Counter
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.primitives import BackendEstimator  # For executing circuits
from typing import Dict, List, Tuple, Set
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.io import write_image

class QuantumTokenizer:
    def __init__(self, num_merges: int = 10):
        self.num_merges = num_merges
        self.vocab: Dict[str, Tuple[float, float]] = {}  # token -> (theta, phi)
        self.tokens: List[str] = []
        
    def byte_pair_encoding(self, text: str) -> List[str]:
        """Implement BPE tokenization"""
        words = list(text)
        
        for _ in range(self.num_merges):
            pairs = Counter(zip(words, words[1:]))
            if not pairs:
                break
            most_common = max(pairs, key=pairs.get)
            new_words = []
            i = 0
            while i < len(words) - 1:
                if (words[i], words[i+1]) == most_common:
                    new_words.append(''.join(most_common))
                    i += 2
                else:
                    new_words.append(words[i])
                    i += 1
            if i == len(words) - 1:
                new_words.append(words[-1])
            words = new_words
            
        return words

    def map_to_bloch(self, tokens: List[str]) -> None:
        """Map tokens to quantum states on Bloch sphere"""
        unique_tokens = set(tokens)
        for token in unique_tokens:
            if token not in self.vocab:
                theta = np.random.uniform(0, np.pi)
                phi = np.random.uniform(0, 2*np.pi)
                self.vocab[token] = (theta, phi)

    def get_quantum_state(self, token: str) -> np.ndarray:
        """Get quantum state representation for a token"""
        if token not in self.vocab:
            raise ValueError(f"Token '{token}' not in vocabulary")
            
        theta, phi = self.vocab[token]
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)
        qc.rz(phi, 0)
        
        backend = Aer.get_backend('statevector_simulator')
        transpiled_circuit = transpile(qc, backend)
        job = backend.run(transpiled_circuit)
        result = job.result()
        return result.get_statevector()

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text and create quantum mappings"""
        self.tokens = self.byte_pair_encoding(text)
        self.map_to_bloch(self.tokens)
        return self.tokens
        
    def save_vocab(self, filepath: str) -> None:
        """Save vocabulary to file"""
        # Convert numpy types to native Python types for JSON serialization
        vocab_to_save = {
            token: (float(theta), float(phi)) 
            for token, (theta, phi) in self.vocab.items()
        }
        with open(filepath, 'w') as f:
            json.dump(vocab_to_save, f)
            
    def load_vocab(self, filepath: str) -> None:
        """Load vocabulary from file"""
        with open(filepath, 'r') as f:
            self.vocab = json.load(f)

    def get_stats(self) -> Dict:
        """Get tokenization statistics"""
        if not self.tokens:
            return {}
            
        return {
            'num_tokens': len(self.tokens),
            'vocab_size': len(self.vocab),
            'classical_size': len(self.tokens) * 8,  # bits
            'quantum_size': len(self.vocab) * 2,  # parameters
            'compression_ratio': (len(self.vocab) * 2) / (len(self.tokens) * 8)
        } 

    def bloch_coordinates(self, theta: float, phi: float) -> Tuple[float, float, float]:
        """Convert spherical coordinates to cartesian for Bloch sphere"""
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return x, y, z

    def visualize_tokens_3d(self) -> go.Figure:
        """Create 3D visualization of tokens on Bloch sphere"""
        # Create the Bloch sphere
        phi = np.linspace(0, 2*np.pi, 100)
        theta = np.linspace(0, np.pi, 100)
        phi_grid, theta_grid = np.meshgrid(phi, theta)
        
        x_sphere = np.sin(theta_grid) * np.cos(phi_grid)
        y_sphere = np.sin(theta_grid) * np.sin(phi_grid)
        z_sphere = np.cos(theta_grid)

        fig = go.Figure()

        # Add semi-transparent Bloch sphere
        fig.add_trace(go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            opacity=0.1,
            showscale=False,
            name='Bloch Sphere'
        ))

        # Add token points
        x_tokens, y_tokens, z_tokens = [], [], []
        token_labels = []
        
        for token, (theta, phi) in self.vocab.items():
            x, y, z = self.bloch_coordinates(theta, phi)
            x_tokens.append(x)
            y_tokens.append(y)
            z_tokens.append(z)
            token_labels.append(token)

        fig.add_trace(go.Scatter3d(
            x=x_tokens,
            y=y_tokens,
            z=z_tokens,
            mode='markers+text',
            text=token_labels,
            hovertemplate="Token: %{text}<br>" +
                         "x: %{x:.3f}<br>" +
                         "y: %{y:.3f}<br>" +
                         "z: %{z:.3f}",
            marker=dict(
                size=8,
                color=list(range(len(token_labels))),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Token Index")
            )
        ))

        # Update layout
        fig.update_layout(
            title="Quantum Token Representations on Bloch Sphere",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z"
            ),
            width=800,
            height=800
        )

        return fig

    def visualize_stats(self) -> go.Figure:
        """Create visualization of tokenization statistics"""
        stats = self.get_stats()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Token Distribution",
                "Compression Analysis",
                "Token Length Distribution",
                "Token Frequency"
            )
        )

        # Token distribution
        token_lengths = [len(t) for t in self.tokens]
        fig.add_trace(
            go.Histogram(x=token_lengths, name="Token Lengths"),
            row=1, col=1
        )

        # Compression visualization
        fig.add_trace(
            go.Bar(
                x=['Classical', 'Quantum'],
                y=[stats['classical_size'], stats['quantum_size']],
                name="Size Comparison"
            ),
            row=1, col=2
        )

        # Token length distribution
        token_freq = Counter(self.tokens)
        lengths = list(map(len, token_freq.keys()))
        freqs = list(token_freq.values())
        fig.add_trace(
            go.Scatter(x=lengths, y=freqs, mode='markers', name="Length vs Frequency"),
            row=2, col=1
        )

        # Token frequency
        most_common = Counter(self.tokens).most_common(10)
        fig.add_trace(
            go.Bar(
                x=[t[0] for t in most_common],
                y=[t[1] for t in most_common],
                name="Most Common Tokens"
            ),
            row=2, col=2
        )

        fig.update_layout(height=800, width=1000, title_text="Tokenization Analysis")
        return fig 

    def save_visualizations(self, prefix: str = "docs/images") -> None:
        """Save both interactive HTML and static images of visualizations"""
        # Create Bloch sphere visualization
        bloch_fig = self.visualize_tokens_3d()
        bloch_fig.write_html(f"{prefix}/bloch_visualization.html")
        bloch_fig.write_image(f"{prefix}/bloch_sphere.png", scale=2)

        # Create stats visualization
        stats_fig = self.visualize_stats()
        stats_fig.write_html(f"{prefix}/stats_visualization.html")
        stats_fig.write_image(f"{prefix}/stats_analysis.png", scale=2) 