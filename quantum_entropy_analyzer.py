import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector, entropy, partial_trace
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from collections import Counter
from scipy.stats import entropy as classical_entropy
import time

class QuantumEntropyAnalyzer:
    def __init__(self):
        """Initialize entropy analyzer"""
        os.makedirs('docs/images/entropy_analysis', exist_ok=True)
        self.backend = Aer.get_backend('statevector_simulator')
        np.random.seed(42)

    def classical_entropy(self, text):
        """Calculate classical Shannon entropy"""
        freq = Counter(text)
        probs = np.array(list(freq.values())) / len(text)
        return classical_entropy(probs, base=2)

    def quantum_entropy(self, state):
        """Calculate von Neumann entropy"""
        if isinstance(state, np.ndarray):
            state = Statevector(state)
        return entropy(state)

    def create_quantum_state(self, text_chunk, use_phase=True):
        """Create quantum state from text chunk"""
        qc = QuantumCircuit(2)  # Using 2 qubits for richer encoding
        
        # Map text features to angles
        freq = Counter(text_chunk)
        most_common = freq.most_common(1)[0][1] / len(text_chunk)
        pattern_score = len(set(text_chunk)) / len(text_chunk)
        
        # Encode information in rotations and entanglement
        theta1 = np.pi * most_common
        theta2 = np.pi * pattern_score
        phi = 2 * np.pi * (hash(text_chunk) % 12) / 12 if use_phase else 0
        
        qc.ry(theta1, 0)
        qc.ry(theta2, 1)
        if use_phase:
            qc.rz(phi, 0)
        qc.cx(0, 1)  # Entangle qubits
        
        # Get statevector
        job = self.backend.run(qc)
        state = job.result().get_statevector()
        return state

    def analyze_entropy_reduction(self, text, window_size=4):
        """Analyze entropy reduction using different methods"""
        results = {
            'classical': [],
            'quantum_basic': [],
            'quantum_phase': [],
            'quantum_entangled': []
        }
        
        # Analyze text in chunks
        chunks = [text[i:i+window_size] for i in range(0, len(text), window_size)]
        
        for chunk in chunks:
            # Classical entropy
            classical = self.classical_entropy(chunk)
            results['classical'].append(classical)
            
            # Basic quantum encoding (no phase)
            state_basic = self.create_quantum_state(chunk, use_phase=False)
            quantum_basic = self.quantum_entropy(state_basic)
            results['quantum_basic'].append(quantum_basic)
            
            # Quantum with phase encoding
            state_phase = self.create_quantum_state(chunk, use_phase=True)
            quantum_phase = self.quantum_entropy(state_phase)
            results['quantum_phase'].append(quantum_phase)
            
            # Entangled state analysis
            reduced_state = partial_trace(state_phase, [1])
            quantum_entangled = self.quantum_entropy(reduced_state)
            results['quantum_entangled'].append(quantum_entangled)
        
        return results

    def visualize_entropy_analysis(self, results):
        """Create visualizations of entropy analysis"""
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{'type': 'xy'}, {'type': 'xy'}],
                [{'type': 'scene'}, {'type': 'xy'}]
            ],
            subplot_titles=[
                "Entropy Evolution",
                "Method Comparison",
                "Quantum State Distribution",
                "Entropy Reduction"
            ]
        )
        
        # Entropy evolution over chunks
        chunks = range(len(results['classical']))
        for method in results:
            fig.add_trace(
                go.Scatter(
                    x=list(chunks),
                    y=results[method],
                    name=method.replace('_', ' ').title()
                ),
                row=1, col=1
            )
        
        # Method comparison (boxplot)
        fig.add_trace(
            go.Box(
                y=[np.mean(results[m]) for m in results],
                x=list(results.keys()),
                name="Method Comparison"
            ),
            row=1, col=2
        )
        
        # 3D quantum state visualization
        # Use first chunk's quantum state
        state = self.create_quantum_state("test", use_phase=True)
        x = np.real(state)
        y = np.imag(state)
        z = np.abs(state)
        
        fig.add_trace(
            go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(size=10),
                name="Quantum States"
            ),
            row=2, col=1
        )
        
        # Entropy reduction percentages
        classical_mean = np.mean(results['classical'])
        reductions = {
            method: (classical_mean - np.mean(values)) / classical_mean * 100
            for method, values in results.items()
        }
        
        fig.add_trace(
            go.Bar(
                x=list(reductions.keys()),
                y=list(reductions.values()),
                name="Entropy Reduction %"
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Quantum vs Classical Entropy Analysis",
            height=1000,
            showlegend=True
        )
        
        # Save visualizations
        fig.write_html('docs/images/entropy_analysis/entropy_analysis.html')
        fig.write_image('docs/images/entropy_analysis/entropy_analysis.png')
        
        return reductions

# Test the analyzer
if __name__ == "__main__":
    sample_text = """
    It is a truth universally acknowledged, that a single man in possession 
    of a good fortune, must be in want of a wife. However little known the 
    feelings or views of such a man may be on his first entering a 
    neighbourhood, this truth is so well fixed in the minds of the 
    surrounding families, that he is considered the rightful property of 
    some one or other of their daughters.
    """
    
    analyzer = QuantumEntropyAnalyzer()
    results = analyzer.analyze_entropy_reduction(sample_text)
    reductions = analyzer.visualize_entropy_analysis(results)
    
    print("\nEntropy Reduction Results:")
    for method, reduction in reductions.items():
        print(f"{method}: {reduction:.2f}% reduction") 