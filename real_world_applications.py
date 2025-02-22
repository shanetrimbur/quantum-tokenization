import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.fft import fft, ifft
import os
import requests
from PIL import Image
from io import BytesIO
import nltk
from nltk.corpus import reuters
import numpy as np

class RealWorldCompression:
    def __init__(self):
        """Initialize with directories for visualizations"""
        os.makedirs('docs/images/applications', exist_ok=True)
        np.random.seed(42)
        
        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            nltk.download('reuters')

    def visualize_text_compression(self):
        """Visualize harmonic compression on text data"""
        # Use a simple text sample instead of Reuters corpus
        text = """
        Quantum computing is an emerging technology that leverages quantum mechanics 
        to solve complex computational problems. Unlike classical computers that use 
        bits, quantum computers use quantum bits or qubits. These qubits can exist 
        in multiple states simultaneously, thanks to quantum superposition.
        """
        
        # Convert text to numerical values (using ASCII)
        text_values = np.array([ord(c) for c in text])
        
        frames = []
        for compression_level in range(10, 90, 10):
            # Apply FFT
            fft_result = fft(text_values)
            
            # Keep only top % components
            n_components = int(len(fft_result) * (compression_level/100))
            mask = np.zeros_like(fft_result)
            sorted_indices = np.argsort(np.abs(fft_result))[::-1]
            mask[sorted_indices[:n_components]] = 1
            
            compressed_fft = fft_result * mask
            reconstructed = np.real(ifft(compressed_fft)).astype(int)
            
            # Create visualization frame
            fig = make_subplots(rows=2, cols=1,
                              subplot_titles=[
                                  "Original vs Reconstructed Text Signal",
                                  "Frequency Components"
                              ])
            
            # Signal comparison
            fig.add_trace(
                go.Scatter(y=text_values, name="Original",
                          line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(y=reconstructed, name="Reconstructed",
                          line=dict(color='red')),
                row=1, col=1
            )
            
            # Frequency spectrum
            fig.add_trace(
                go.Bar(y=np.abs(compressed_fft[:len(compressed_fft)//2]),
                       name="Frequency Components"),
                row=2, col=1
            )
            
            fig.update_layout(
                title=f"Text Compression (Using {compression_level}% of frequencies)",
                height=800
            )
            
            frames.append(go.Frame(data=fig.data, layout=fig.layout))
        
        # Create animated figure
        fig = go.Figure(
            data=frames[0].data,
            layout=frames[0].layout,
            frames=frames
        )
        
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [{
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {'frame': {'duration': 1000}}]
                }]
            }]
        )
        
        fig.write_html('docs/images/applications/text_compression.html')
        fig.write_image('docs/images/applications/text_compression.png')

    def visualize_neural_harmonic_learning(self):
        """Visualize neural network learning harmonic patterns"""
        # Simulate neural network learning process
        t = np.linspace(0, 1, 1000)
        epochs = np.linspace(0, 10, 50)
        
        frames = []
        for epoch in epochs:
            # Simulate improving approximation of harmonic structure
            learned_signal = np.sin(2*np.pi*t) + \
                           (1/(epoch+1))*np.sin(5*2*np.pi*t) + \
                           (1/(epoch+1))*np.random.randn(len(t))
            
            target_signal = np.sin(2*np.pi*t)
            
            fig = make_subplots(rows=2, cols=1,
                              subplot_titles=[
                                  "Neural Network Learning Process",
                                  "Error Over Time"
                              ])
            
            # Signal comparison
            fig.add_trace(
                go.Scatter(x=t, y=target_signal,
                          name="Target Pattern",
                          line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=t, y=learned_signal,
                          name="Learned Pattern",
                          line=dict(color='red')),
                row=1, col=1
            )
            
            # Learning error
            error = np.abs(target_signal - learned_signal).mean()
            fig.add_trace(
                go.Scatter(x=epochs[:int(epoch)+1],
                          y=[1/(e+1) for e in range(int(epoch)+1)],
                          name="Learning Error"),
                row=2, col=1
            )
            
            fig.update_layout(
                title=f"Neural Harmonic Learning (Epoch {int(epoch)})",
                height=800
            )
            
            frames.append(go.Frame(data=fig.data, layout=fig.layout))
        
        fig = go.Figure(
            data=frames[0].data,
            layout=frames[0].layout,
            frames=frames
        )
        
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [{
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {'frame': {'duration': 100}}]
                }]
            }]
        )
        
        fig.write_html('docs/images/applications/neural_learning.html')
        fig.write_image('docs/images/applications/neural_learning.png')

    def visualize_quantum_enhanced_compression(self):
        """Visualize quantum-enhanced harmonic compression"""
        # Simulate quantum state space
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x, y)
        
        frames = []
        for t in np.linspace(0, 2*np.pi, 60):
            # Simulate quantum superposition of harmonic states
            psi_1 = np.exp(-(X**2 + Y**2)/2) * np.exp(1j*t)
            psi_2 = (X + 1j*Y) * np.exp(-(X**2 + Y**2)/2) * np.exp(2j*t)
            psi = (psi_1 + psi_2)/np.sqrt(2)
            
            probability = np.abs(psi)**2
            
            fig = go.Figure(data=[
                go.Surface(
                    z=probability,
                    colorscale='Viridis',
                    opacity=0.8
                )
            ])
            
            fig.update_layout(
                title="Quantum-Enhanced Harmonic Compression",
                scene=dict(
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                )
            )
            
            frames.append(go.Frame(data=fig.data))
        
        fig = go.Figure(
            data=frames[0].data,
            layout=go.Layout(
                title="Quantum-Enhanced Compression States",
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [{
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': 50}}]
                    }]
                }],
                scene=dict(
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                )
            ),
            frames=frames
        )
        
        fig.write_html('docs/images/applications/quantum_enhanced.html')
        fig.write_image('docs/images/applications/quantum_enhanced.png')

# Generate visualizations
applications = RealWorldCompression()
applications.visualize_text_compression()
applications.visualize_neural_harmonic_learning()
applications.visualize_quantum_enhanced_compression() 