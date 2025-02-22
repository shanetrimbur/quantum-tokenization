import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.fft import fft, ifft
import os

class HarmonicCompressionVisualizer:
    def __init__(self):
        os.makedirs('docs/images/harmonic_compression', exist_ok=True)
        
    def visualize_frequency_mapping(self):
        """Visualize mapping of data patterns to harmonic frequency space"""
        # Generate sample data with repeating patterns
        t = np.linspace(0, 10, 1000)
        data = np.zeros_like(t)
        
        # Add some "data patterns" as frequency components
        patterns = [
            (1.0, 440),   # Pattern A maps to A4
            (0.5, 554),   # Pattern B maps to C#5
            (0.3, 659),   # Pattern C maps to E5
            (0.2, 880)    # Pattern D maps to A5
        ]
        
        frames = []
        for phase in np.linspace(0, 2*np.pi, 60):
            # Create time-domain signal
            signal = np.zeros_like(t)
            for amp, freq in patterns:
                signal += amp * np.sin(2*np.pi*freq*t + phase)
            
            # Compute frequency spectrum
            spectrum = np.abs(fft(signal))
            freqs = np.fft.fftfreq(len(t), t[1]-t[0])
            
            fig = make_subplots(rows=2, cols=1,
                              subplot_titles=["Time Domain (Data Patterns)",
                                            "Frequency Domain (Harmonic Mapping)"])
            
            # Time domain plot
            fig.add_trace(
                go.Scatter(x=t[:100], y=signal[:100], mode='lines',
                          name='Data Pattern'),
                row=1, col=1
            )
            
            # Frequency domain plot
            fig.add_trace(
                go.Scatter(x=freqs[:len(freqs)//2], y=spectrum[:len(freqs)//2],
                          mode='lines', name='Harmonic Structure'),
                row=2, col=1
            )
            
            frames.append(go.Frame(data=fig.data))
        
        # Create animated figure
        fig = go.Figure(
            data=frames[0].data,
            layout=go.Layout(
                title="Harmonic Data Pattern Mapping",
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [{
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': 50}}]
                    }]
                }],
                height=800
            ),
            frames=frames
        )
        
        fig.write_html('docs/images/harmonic_compression/frequency_mapping.html')
        fig.write_image('docs/images/harmonic_compression/frequency_mapping.png')

    def visualize_cymatic_encoding(self):
        """Visualize data encoding using cymatic patterns"""
        size = 100
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, y)
        
        frames = []
        # Initialize fixed data points
        data_points = np.random.rand(20, 2) * 2 - 1
        
        for t in np.linspace(0, 2*np.pi, 60):
            # Create evolving cymatic pattern
            Z = np.sin(5*np.pi*X + t) * np.sin(5*np.pi*Y + t) + \
                0.5 * np.sin(7*np.pi*X - t) * np.sin(7*np.pi*Y - t)
            
            # Calculate gradient for attraction
            grad_x, grad_y = np.gradient(Z)
            
            # Update point positions based on local gradient
            attracted_points = np.zeros_like(data_points)
            for i, point in enumerate(data_points):
                # Map point coordinates to grid indices
                x_idx = int((point[0] + 1) * (size-1) / 2)
                y_idx = int((point[1] + 1) * (size-1) / 2)
                
                # Bound indices to valid range
                x_idx = np.clip(x_idx, 0, size-1)
                y_idx = np.clip(y_idx, 0, size-1)
                
                # Apply attraction based on local gradient
                attracted_points[i] = point + 0.1 * np.array([
                    grad_x[y_idx, x_idx],
                    grad_y[y_idx, x_idx]
                ])
            
            fig = go.Figure()
            
            # Add cymatic surface
            fig.add_trace(go.Surface(
                z=Z,
                colorscale='Viridis',
                opacity=0.8
            ))
            
            # Add data points
            fig.add_trace(go.Scatter3d(
                x=attracted_points[:,0],
                y=attracted_points[:,1],
                z=Z[np.clip(np.round((attracted_points[:,1] + 1) * (size-1) / 2).astype(int), 0, size-1),
                    np.clip(np.round((attracted_points[:,0] + 1) * (size-1) / 2).astype(int), 0, size-1)],
                mode='markers',
                marker=dict(size=5, color='red')
            ))
            
            frames.append(go.Frame(data=fig.data))
        
        fig = go.Figure(
            data=frames[0].data,
            layout=go.Layout(
                title="Cymatic Data Encoding",
                scene=dict(
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [{
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': 50}}]
                    }]
                }]
            ),
            frames=frames
        )
        
        fig.write_html('docs/images/harmonic_compression/cymatic_encoding.html')
        fig.write_image('docs/images/harmonic_compression/cymatic_encoding.png')

    def visualize_quantum_harmonic_encoding(self):
        """Visualize quantum harmonic oscillator-based encoding"""
        x = np.linspace(-4, 4, 100)
        y = np.linspace(-4, 4, 100)
        X, Y = np.meshgrid(x, y)
        
        frames = []
        for t in np.linspace(0, 2*np.pi, 60):
            # Create quantum harmonic oscillator states
            psi_0 = np.exp(-(X**2 + Y**2)/2) / np.sqrt(np.pi)  # Ground state
            psi_1 = psi_0 * X * np.sqrt(2)  # First excited state
            psi_2 = psi_0 * (2*X**2 - 1) / np.sqrt(2)  # Second excited state
            
            # Create superposition
            psi = (psi_0 + np.exp(1j*t)*psi_1 + np.exp(2j*t)*psi_2) / np.sqrt(3)
            probability = np.abs(psi)**2
            
            fig = go.Figure(data=[
                go.Surface(
                    z=probability,
                    colorscale='Viridis',
                    opacity=0.8
                )
            ])
            
            frames.append(go.Frame(data=fig.data))
        
        fig = go.Figure(
            data=frames[0].data,
            layout=go.Layout(
                title="Quantum Harmonic Encoding",
                scene=dict(
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [{
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': 50}}]
                    }]
                }]
            ),
            frames=frames
        )
        
        fig.write_html('docs/images/harmonic_compression/quantum_harmonic.html')
        fig.write_image('docs/images/harmonic_compression/quantum_harmonic.png')

# Generate visualizations
visualizer = HarmonicCompressionVisualizer()
visualizer.visualize_frequency_mapping()
visualizer.visualize_cymatic_encoding()
visualizer.visualize_quantum_harmonic_encoding() 