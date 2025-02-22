import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.fft import fft, ifft
import os

class ModularHarmonicVisualizer:
    def __init__(self):
        """Initialize visualization system"""
        os.makedirs('docs/images/modular_harmonic', exist_ok=True)
        np.random.seed(42)

    def visualize_modular_circle(self):
        """Create animated visualization of modular circle transformations"""
        # Generate points on circle
        theta = np.linspace(0, 2*np.pi, 12)
        r = 1
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Musical notes for Circle of Fifths
        notes = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F']
        
        frames = []
        # Animate different modular transformations
        for t in np.linspace(0, 2*np.pi, 60):
            # Apply modular transformation
            x_transformed = x * np.cos(t) - y * np.sin(t)
            y_transformed = x * np.sin(t) + y * np.cos(t)
            
            fig = go.Figure()
            
            # Original circle
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines+markers+text',
                text=notes,
                name='Original Circle',
                line=dict(color='blue', width=2),
                textposition="top center"
            ))
            
            # Transformed circle
            fig.add_trace(go.Scatter(
                x=x_transformed, y=y_transformed,
                mode='lines+markers',
                name='Modular Transform',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title="Modular Circle Transformations",
                xaxis_range=[-1.5, 1.5],
                yaxis_range=[-1.5, 1.5],
                showlegend=True
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
                        'args': [None, {'frame': {'duration': 50}}]
                    }]
                }]
            ),
            frames=frames
        )
        
        fig.write_html('docs/images/modular_harmonic/modular_circle.html')
        fig.write_image('docs/images/modular_harmonic/modular_circle.png')

    def visualize_harmonic_cryptography(self):
        """Visualize harmonic-cryptographic encoding process"""
        # Generate sample data
        t = np.linspace(0, 1, 1000)
        data = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*10*t)
        
        frames = []
        for mod_n in range(2, 20):
            # Apply modular transformation
            modular_data = np.mod(data * mod_n, 1)
            
            # Compute FFT
            fft_result = fft(modular_data)
            frequencies = np.fft.fftfreq(len(t), t[1]-t[0])
            
            fig = make_subplots(rows=2, cols=1,
                              subplot_titles=[
                                  f"Modular Transform (mod {mod_n})",
                                  "Frequency Components"
                              ])
            
            # Modular transformation
            fig.add_trace(
                go.Scatter(x=t, y=modular_data,
                          name='Modular Signal'),
                row=1, col=1
            )
            
            # Frequency spectrum
            fig.add_trace(
                go.Bar(x=frequencies[:len(frequencies)//2],
                       y=np.abs(fft_result)[:len(frequencies)//2],
                       name='Harmonic Components'),
                row=2, col=1
            )
            
            fig.update_layout(height=600)
            frames.append(go.Frame(data=fig.data, layout=fig.layout))
        
        fig = go.Figure(
            data=frames[0].data,
            layout=go.Layout(
                title="Harmonic-Cryptographic Encoding",
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
        
        fig.write_html('docs/images/modular_harmonic/crypto_encoding.html')
        fig.write_image('docs/images/modular_harmonic/crypto_encoding.png')

    def visualize_modular_wavelet(self):
        """Visualize modular wavelet transformations"""
        # Generate sample signal
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2*np.pi*10*t) * np.exp(-5*t)
        
        frames = []
        for scale in np.linspace(0.1, 2, 50):
            # Apply wavelet-like transform with modular scaling
            scaled_t = np.mod(t * scale, 1)
            transformed = signal * np.sin(2*np.pi*scaled_t)
            
            fig = make_subplots(rows=1, cols=2,
                              subplot_titles=[
                                  "Original Signal",
                                  f"Modular Wavelet (scale={scale:.2f})"
                              ])
            
            # Original signal
            fig.add_trace(
                go.Scatter(x=t, y=signal,
                          name='Original'),
                row=1, col=1
            )
            
            # Transformed signal
            fig.add_trace(
                go.Scatter(x=t, y=transformed,
                          name='Transformed'),
                row=1, col=2
            )
            
            fig.update_layout(height=400)
            frames.append(go.Frame(data=fig.data, layout=fig.layout))
        
        fig = go.Figure(
            data=frames[0].data,
            layout=go.Layout(
                title="Modular Wavelet Transformations",
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
        
        fig.write_html('docs/images/modular_harmonic/modular_wavelet.html')
        fig.write_image('docs/images/modular_harmonic/modular_wavelet.png')

# Generate visualizations
visualizer = ModularHarmonicVisualizer()
visualizer.visualize_modular_circle()
visualizer.visualize_harmonic_cryptography()
visualizer.visualize_modular_wavelet() 