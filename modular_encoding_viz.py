import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.fft import fft, ifft
import os

class ModularEncodingVisualizer:
    def __init__(self):
        """Initialize visualization system"""
        os.makedirs('docs/images/modular_encoding', exist_ok=True)
        np.random.seed(42)

    def visualize_modular_encoding_process(self):
        """Create animated visualization of modular encoding process"""
        # Generate sample data
        data_length = 100
        data = np.random.randint(0, 256, data_length)
        
        frames = []
        for modulus in range(4, 24, 2):  # Try different moduli
            # Apply modular transformation
            mod_data = data % modulus
            
            # Generate harmonic encoding
            t = np.linspace(0, 2*np.pi, data_length)
            harmonic_wave = np.sin(2 * np.pi * mod_data / modulus)
            
            # Create visualization frame
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=[
                    "Original Byte Data",
                    f"Modular Encoding (mod {modulus})",
                    "Harmonic Wave Representation"
                ]
            )
            
            # Original data
            fig.add_trace(
                go.Bar(y=data, name="Original Bytes"),
                row=1, col=1
            )
            
            # Modular data
            fig.add_trace(
                go.Scatter(y=mod_data, mode='markers+lines',
                          name="Modular Values"),
                row=2, col=1
            )
            
            # Harmonic encoding
            fig.add_trace(
                go.Scatter(x=t, y=harmonic_wave,
                          name="Harmonic Encoding"),
                row=3, col=1
            )
            
            fig.update_layout(height=800)
            frames.append(go.Frame(data=fig.data, layout=fig.layout))
        
        fig = go.Figure(
            data=frames[0].data,
            layout=go.Layout(
                title="Modular Harmonic Encoding Process",
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [{
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': 500}}]
                    }]
                }]
            ),
            frames=frames
        )
        
        fig.write_html('docs/images/modular_encoding/encoding_process.html')
        fig.write_image('docs/images/modular_encoding/encoding_process.png')

    def visualize_modular_patterns(self):
        """Visualize emerging patterns in modular encoding"""
        # Generate 2D grid of data
        size = 50
        data = np.random.randint(0, 256, (size, size))
        
        frames = []
        for modulus in range(4, 24, 2):
            # Apply modular transformation
            mod_data = data % modulus
            
            # Create harmonic pattern
            X, Y = np.meshgrid(np.linspace(0, 2*np.pi, size),
                             np.linspace(0, 2*np.pi, size))
            pattern = np.sin(2*np.pi*mod_data/modulus) * \
                     np.cos(X) * np.sin(Y)
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=[
                    f"Modular Grid (mod {modulus})",
                    "Harmonic Pattern"
                ],
                specs=[[{'type': 'heatmap'}, {'type': 'surface'}]]
            )
            
            # Modular grid
            fig.add_trace(
                go.Heatmap(z=mod_data, colorscale='Viridis'),
                row=1, col=1
            )
            
            # Harmonic pattern
            fig.add_trace(
                go.Surface(z=pattern, colorscale='Viridis'),
                row=1, col=2
            )
            
            fig.update_layout(
                title=f"Modular Encoding Patterns (mod {modulus})",
                height=500
            )
            
            frames.append(go.Frame(data=fig.data, layout=fig.layout))
        
        fig = go.Figure(
            data=frames[0].data,
            layout=go.Layout(
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [{
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': 500}}]
                    }]
                }],
                scene=dict(
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                )
            ),
            frames=frames
        )
        
        fig.write_html('docs/images/modular_encoding/encoding_patterns.html')
        fig.write_image('docs/images/modular_encoding/encoding_patterns.png')

# Generate visualizations
visualizer = ModularEncodingVisualizer()
visualizer.visualize_modular_encoding_process()
visualizer.visualize_modular_patterns() 