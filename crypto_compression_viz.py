import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

class CryptoCompressionVisualizer:
    def __init__(self):
        """Initialize visualization system"""
        os.makedirs('docs/images/crypto_compression', exist_ok=True)
        np.random.seed(42)

    def modular_exponentiation(self, data, base, modulus):
        """Apply modular exponentiation to data"""
        return np.array([pow(int(x), base, modulus) for x in data])

    def visualize_crypto_compression(self):
        """Visualize cryptographic compression process"""
        # Generate sample data
        data_length = 100
        original_data = np.random.randint(0, 256, data_length)
        
        frames = []
        for modulus in range(7, 23, 2):  # Try different moduli
            # Apply modular transformations
            mod_data = original_data % modulus
            crypto_data = self.modular_exponentiation(mod_data, base=3, modulus=modulus)
            
            # Create harmonic wave representation
            t = np.linspace(0, 2*np.pi, data_length)
            harmonic_wave = np.sin(2 * np.pi * crypto_data / modulus)
            
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=[
                    "Original vs Modular Data",
                    f"Cryptographic Transform (mod {modulus})",
                    "Harmonic Wave Encoding"
                ]
            )
            
            # Original and modular data
            fig.add_trace(
                go.Scatter(y=original_data, name="Original",
                          mode='lines+markers'),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(y=mod_data, name="Modular",
                          mode='lines+markers'),
                row=1, col=1
            )
            
            # Cryptographic transformation
            fig.add_trace(
                go.Bar(y=crypto_data, name="Crypto Transform"),
                row=2, col=1
            )
            
            # Harmonic encoding
            fig.add_trace(
                go.Scatter(x=t, y=harmonic_wave, name="Harmonic"),
                row=3, col=1
            )
            
            fig.update_layout(height=900)
            frames.append(go.Frame(data=fig.data, layout=fig.layout))
        
        fig = go.Figure(
            data=frames[0].data,
            layout=go.Layout(
                title="Cryptographic Compression Process",
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
        
        fig.write_html('docs/images/crypto_compression/process.html')
        fig.write_image('docs/images/crypto_compression/process.png')

    def visualize_geometric_crypto(self):
        """Visualize geometric cryptographic patterns"""
        size = 50
        data = np.random.randint(0, 256, (size, size))
        
        frames = []
        for modulus in range(7, 23, 2):
            # Apply cryptographic transformation
            mod_data = data % modulus
            crypto_data = np.array([[pow(int(x), 3, modulus) for x in row] 
                                  for row in mod_data])
            
            # Create geometric pattern
            X, Y = np.meshgrid(np.linspace(-2, 2, size),
                             np.linspace(-2, 2, size))
            Z = np.sin(2*np.pi*crypto_data/modulus) * \
                np.exp(-(X**2 + Y**2)/4)
            
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'surface'}, {'type': 'surface'}]],
                subplot_titles=[
                    "Cryptographic Pattern",
                    "Harmonic Surface"
                ]
            )
            
            # Cryptographic pattern
            fig.add_trace(
                go.Surface(z=crypto_data, colorscale='Viridis'),
                row=1, col=1
            )
            
            # Harmonic surface
            fig.add_trace(
                go.Surface(z=Z, colorscale='Viridis'),
                row=1, col=2
            )
            
            fig.update_layout(
                title=f"Geometric Cryptographic Patterns (mod {modulus})",
                height=600,
                scene=dict(
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                scene2=dict(
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                )
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
                }]
            ),
            frames=frames
        )
        
        fig.write_html('docs/images/crypto_compression/geometric.html')
        fig.write_image('docs/images/crypto_compression/geometric.png')

# Generate visualizations
visualizer = CryptoCompressionVisualizer()
visualizer.visualize_crypto_compression()
visualizer.visualize_geometric_crypto() 