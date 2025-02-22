import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.fft import fft, fftfreq, fft2
import scipy.signal as signal
import pywt
from mpl_toolkits.mplot3d import Axes3D
import os

class UnifiedWaveAnalyzer:
    def __init__(self):
        os.makedirs('docs/images/unified', exist_ok=True)
        self.sample_rate = 44100
        self.duration = 2.0

    def generate_chladni_pattern(self, n, m, size=100):
        """Generate Chladni plate vibration pattern"""
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, y)
        
        # Chladni equation for rectangular plate
        Z = np.sin(n * np.pi * X) * np.sin(m * np.pi * Y) - \
            np.sin(m * np.pi * X) * np.sin(n * np.pi * Y)
        return X, Y, Z

    def visualize_chladni_patterns(self):
        """Create animated Chladni patterns"""
        patterns = []
        modes = [(2,3), (3,3), (4,4), (5,5)]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f"Mode ({n},{m})" for n,m in modes]
        )
        
        for idx, (n, m) in enumerate(modes, 1):
            X, Y, Z = self.generate_chladni_pattern(n, m)
            row = (idx-1) // 2 + 1
            col = (idx-1) % 2 + 1
            
            fig.add_trace(
                go.Heatmap(
                    z=Z,
                    colorscale='Viridis',
                    showscale=False
                ),
                row=row, col=col
            )

        fig.update_layout(
            title="Chladni Patterns: Standing Wave Geometries",
            height=800,
            width=1000
        )

        fig.write_html('docs/images/unified/chladni_patterns.html')
        fig.write_image('docs/images/unified/chladni_patterns.png')

    def morlet_wavelet(self, t, w0=5.0):
        """Generate a Morlet wavelet"""
        return np.exp(-t**2/2) * np.cos(w0*t)

    def create_wavelet_decomposition(self, signal_type='music'):
        """Visualize wavelet decomposition of different signals"""
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        
        if signal_type == 'music':
            # Generate a complex musical signal
            wave_data = np.sin(2 * np.pi * 440 * t) + \
                    0.5 * np.sin(2 * np.pi * 880 * t) + \
                    0.25 * np.sin(2 * np.pi * 1320 * t)
        else:
            # Generate light-like wave packet using our Morlet function
            t_wavelet = np.linspace(-4, 4, len(t))  # Normalized time for wavelet
            wave_data = self.morlet_wavelet(t_wavelet)

        # Compute continuous wavelet transform using pywt
        scales = np.arange(1, 31)
        cwtmatr, freqs = pywt.cwt(wave_data, scales, 'morl')

        # Create visualization
        fig = go.Figure()

        fig.add_trace(
            go.Heatmap(
                z=np.abs(cwtmatr),
                colorscale='Viridis',
                name='Wavelet Transform'
            )
        )

        fig.update_layout(
            title=f"Wavelet Decomposition: {signal_type.capitalize()}",
            xaxis_title="Time",
            yaxis_title="Scale",
            height=600,
            width=1000
        )

        fig.write_html(f'docs/images/unified/wavelet_{signal_type}.html')
        fig.write_image(f'docs/images/unified/wavelet_{signal_type}.png')

    def visualize_quantum_encoding(self):
        """Visualize quantum state encoding of wave patterns"""
        # Generate sample data
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x, y)
        
        # Create quantum-like wavefunction
        psi = np.exp(-(X**2 + Y**2)) * (X + 1j*Y)
        probability = np.abs(psi)**2
        
        # Create animated visualization
        frames = []
        for phase in np.linspace(0, 2*np.pi, 60):
            rotated_psi = psi * np.exp(1j * phase)
            frames.append(
                go.Frame(
                    data=[go.Surface(
                        z=np.abs(rotated_psi),
                        colorscale='Viridis'
                    )]
                )
            )

        fig = go.Figure(
            data=frames[0].data,
            layout=go.Layout(
                title="Quantum State Encoding",
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [{
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 50},
                            'fromcurrent': True,
                            'mode': 'immediate'
                        }]
                    }]
                }],
                scene=dict(
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                )
            ),
            frames=frames
        )

        fig.write_html('docs/images/unified/quantum_encoding.html')
        fig.write_image('docs/images/unified/quantum_encoding.png')

    def animate_chladni_evolution(self):
        """Create animated evolution of Chladni patterns"""
        frames = []
        n_range = np.linspace(2, 5, 60)
        m_range = np.linspace(2, 5, 60)
        
        for n, m in zip(n_range, m_range):
            X, Y, Z = self.generate_chladni_pattern(n, m)
            frames.append(
                go.Frame(
                    data=[go.Surface(
                        z=Z,
                        colorscale='Viridis'
                    )]
                )
            )

        fig = go.Figure(
            data=frames[0].data,
            layout=go.Layout(
                title="Evolving Chladni Patterns",
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [{
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 50},
                            'fromcurrent': True,
                            'mode': 'immediate'
                        }]
                    }]
                }],
                scene=dict(
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                )
            ),
            frames=frames
        )
        
        fig.write_html('docs/images/unified/animated_chladni.html')

    def animate_wavelet_music_light_comparison(self):
        """Create side-by-side animated comparison of music and light wavelets"""
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        frames = []
        
        for phase in np.linspace(0, 2*np.pi, 60):
            # Music signal with phase modulation
            music_signal = np.sin(2 * np.pi * 440 * t + phase) + \
                         0.5 * np.sin(2 * np.pi * 880 * t + 2*phase)
            
            # Light-like signal with phase modulation
            light_signal = self.morlet_wavelet(t) * np.exp(1j * phase)
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=["Musical Harmonics", "Light Wave Packet"]
            )
            
            fig.add_trace(
                go.Scatter(x=t[:1000], y=music_signal[:1000], mode='lines'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=t[:1000], y=np.real(light_signal[:1000]), mode='lines'),
                row=1, col=2
            )
            
            frames.append(go.Frame(data=fig.data))

        fig = go.Figure(
            data=frames[0].data,
            layout=go.Layout(
                title="Music and Light Wave Comparison",
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [{
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 50},
                            'fromcurrent': True,
                            'mode': 'immediate'
                        }]
                    }]
                }],
                height=400,
                width=1000
            ),
            frames=frames
        )
        
        fig.write_html('docs/images/unified/animated_wave_comparison.html')

# Generate visualizations
analyzer = UnifiedWaveAnalyzer()
analyzer.visualize_chladni_patterns()
analyzer.create_wavelet_decomposition('music')
analyzer.create_wavelet_decomposition('light')
analyzer.visualize_quantum_encoding()
analyzer.animate_chladni_evolution()
analyzer.animate_wavelet_music_light_comparison() 