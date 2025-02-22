import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from scipy.fft import fft, fftfreq
import math
import os
from plotly.subplots import make_subplots

# Create docs/images directory if it doesn't exist
os.makedirs('docs/images/harmonics', exist_ok=True)

class HarmonicAnalyzer:
    def __init__(self):
        self.fundamental_freq = 440  # A4 note
        self.sample_rate = 44100
        self.duration = 1.0
        
    def generate_harmonic_series(self, num_harmonics=8):
        """Generate time series for harmonic frequencies"""
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        series = {}
        
        for n in range(1, num_harmonics + 1):
            freq = n * self.fundamental_freq
            series[f'H{n}'] = np.sin(2 * np.pi * freq * t)
            
        return t, series
    
    def plot_circle_of_fifths_3d(self):
        """Create 3D spiral representation of the Circle of Fifths"""
        # Generate points for a helical Circle of Fifths
        theta = np.linspace(0, 2*np.pi, 12)  # 12 notes
        r = 1  # radius
        h = np.linspace(0, 2, 12)  # height variation
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = h
        
        notes = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F']
        
        # Create 3D plot using plotly
        fig = go.Figure(data=[
            go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers+lines+text',
                text=notes,
                marker=dict(
                    size=10,
                    color=z,
                    colorscale='Viridis',
                ),
                line=dict(color='rgba(100,100,100,0.5)', width=2),
                textposition="top center"
            )
        ])
        
        fig.update_layout(
            title="3D Helix of Fifths",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Octave Position"
            ),
            width=800,
            height=800
        )
        
        # Save visualization
        fig.write_html('docs/images/harmonics/helix_of_fifths.html')
        fig.write_image('docs/images/harmonics/helix_of_fifths.png')
        
    def plot_lissajous_patterns(self):
        """Generate Lissajous patterns for different frequency ratios"""
        t = np.linspace(0, 2*np.pi, 1000)
        ratios = [(1,1), (3,2), (2,1), (3,1)]  # Some common harmonic ratios
        
        fig = plt.figure(figsize=(15, 4))
        for i, (fx, fy) in enumerate(ratios, 1):
            ax = fig.add_subplot(1, 4, i)
            x = np.sin(fx * t)
            y = np.sin(fy * t)
            ax.plot(x, y)
            ax.set_title(f'Ratio {fx}:{fy}')
            ax.axis('equal')
            
        plt.tight_layout()
        plt.savefig('docs/images/harmonics/lissajous_patterns.png', dpi=300, bbox_inches='tight')
        
    def plot_quantum_harmonic_states(self, n_states=4):
        """Visualize quantum harmonic oscillator wavefunctions"""
        x = np.linspace(-4, 4, 200)
        
        def hermite(x, n):
            if n == 0:
                return np.ones_like(x)
            elif n == 1:
                return 2*x
            else:
                return 2*x*hermite(x,n-1) - 2*(n-1)*hermite(x,n-2)
        
        def psi(x, n):
            return (1/np.sqrt(2**n * math.factorial(n))) * \
                   hermite(x, n) * \
                   np.exp(-x**2/2)
        
        plt.figure(figsize=(10, 6))
        for n in range(n_states):
            plt.plot(x, psi(x, n) + n, label=f'n={n}')
            
        plt.title('Quantum Harmonic Oscillator States')
        plt.xlabel('Position')
        plt.ylabel('Wavefunction (offset for clarity)')
        plt.legend()
        plt.grid(True)
        plt.savefig('docs/images/harmonics/quantum_states.png', dpi=300, bbox_inches='tight')

    def animate_helix_of_fifths(self):
        """Create animated 3D spiral of the Circle of Fifths"""
        theta = np.linspace(0, 2*np.pi, 12)
        notes = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F']
        
        # Create frames for animation
        frames = []
        for t in np.linspace(0, 4*np.pi, 60):
            x = np.cos(theta)
            y = np.sin(theta)
            z = np.linspace(0, 2, 12) + np.sin(t)
            
            frames.append(
                go.Frame(
                    data=[go.Scatter3d(
                        x=x, y=y, z=z,
                        mode='markers+lines+text',
                        text=notes,
                        marker=dict(
                            size=10,
                            color=z,
                            colorscale='Viridis',
                        ),
                        line=dict(color='rgba(100,100,100,0.5)', width=2),
                        textposition="top center"
                    )]
                )
            )

        # Create initial figure
        fig = go.Figure(
            data=frames[0].data,
            layout=go.Layout(
                title="Animated Helix of Fifths",
                scene=dict(
                    camera=dict(
                        up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=0),
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    ),
                    xaxis_title="X",
                    yaxis_title="Y",
                    zaxis_title="Octave Position"
                ),
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [{
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 50, 'redraw': True},
                            'fromcurrent': True,
                            'mode': 'immediate',
                        }]
                    }]
                }],
                width=800,
                height=800
            ),
            frames=frames
        )

        # Save as HTML for interactive viewing
        fig.write_html('docs/images/harmonics/animated_helix.html')

    def animate_lissajous(self):
        """Create animated Lissajous patterns"""
        t = np.linspace(0, 10*np.pi, 1000)
        ratios = [(1,1), (3,2), (2,1), (3,1)]
        
        frames = []
        for phase in np.linspace(0, 2*np.pi, 60):
            fig = make_subplots(rows=1, cols=4, 
                              subplot_titles=[f'Ratio {fx}:{fy}' for fx, fy in ratios])
            
            for i, (fx, fy) in enumerate(ratios, 1):
                x = np.sin(fx * t)
                y = np.sin(fy * t + phase)
                
                fig.add_trace(
                    go.Scatter(x=x, y=y, mode='lines',
                              line=dict(color='blue', width=1)),
                    row=1, col=i
                )
                
                fig.update_xaxes(range=[-1.5, 1.5], row=1, col=i)
                fig.update_yaxes(range=[-1.5, 1.5], row=1, col=i)
            
            frames.append(go.Frame(data=fig.data))

        # Create the animated figure
        fig = go.Figure(
            data=frames[0].data,
            layout=go.Layout(
                title="Animated Lissajous Patterns",
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [{
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 50, 'redraw': True},
                            'fromcurrent': True,
                            'mode': 'immediate',
                        }]
                    }]
                }],
                width=1200,
                height=300
            ),
            frames=frames
        )

        fig.write_html('docs/images/harmonics/animated_lissajous.html')

# Generate visualizations
analyzer = HarmonicAnalyzer()
analyzer.plot_circle_of_fifths_3d()
analyzer.plot_lissajous_patterns()
analyzer.plot_quantum_harmonic_states()

# Generate animated visualizations
analyzer.animate_helix_of_fifths()
analyzer.animate_lissajous() 