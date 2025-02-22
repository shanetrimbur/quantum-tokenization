import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

class GeometricHarmonyVisualizer:
    def __init__(self):
        """Initialize visualization system"""
        os.makedirs('docs/images/geometric_harmony', exist_ok=True)
        self.notes = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F']
        
    def create_circle_of_fifths_geometry(self):
        """Visualize geometric relationships in Circle of Fifths"""
        # Generate points for 12-sided polygon (dodecagon)
        theta = np.linspace(0, 2*np.pi, 12, endpoint=False)
        r = 1
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        frames = []
        # Animate different geometric relationships
        for t in np.linspace(0, 2*np.pi, 60):
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'xy'}, {'type': 'surface'}]],
                subplot_titles=["Circle of Fifths Geometry", "Harmonic Surface"]
            )
            
            # Basic circle with notes
            fig.add_trace(
                go.Scatter(
                    x=x, y=y,
                    mode='markers+text+lines',
                    text=self.notes,
                    textposition="top center",
                    name='Circle of Fifths'
                ),
                row=1, col=1
            )
            
            # Add geometric relationships
            for i in range(12):
                # Draw lines between perfect fifths
                fifth_idx = (i + 7) % 12
                fig.add_trace(
                    go.Scatter(
                        x=[x[i], x[fifth_idx]],
                        y=[y[i], y[fifth_idx]],
                        mode='lines',
                        line=dict(
                            color='rgba(100,100,255,0.2)',
                            width=1
                        ),
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            # Create harmonic surface
            X, Y = np.meshgrid(np.linspace(-1, 1, 50), np.linspace(-1, 1, 50))
            Z = np.sin(5*X + t) * np.cos(5*Y + t) * np.exp(-(X**2 + Y**2))
            
            fig.add_trace(
                go.Surface(
                    x=X, y=Y, z=Z,
                    colorscale='Viridis',
                    showscale=False
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                title="Geometric Harmony in Music and Data",
                height=600,
                width=1200,
                showlegend=False
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
                        'args': [None, {'frame': {'duration': 100}}]
                    }]
                }],
                scene=dict(
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                )
            ),
            frames=frames
        )
        
        fig.write_html('docs/images/geometric_harmony/circle_geometry.html')
        fig.write_image('docs/images/geometric_harmony/circle_geometry.png')

    def visualize_harmonic_symmetries(self):
        """Visualize symmetries between music, geometry, and data"""
        # Generate data points
        n_points = 12
        theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        
        frames = []
        for phase in np.linspace(0, 2*np.pi, 60):
            # Create different geometric transformations
            r1 = 1 + 0.3*np.sin(5*theta + phase)
            r2 = 1 + 0.3*np.cos(7*theta + phase)
            
            x1, y1 = r1*np.cos(theta), r1*np.sin(theta)
            x2, y2 = r2*np.cos(theta), r2*np.sin(theta)
            
            fig = make_subplots(
                rows=2, cols=2,
                specs=[
                    [{'type': 'xy'}, {'type': 'xy'}],
                    [{'type': 'surface', 'colspan': 2}, None]
                ],
                subplot_titles=[
                    "Harmonic Pattern 1",
                    "Harmonic Pattern 2",
                    "Combined Harmonic Surface"
                ]
            )
            
            # Pattern 1
            fig.add_trace(
                go.Scatter(
                    x=x1, y=y1,
                    mode='lines+markers',
                    name='Pattern 1'
                ),
                row=1, col=1
            )
            
            # Pattern 2
            fig.add_trace(
                go.Scatter(
                    x=x2, y=y2,
                    mode='lines+markers',
                    name='Pattern 2'
                ),
                row=1, col=2
            )
            
            # Combined surface
            X, Y = np.meshgrid(np.linspace(-1.5, 1.5, 50), np.linspace(-1.5, 1.5, 50))
            Z = (np.sin(5*X + phase) * np.cos(5*Y) + 
                 np.cos(7*X) * np.sin(7*Y + phase)) * np.exp(-(X**2 + Y**2)/4)
            
            fig.add_trace(
                go.Surface(
                    x=X, y=Y, z=Z,
                    colorscale='Viridis',
                    showscale=False
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                title="Harmonic Symmetries in Music and Data",
                height=900,
                width=1200,
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
                        'args': [None, {'frame': {'duration': 100}}]
                    }]
                }]
            ),
            frames=frames
        )
        
        fig.write_html('docs/images/geometric_harmony/harmonic_symmetries.html')
        fig.write_image('docs/images/geometric_harmony/harmonic_symmetries.png')

# Generate visualizations
visualizer = GeometricHarmonyVisualizer()
visualizer.create_circle_of_fifths_geometry()
visualizer.visualize_harmonic_symmetries() 