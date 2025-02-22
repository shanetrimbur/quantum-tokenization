import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

class BlochEncodingVisualizer:
    def __init__(self):
        """Initialize visualization system"""
        os.makedirs('docs/images/bloch_encoding', exist_ok=True)
        np.random.seed(42)
        self.notes = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F']

    def visualize_circle_to_sphere(self):
        """Visualize transition from Circle of Fifths to Bloch sphere"""
        # Generate sample data
        data_length = 50
        data = np.random.randint(0, 256, data_length)
        modulus = 12
        mod_data = data % modulus
        
        frames = []
        for t in np.linspace(0, 2*np.pi, 60):
            # Create circle coordinates
            theta_circle = 2*np.pi*np.arange(12)/12
            x_circle = np.cos(theta_circle)
            y_circle = np.sin(theta_circle)
            
            # Create Bloch sphere coordinates
            theta = np.pi * mod_data / modulus
            phi = 2*np.pi * mod_data / modulus + t
            
            x_sphere = np.sin(theta) * np.cos(phi)
            y_sphere = np.sin(theta) * np.sin(phi)
            z_sphere = np.cos(theta)
            
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'xy'}, {'type': 'scene'}]],
                subplot_titles=[
                    "Circle of Fifths",
                    "Bloch Sphere Encoding"
                ]
            )
            
            # Circle visualization
            fig.add_trace(
                go.Scatter(
                    x=x_circle, y=y_circle,
                    mode='lines+markers+text',
                    text=self.notes,
                    textposition="top center",
                    name='Circle of Fifths'
                ),
                row=1, col=1
            )
            
            # Bloch sphere visualization
            fig.add_trace(
                go.Scatter3d(
                    x=x_sphere, y=y_sphere, z=z_sphere,
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=mod_data,
                        colorscale='Viridis'
                    ),
                    name='Quantum States'
                ),
                row=1, col=2
            )
            
            # Add sphere wireframe
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x_wire = np.cos(u)*np.sin(v)
            y_wire = np.sin(u)*np.sin(v)
            z_wire = np.cos(v)
            
            fig.add_trace(
                go.Surface(
                    x=x_wire, y=y_wire, z=z_wire,
                    opacity=0.2,
                    showscale=False
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                title="From Circle of Fifths to Bloch Sphere",
                height=600,
                width=1200,
                scene=dict(
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                )
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
        
        fig.write_html('docs/images/bloch_encoding/circle_to_sphere.html')
        fig.write_image('docs/images/bloch_encoding/circle_to_sphere.png')

    def visualize_quantum_transformations(self):
        """Visualize quantum transformations on encoded data"""
        # Generate data points
        data_length = 30
        data = np.random.randint(0, 256, data_length)
        modulus = 12
        mod_data = data % modulus
        
        frames = []
        for t in np.linspace(0, 4*np.pi, 60):
            # Apply quantum rotation
            theta = np.pi * mod_data / modulus
            phi = 2*np.pi * mod_data / modulus
            
            # Simulate quantum gates
            x_rot = np.sin(theta + t) * np.cos(phi)
            y_rot = np.sin(theta + t) * np.sin(phi)
            z_rot = np.cos(theta + t)
            
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'scene'}, {'type': 'scene'}]],
                subplot_titles=[
                    "Original Quantum States",
                    "Transformed States"
                ]
            )
            
            # Original states
            fig.add_trace(
                go.Scatter3d(
                    x=np.sin(theta)*np.cos(phi),
                    y=np.sin(theta)*np.sin(phi),
                    z=np.cos(theta),
                    mode='markers',
                    marker=dict(size=8, color=mod_data, colorscale='Viridis'),
                    name='Original'
                ),
                row=1, col=1
            )
            
            # Transformed states
            fig.add_trace(
                go.Scatter3d(
                    x=x_rot, y=y_rot, z=z_rot,
                    mode='markers',
                    marker=dict(size=8, color=mod_data, colorscale='Viridis'),
                    name='Transformed'
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                title="Quantum State Transformations",
                height=600,
                scene=dict(camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))),
                scene2=dict(camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)))
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
        
        fig.write_html('docs/images/bloch_encoding/quantum_transforms.html')
        fig.write_image('docs/images/bloch_encoding/quantum_transforms.png')

# Generate visualizations
visualizer = BlochEncodingVisualizer()
visualizer.visualize_circle_to_sphere()
visualizer.visualize_quantum_transformations() 