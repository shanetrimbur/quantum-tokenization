import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

class QuantumCircuitVisualizer:
    def __init__(self):
        """Initialize visualization system"""
        os.makedirs('docs/images/quantum_circuits', exist_ok=True)
        np.random.seed(42)

    def simulate_quantum_gate(self, theta, phi):
        """Simulate quantum rotation gates"""
        # Create rotation matrices
        Ry = np.array([[np.cos(theta/2), -np.sin(theta/2)],
                      [np.sin(theta/2), np.cos(theta/2)]])
        Rz = np.array([[np.exp(-1j*phi/2), 0],
                      [0, np.exp(1j*phi/2)]])
        # Initial state |0⟩
        state = np.array([1, 0])
        # Apply gates
        state = Rz @ Ry @ state
        return state

    def visualize_quantum_encoding(self):
        """Visualize quantum circuit encoding process"""
        # Generate sample data
        data_length = 4
        data = np.random.randint(0, 256, data_length)
        modulus = 12
        mod_data = data % modulus
        
        frames = []
        for t in np.linspace(0, 2*np.pi, 60):
            # Create quantum states
            states = []
            for val in mod_data:
                theta = np.pi * val / modulus
                phi = 2*np.pi * val / modulus + t
                state = self.simulate_quantum_gate(theta, phi)
                states.append(state)
            
            fig = make_subplots(
                rows=2, cols=2,
                specs=[
                    [{'type': 'scene'}, {'type': 'xy'}],
                    [{'type': 'scene', 'colspan': 2}, None]
                ],
                subplot_titles=[
                    "Quantum States",
                    "Circuit Diagram",
                    "Entangled State Space"
                ]
            )
            
            # Plot individual states
            for i, state in enumerate(states):
                x = np.real(state[0])
                y = np.real(state[1])
                z = np.imag(state[1])
                
                fig.add_trace(
                    go.Scatter3d(
                        x=[0, x], y=[0, y], z=[0, z],
                        mode='lines+markers',
                        marker=dict(size=8),
                        name=f'Qubit {i}'
                    ),
                    row=1, col=1
                )
            
            # Circuit diagram representation
            fig.add_trace(
                go.Scatter(
                    x=[0, 1, 2, 3],
                    y=[0, 0, 0, 0],
                    mode='lines+markers+text',
                    text=['|0⟩', 'Ry', 'Rz', '|ψ⟩'],
                    marker=dict(size=10),
                    line=dict(width=2)
                ),
                row=1, col=2
            )
            
            # Entangled state visualization
            X, Y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
            Z = np.sin(5*X + t) * np.cos(5*Y + t) * np.exp(-(X**2 + Y**2))
            
            fig.add_trace(
                go.Surface(
                    x=X, y=Y, z=Z,
                    colorscale='Viridis',
                    showscale=False
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                title="Quantum Circuit Encoding Process",
                height=900,
                width=1200,
                scene=dict(
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                scene2=dict(
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
        
        fig.write_html('docs/images/quantum_circuits/encoding_process.html')
        fig.write_image('docs/images/quantum_circuits/encoding_process.png')

    def visualize_quantum_optimization(self):
        """Visualize quantum optimization of encoding"""
        # Generate sample data
        data_length = 50
        data = np.random.randint(0, 256, data_length)
        modulus = 12
        mod_data = data % modulus
        
        frames = []
        for t in np.linspace(0, 2*np.pi, 60):
            # Simulate optimization process
            optimized_states = []
            for val in mod_data:
                theta = np.pi * val / modulus
                phi = 2*np.pi * val / modulus
                # Add optimization term
                opt_factor = np.exp(-t/10)
                state = self.simulate_quantum_gate(theta * opt_factor, phi)
                optimized_states.append(state)
            
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'scene'}, {'type': 'xy'}]],
                subplot_titles=[
                    "Optimized Quantum States",
                    "Optimization Progress"
                ]
            )
            
            # Plot optimized states
            for state in optimized_states[:10]:  # Show first 10 states
                x = np.real(state[0])
                y = np.real(state[1])
                z = np.imag(state[1])
                
                fig.add_trace(
                    go.Scatter3d(
                        x=[0, x], y=[0, y], z=[0, z],
                        mode='lines',
                        line=dict(width=2)
                    ),
                    row=1, col=1
                )
            
            # Optimization progress
            fig.add_trace(
                go.Scatter(
                    x=np.linspace(0, t, 20),
                    y=np.exp(-np.linspace(0, t, 20)/10),
                    mode='lines',
                    name='Optimization'
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                title="Quantum State Optimization",
                height=600,
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
        
        fig.write_html('docs/images/quantum_circuits/optimization.html')
        fig.write_image('docs/images/quantum_circuits/optimization.png')

# Generate visualizations
visualizer = QuantumCircuitVisualizer()
visualizer.visualize_quantum_encoding()
visualizer.visualize_quantum_optimization() 