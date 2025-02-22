import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.visualization import plot_bloch_multivector
from qiskit.quantum_info import Statevector
import os
import matplotlib.pyplot as plt
from io import BytesIO

class QiskitCircuitVisualizer:
    def __init__(self):
        """Initialize visualization system"""
        os.makedirs('docs/images/qiskit_circuits', exist_ok=True)
        np.random.seed(42)
        self.backend = Aer.get_backend('statevector_simulator')

    def create_encoding_circuit(self, data_val, modulus=12):
        """Create quantum circuit for encoding a data value"""
        qc = QuantumCircuit(1)
        
        # Map data to angles
        theta = np.pi * (data_val % modulus) / modulus
        phi = 2 * np.pi * (data_val % modulus) / modulus
        
        # Apply rotation gates
        qc.ry(theta, 0)
        qc.rz(phi, 0)
        
        return qc

    def get_statevector(self, circuit):
        """Helper method to get statevector from circuit"""
        # Use the backend directly
        job = self.backend.run(circuit)
        result = job.result()
        return result.get_statevector()

    def visualize_qiskit_encoding(self):
        """Visualize Qiskit-based quantum encoding"""
        # Generate sample data
        data = np.random.randint(0, 256, 4)
        modulus = 12
        
        frames = []
        for t in np.linspace(0, 2*np.pi, 60):
            fig = make_subplots(
                rows=2, cols=2,
                specs=[
                    [{'type': 'scene'}, {'type': 'xy'}],
                    [{'type': 'xy', 'colspan': 2}, None]
                ],
                subplot_titles=[
                    "Quantum State Evolution",
                    "Circuit Diagram",
                    "State Amplitudes"
                ]
            )
            
            # Create and simulate circuits
            for i, val in enumerate(data):
                qc = self.create_encoding_circuit(val)
                # Add time-dependent phase
                qc.rz(t, 0)
                
                # Get statevector
                state = self.get_statevector(qc)
                
                # Plot state in 3D
                x = np.real(state[0])
                y = np.real(state[1])
                z = np.imag(state[1])
                
                fig.add_trace(
                    go.Scatter3d(
                        x=[0, x], y=[0, y], z=[0, z],
                        mode='lines+markers',
                        name=f'Qubit {i}'
                    ),
                    row=1, col=1
                )
            
            # Circuit representation
            fig.add_trace(
                go.Scatter(
                    x=[0, 1, 2, 3, 4],
                    y=[0, 0, 0, 0, 0],
                    mode='lines+markers+text',
                    text=['|0⟩', 'Ry', 'Rz', 'Rz(t)', '|ψ⟩'],
                    marker=dict(size=10),
                    line=dict(width=2)
                ),
                row=1, col=2
            )
            
            # State amplitudes
            amplitudes = [np.abs(self.get_statevector(self.create_encoding_circuit(val))) 
                        for val in data]
            
            fig.add_trace(
                go.Bar(
                    x=list(range(len(data))),
                    y=[amp[0] for amp in amplitudes],
                    name='|0⟩ amplitude'
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=list(range(len(data))),
                    y=[amp[1] for amp in amplitudes],
                    name='|1⟩ amplitude'
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                title="Qiskit Quantum Encoding Process",
                height=900,
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
        
        fig.write_html('docs/images/qiskit_circuits/encoding_process.html')
        fig.write_image('docs/images/qiskit_circuits/encoding_process.png')

    def visualize_quantum_entanglement(self):
        """Visualize quantum entanglement in encoding"""
        # Create entangled circuit
        qc = QuantumCircuit(2)
        qc.h(0)  # Hadamard gate
        qc.cx(0, 1)  # CNOT gate
        
        frames = []
        for t in np.linspace(0, 2*np.pi, 60):
            # Add time-dependent phase
            qc_t = qc.copy()
            qc_t.rz(t, 0)
            
            # Get statevector
            state = self.get_statevector(qc_t)
            
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'scene'}, {'type': 'xy'}]],
                subplot_titles=[
                    "Entangled State",
                    "Bell State Evolution"
                ]
            )
            
            # Plot entangled state
            prob_00 = np.abs(state[0])**2
            prob_11 = np.abs(state[3])**2
            
            X, Y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
            Z = prob_00 * np.sin(5*X + t) + prob_11 * np.cos(5*Y + t)
            
            fig.add_trace(
                go.Surface(
                    x=X, y=Y, z=Z,
                    colorscale='Viridis'
                ),
                row=1, col=1
            )
            
            # Plot state evolution
            fig.add_trace(
                go.Scatter(
                    x=[0, t],
                    y=[prob_00, prob_11],
                    mode='lines',
                    name='State Probabilities'
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                title="Quantum Entanglement in Encoding",
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
        
        fig.write_html('docs/images/qiskit_circuits/entanglement.html')
        fig.write_image('docs/images/qiskit_circuits/entanglement.png')

# Generate visualizations
visualizer = QiskitCircuitVisualizer()
visualizer.visualize_qiskit_encoding()
visualizer.visualize_quantum_entanglement() 