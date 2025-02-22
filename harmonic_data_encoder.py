import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.fft import fft, fftfreq, ifft
import os

class HarmonicDataEncoder:
    def __init__(self):
        """Initialize the harmonic data encoder with directories for outputs"""
        os.makedirs('docs/images/harmonic_encoding', exist_ok=True)
        self.sample_rate = 1000
        np.random.seed(42)  # For reproducibility

    def generate_data_wave(self, n_samples=100):
        """
        Generate sample data and represent it as a wave
        
        Args:
            n_samples: Number of data points to generate
            
        Returns:
            tuple: (time array, data array, wave representation)
        """
        # Generate random byte-like data
        data = np.random.randint(50, 200, n_samples)
        time = np.linspace(0, 1, n_samples)
        
        # Create wave representation by interpolating between points
        wave = np.interp(
            np.linspace(0, 1, self.sample_rate),
            time,
            data
        )
        
        return time, data, wave

    def visualize_harmonic_encoding(self):
        """Create animated visualization of harmonic encoding process"""
        # Generate data and initial FFT
        time, data, wave = self.generate_data_wave()
        frequencies = fftfreq(self.sample_rate)
        fft_result = fft(wave)
        
        frames = []
        # Animate through different levels of harmonic compression
        for n_harmonics in range(1, 50):
            # Keep only n strongest frequency components
            fft_filtered = np.zeros_like(fft_result, dtype=complex)
            sorted_indices = np.argsort(np.abs(fft_result))[::-1]
            fft_filtered[sorted_indices[:n_harmonics]] = fft_result[sorted_indices[:n_harmonics]]
            
            # Reconstruct signal
            reconstructed = np.real(ifft(fft_filtered))
            
            # Create frame
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=[
                    "Original Data as Wave",
                    f"Frequency Components (Top {n_harmonics} Harmonics)",
                    "Reconstructed Signal"
                ]
            )
            
            # Original data
            fig.add_trace(
                go.Scatter(
                    x=np.linspace(0, 1, len(wave)),
                    y=wave,
                    name="Original",
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            # Frequency spectrum
            fig.add_trace(
                go.Bar(
                    x=frequencies[:len(frequencies)//2],
                    y=np.abs(fft_filtered)[:len(frequencies)//2],
                    name="Harmonics",
                    marker_color='red'
                ),
                row=2, col=1
            )
            
            # Reconstructed signal
            fig.add_trace(
                go.Scatter(
                    x=np.linspace(0, 1, len(reconstructed)),
                    y=reconstructed,
                    name="Reconstructed",
                    line=dict(color='green')
                ),
                row=3, col=1
            )
            
            fig.update_layout(height=900, showlegend=False)
            frames.append(go.Frame(data=fig.data, layout=fig.layout))
        
        # Create final figure with animation
        fig = go.Figure(
            data=frames[0].data,
            layout=go.Layout(
                title="Harmonic Data Encoding Process",
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [{
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 100},
                            'fromcurrent': True,
                            'mode': 'immediate'
                        }]
                    }]
                }],
                height=900
            ),
            frames=frames
        )
        
        # Save visualizations
        fig.write_html('docs/images/harmonic_encoding/encoding_process.html')
        fig.write_image('docs/images/harmonic_encoding/encoding_process.png')

    def visualize_compression_efficiency(self):
        """Visualize compression efficiency with different numbers of harmonics"""
        time, data, wave = self.generate_data_wave()
        fft_result = fft(wave)
        
        # Calculate reconstruction error for different numbers of harmonics
        n_harmonics_range = range(1, 50)
        errors = []
        compression_ratios = []
        
        for n in n_harmonics_range:
            # Filter FFT
            fft_filtered = np.zeros_like(fft_result, dtype=complex)
            sorted_indices = np.argsort(np.abs(fft_result))[::-1]
            fft_filtered[sorted_indices[:n]] = fft_result[sorted_indices[:n]]
            
            # Reconstruct and calculate error
            reconstructed = np.real(ifft(fft_filtered))
            error = np.mean((wave - reconstructed) ** 2)
            compression_ratio = n / len(wave)
            
            errors.append(error)
            compression_ratios.append(compression_ratio)
        
        # Create visualization
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "Reconstruction Error vs. Harmonics",
                "Compression Ratio vs. Harmonics"
            ]
        )
        
        fig.add_trace(
            go.Scatter(
                x=list(n_harmonics_range),
                y=errors,
                mode='lines+markers',
                name='Error'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=list(n_harmonics_range),
                y=compression_ratios,
                mode='lines+markers',
                name='Compression'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Harmonic Compression Efficiency",
            height=500,
            width=1000
        )
        
        fig.write_html('docs/images/harmonic_encoding/compression_efficiency.html')
        fig.write_image('docs/images/harmonic_encoding/compression_efficiency.png')

# Generate visualizations
encoder = HarmonicDataEncoder()
encoder.visualize_harmonic_encoding()
encoder.visualize_compression_efficiency() 