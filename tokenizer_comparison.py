import numpy as np
from transformers import GPT2Tokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from quantum_text_compressor import QuantumTextCompressor
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from collections import Counter
import time

def get_sample_text():
    """Get sample text for testing"""
    return """
    It is a truth universally acknowledged, that a single man in possession of a good fortune, 
    must be in want of a wife. However little known the feelings or views of such a man may be 
    on his first entering a neighbourhood, this truth is so well fixed in the minds of the 
    surrounding families, that he is considered the rightful property of some one or other of 
    their daughters.

    "My dear Mr. Bennet," said his lady to him one day, "have you heard that Netherfield Park 
    is let at last?"

    Mr. Bennet replied that he had not.

    "But it is," returned she; "for Mrs. Long has just been here, and she told me all about it."
    Mr. Bennet made no answer.

    "Do you not want to know who has taken it?" cried his wife impatiently.

    "You want to tell me, and I have no objection to hearing it."

    This was invitation enough.
    """

class TokenizerComparison:
    def __init__(self):
        """Initialize tokenizer comparison"""
        os.makedirs('docs/images/tokenizer_comparison', exist_ok=True)
        self.quantum_compressor = QuantumTextCompressor()
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        # Create BPE tokenizer
        self.bpe_tokenizer = Tokenizer(models.BPE())
        self.bpe_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

    def train_bpe(self, text):
        """Train BPE tokenizer on text"""
        trainer = trainers.BpeTrainer(vocab_size=1000, min_frequency=2)
        self.bpe_tokenizer.train_from_iterator([text], trainer=trainer)

    def compare_methods(self, text):
        """Compare different tokenization methods"""
        results = {}
        
        # Time and analyze each method
        print("Testing GPT-2 tokenization...")
        start = time.time()
        gpt2_tokens = self.gpt2_tokenizer.encode(text)
        results['gpt2'] = {
            'time': time.time() - start,
            'num_tokens': len(gpt2_tokens),
            'unique_tokens': len(set(gpt2_tokens)),
            'compression_ratio': len(text) / len(gpt2_tokens)
        }

        print("Training and testing BPE tokenization...")
        start = time.time()
        self.train_bpe(text)
        bpe_tokens = self.bpe_tokenizer.encode(text).ids
        results['bpe'] = {
            'time': time.time() - start,
            'num_tokens': len(bpe_tokens),
            'unique_tokens': len(set(bpe_tokens)),
            'compression_ratio': len(text) / len(bpe_tokens)
        }

        print("Testing quantum tokenization...")
        start = time.time()
        quantum_tokens = self.quantum_compressor.compress_text(text)
        results['quantum'] = {
            'time': time.time() - start,
            'num_tokens': len(quantum_tokens),
            'unique_tokens': len(set(t[0] for t in quantum_tokens)),
            'compression_ratio': len(text) / len(quantum_tokens)
        }

        return results

    def visualize_comparison(self, results):
        """Create comparison visualizations"""
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{'type': 'xy'}, {'type': 'xy'}],
                [{'type': 'xy'}, {'type': 'xy'}]
            ],
            subplot_titles=[
                "Processing Time",
                "Number of Tokens",
                "Unique Tokens",
                "Compression Ratio"
            ]
        )

        methods = list(results.keys())
        
        # Processing time
        fig.add_trace(
            go.Bar(
                x=methods,
                y=[results[m]['time'] for m in methods],
                name='Processing Time'
            ),
            row=1, col=1
        )

        # Number of tokens
        fig.add_trace(
            go.Bar(
                x=methods,
                y=[results[m]['num_tokens'] for m in methods],
                name='Total Tokens'
            ),
            row=1, col=2
        )

        # Unique tokens
        fig.add_trace(
            go.Bar(
                x=methods,
                y=[results[m]['unique_tokens'] for m in methods],
                name='Unique Tokens'
            ),
            row=2, col=1
        )

        # Compression ratio
        fig.add_trace(
            go.Bar(
                x=methods,
                y=[results[m]['compression_ratio'] for m in methods],
                name='Compression Ratio'
            ),
            row=2, col=2
        )

        fig.update_layout(
            title="Tokenization Method Comparison",
            height=1000,
            showlegend=False
        )

        fig.write_html('docs/images/tokenizer_comparison/comparison.html')
        fig.write_image('docs/images/tokenizer_comparison/comparison.png')

if __name__ == "__main__":
    # Get sample text from quantum compressor
    sample_text = get_sample_text()
    
    print("Initializing comparison...")
    comparison = TokenizerComparison()
    
    print("Running comparisons...")
    results = comparison.compare_methods(sample_text)
    
    print("\nResults Summary:")
    for method, metrics in results.items():
        print(f"\n{method.upper()} Tokenization:")
        print(f"Processing time: {metrics['time']:.3f} seconds")
        print(f"Total tokens: {metrics['num_tokens']}")
        print(f"Unique tokens: {metrics['unique_tokens']}")
        print(f"Compression ratio: {metrics['compression_ratio']:.2f}x")
    
    print("\nGenerating comparison visualizations...")
    comparison.visualize_comparison(results)
    print("Done! Check the docs/images/tokenizer_comparison directory for results.") 