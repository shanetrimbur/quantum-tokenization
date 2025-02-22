from quantum_tokenizer import QuantumTokenizer

def test_basic_tokenization():
    # Sample text for testing
    sample_text = """
    Quantum computing is an emerging technology that leverages quantum mechanics 
    to solve complex computational problems. Unlike classical computers that use 
    bits, quantum computers use quantum bits or qubits.
    """
    
    # Create tokenizer instance
    tokenizer = QuantumTokenizer(num_merges=10)
    
    # Run tokenization
    tokens = tokenizer.tokenize(sample_text)
    
    # Print results
    print("\nFirst 20 tokens:")
    print(tokens[:20])
    
    # Get and print quantum states
    print("\nExample quantum states:")
    for token in list(tokenizer.vocab.keys())[:5]:
        state = tokenizer.get_quantum_state(token)
        print(f"Token: '{token}', State: {state}")
    
    # Show statistics
    stats = tokenizer.get_stats()
    print("\nTokenization Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    bloch_viz = tokenizer.visualize_tokens_3d()
    bloch_viz.write_html("bloch_visualization.html")
    
    stats_viz = tokenizer.visualize_stats()
    stats_viz.write_html("stats_visualization.html")
    
    print("\nVisualizations have been saved to HTML files")

if __name__ == "__main__":
    test_basic_tokenization() 