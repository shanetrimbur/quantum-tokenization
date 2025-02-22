import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import os

# Create docs/images directory if it doesn't exist
os.makedirs('docs/images', exist_ok=True)

# Try to use seaborn style, fall back to default if not available
try:
    plt.style.use('seaborn')
except:
    print("Seaborn style not available, using default style")

# --- Step 1: Generate a Large Random Byte Data Block ---
np.random.seed(42)
data_block = np.random.randint(50, 200, size=(50, 50))  # 50x50 matrix of random byte values

# Introduce frequent patterns of (136, 32)
for _ in range(100):  # Insert 100 occurrences of (136, 32)
    x, y = np.random.randint(0, 49, size=2)
    data_block[x, y] = 136
    data_block[x, y + 1] = 32  # Ensure adjacent (136, 32) pairs

# --- Step 2: Apply Byte Pair Encoding (BPE) ---
def apply_bpe(data, target_pair=(136, 32), new_token=999):
    """
    Simulates a BPE-like compression by replacing frequent (136, 32) pairs with a single token.

    Args:
        data (numpy array): The input dataset (matrix of byte values).
        target_pair (tuple): The byte pair to replace.
        new_token (int): The new token replacing the frequent byte pair.

    Returns:
        numpy array: Compressed data.
        int: Number of replacements made.
    """
    compressed_data = np.copy(data)  # Copy the data to avoid modifying the original
    token_replacements = 0

    for i in range(data.shape[0]):
        for j in range(data.shape[1] - 1):  # Avoid out-of-bounds indexing
            if (compressed_data[i, j] == target_pair[0]) and (compressed_data[i, j + 1] == target_pair[1]):
                compressed_data[i, j] = new_token  # Replace first byte of the pair with new token
                compressed_data[i, j + 1] = -1  # Mark the second byte as removed
                token_replacements += 1

    return compressed_data, token_replacements

# Apply BPE compression
compressed_data_block, replacements = apply_bpe(data_block)

# --- Step 3: Compute Shannon Entropy Before and After Compression ---
def compute_entropy(data):
    """
    Computes the Shannon entropy of a given dataset (byte values).

    Args:
        data (numpy array): The input dataset.

    Returns:
        float: Shannon entropy in bits.
    """
    values, counts = np.unique(data[data >= 0], return_counts=True)  # Ignore removed (-1) values
    probabilities = counts / counts.sum()  # Normalize frequencies to get probabilities
    entropy = scipy.stats.entropy(probabilities, base=2)  # Compute Shannon entropy in bits
    return entropy

# Compute entropy values
original_entropy = compute_entropy(data_block)
compressed_entropy = compute_entropy(compressed_data_block)

# --- Step 4: Visualize Before and After BPE Compression ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Original data visualization heatmap
cax1 = axes[0].matshow(data_block, cmap="coolwarm")
axes[0].set_title("Before Compression: Frequent (136, 32) Patterns", pad=20)
fig.colorbar(cax1, ax=axes[0])

# Highlight the (136, 32) occurrences in the original data
for i in range(50):
    for j in range(49):
        if data_block[i, j] == 136 and data_block[i, j + 1] == 32:
            axes[0].text(j, i, "136", va="center", ha="center", color="yellow", fontsize=6, fontweight="bold")
            axes[0].text(j + 1, i, "32", va="center", ha="center", color="yellow", fontsize=6, fontweight="bold")

# Compressed data visualization heatmap
cax2 = axes[1].matshow(compressed_data_block, cmap="coolwarm")
axes[1].set_title(f"After Compression: {replacements} (136,32) Pairs Replaced", pad=20)
fig.colorbar(cax2, ax=axes[1])

# Highlight the new token (999) in the compressed data
for i in range(50):
    for j in range(50):
        if compressed_data_block[i, j] == 999:
            axes[1].text(j, i, "999", va="center", ha="center", color="red", fontsize=6, fontweight="bold")

plt.tight_layout()
# Save BPE compression visualization
plt.savefig('docs/images/bpe_compression.png', dpi=300, bbox_inches='tight')

# --- Step 5: Visualize Entropy Reduction ---
plt.figure(figsize=(8, 6))
bars = plt.bar(["Original", "BPE Compressed"], 
               [original_entropy, compressed_entropy], 
               color=["#2ecc71", "#3498db"])
plt.ylabel("Shannon Entropy (bits)")
plt.title("Entropy Reduction After BPE Compression")

# Display entropy values above bars
for bar, entropy_value in zip(bars, [original_entropy, compressed_entropy]):
    plt.text(bar.get_x() + bar.get_width()/2, 
             entropy_value + 0.1,
             f"{entropy_value:.2f} bits",
             ha="center",
             fontsize=12,
             fontweight="bold")

plt.tight_layout()
# Save entropy reduction visualization
plt.savefig('docs/images/entropy_reduction.png', dpi=300, bbox_inches='tight')

# Print entropy values for reference
print(f"Original Shannon Entropy: {original_entropy:.4f} bits")
print(f"BPE Compressed Shannon Entropy: {compressed_entropy:.4f} bits")
print(f"Entropy Reduction: {original_entropy - compressed_entropy:.4f} bits") 