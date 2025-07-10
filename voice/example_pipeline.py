import torch
import numpy as np
from extract_voice_features import extract_voice_features
from voice_hasher import VoiceHasher
from decoder import decode_hash_to_params

# Step 1: Load and extract acoustic features
audio_path = "voice.wav"  # Replace with your actual audio file
print("Extracting voice features...")
features = extract_voice_features(audio_path)
features_tensor = torch.tensor(features, dtype=torch.float32)

# Step 2: Load model and generate binary hash code
input_dim = features.shape[1]
hash_bits = 64
print("Initializing VoiceHasher model...")
model = VoiceHasher(input_dim=input_dim, hash_bits=hash_bits, use_gumbel=True, temperature=0.4)

print("Generating binary voice hash codes...")
model.eval()
with torch.no_grad():
    hash_codes = model(features_tensor)

# Aggregate hash (e.g., mean over time)
aggregated_hash = torch.mean(hash_codes, dim=0)
binary_hash = (aggregated_hash >= 0.5).float().unsqueeze(0)  # Shape: [1, hash_bits]

print(f"Binary hash code (first 32 bits):\n{binary_hash[0, :32].numpy()}")

# Step 3: Decode hash to neural network hyperparameters
# Example: [learning rate, number of layers, dropout rate]
param_ranges = [
    (0.0001, 0.01),   # Learning rate
    (1, 10),          # Number of layers
    (0.1, 0.9)        # Dropout rate
]

print("Decoding hash to model parameters...")
decoded_params = decode_hash_to_params(binary_hash, param_ranges)
print(f"Decoded parameters:\nLearning Rate: {decoded_params[0, 0].item():.5f}, "
      f"Layers: {int(decoded_params[0, 1].item())}, "
      f"Dropout: {decoded_params[0, 2].item():.2f}")
