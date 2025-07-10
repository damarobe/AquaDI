README.txt
==========

VOICE CHARACTERISTICS HASHING & FEATURE EXTRACTION
--------------------------------------------------

This repository contains Python scripts for real-world voice signal processing and learnable hashing of voice characteristics. 
The code is designed for use in projects involving multimodal generative art, speech-driven interfaces, and 
voice-conditioned neural models such as CPPNs and GANs.

The scripts implement methods described in the paper:

"Multimodal Deep Generative Framework for Fluidic Pattern Art Rendering"
by R. Damaševičius et al. (2025)

GitHub Repo: https://github.com/YOUR_USERNAME/voice-characteristics-hashing

--------------------------------------------------
CONTENTS
--------------------------------------------------

1. voice_feature_extraction.py  
   - Extracts MFCCs, pitch, energy, ZCR, spectral centroid, bandwidth, and formants from audio files.

2. learnable_voice_hashing.py  
   - Defines a learnable neural network model for hashing voice features into compact binary codes 
     using contrastive (triplet) loss.

3. triplet_loss_training.py  
   - Implements the Triplet Loss function for training the hash network.

4. hash_code_mapping.py  
   - Converts binary hash codes into a list of model hyperparameters (e.g., learning rate, dropout rate).

5. example_pipeline.py  
   - Example script demonstrating the end-to-end pipeline:
     - Load voice
     - Extract features
     - Generate hash
     - Decode into model parameters

--------------------------------------------------
REQUIREMENTS
--------------------------------------------------

Python >= 3.8

Install required Python packages:

pip install numpy librosa scipy torch python_speech_features


--------------------------------------------------
USAGE EXAMPLE
--------------------------------------------------

1. Place a WAV audio file in the working directory (e.g., "voice.wav").

2. Run the pipeline:

python example_pipeline.py


This will output:

- Normalized acoustic feature vectors
- Binary hash code of the voice
- Interpretable model parameters decoded from the hash

--------------------------------------------------
APPLICATIONS
--------------------------------------------------

- Generative voice-driven art (e.g., Aqua_DI style)
- Speech-based parameter control for neural networks
- Audio fingerprinting and similarity search
- Emotion-to-visual mappings
- Multimodal creative systems

--------------------------------------------------
CITATION
--------------------------------------------------

If you use this code or methodology in your research, please cite:

Damaševičius, R.; Mickus, V.; Žilaitis, M.; Dangveckaitė, E.
"Multimodal Deep Generative Framework for Fluidic Pattern Art Rendering"
Multimodal Technol. Interact. 2025.

--------------------------------------------------
LICENSE
--------------------------------------------------

This code is licensed under the MIT License.

--------------------------------------------------

Developed by: Robertas Damaševičius, 2025
For academic and creative use.
