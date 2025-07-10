import librosa
import numpy as np
import scipy.signal
import scipy.fftpack
import python_speech_features
from scipy.signal import lfilter

def extract_voice_features(audio_path, sr=22050):
    y, sr = librosa.load(audio_path, sr=sr)
    frame_length = 2048
    hop_length = 512

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T

    # Pitch
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = []
    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch.append(pitches[index, i])
    pitch = np.array(pitch)

    # Energy
    energy = np.array([
        np.sqrt(np.mean(y[i:i + frame_length]**2))
        for i in range(0, len(y) - frame_length, hop_length)
    ])

    # Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length).T

    # Spectral Centroid and Bandwidth
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).T
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).T

    # Formants (simplified LPC-based estimation)
    def formants(y, sr, order=12):
        def lpc(signal, order):
            autocorr = np.correlate(signal, signal, mode='full')[len(signal) - 1:]
            R = autocorr[:order + 1]
            A = np.linalg.pinv(toeplitz(R[:-1])) @ -R[1:]
            return np.concatenate(([1.], A))

        from scipy.linalg import toeplitz
        f1_list = []
        f2_list = []
        for i in range(0, len(y) - frame_length, hop_length):
            frame = y[i:i + frame_length] * np.hamming(frame_length)
            a = lpc(frame, order)
            roots = np.roots(a)
            roots = roots[np.imag(roots) >= 0]
            angles = np.angle(roots)
            freqs = angles * (sr / (2 * np.pi))
            freqs = np.sort(freqs)
            if len(freqs) >= 2:
                f1_list.append(freqs[0])
                f2_list.append(freqs[1])
        return np.array(f1_list), np.array(f2_list)

    f1, f2 = formants(y, sr)

    # Concatenate and normalize
    features = np.hstack([
        mfccs[:len(energy)],
        pitch[:len(energy)].reshape(-1, 1),
        energy.reshape(-1, 1),
        zcr[:len(energy)],
        centroid[:len(energy)],
        bandwidth[:len(energy)],
        f1[:len(energy)].reshape(-1, 1),
        f2[:len(energy)].reshape(-1, 1),
    ])

    # Standardization
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    features = (features - mean) / std

    return features
