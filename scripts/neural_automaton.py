import tensorflow as tf
from tensorflow.keras import layers, models

class RadialNCA(tf.keras.Model):
    def __init__(self, H, W, C_pattern=1, k=3, hidden_dim=128, alpha=0.5, T_steps=64):
        super().__init__()
        self.H = H
        self.W = W
        self.Cp = C_pattern
        self.k = k
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.T_steps = T_steps

        # Reaction MLP
        self.dense1 = layers.Dense(hidden_dim, activation='tanh')
        self.dense2 = layers.Dense(C_pattern)

    def call(self, state):
        # state shape = (batch, H, W, Cp+2)
        batch = tf.shape(state)[0]
        in_dim = (self.Cp + 2) * self.k * self.k

        for _ in range(self.T_steps):
            # 1) Extract k×k patches around each pixel
            patches = tf.image.extract_patches(
                images=state,
                sizes=[1, self.k, self.k, 1],
                strides=[1, 1, 1, 1],
                rates=[1, 1, 1, 1],
                padding='SAME'
            )  # shape = (batch, H, W, in_dim)

            # 2) Flatten per-pixel patches → shape (batch*H*W, in_dim)
            flat = tf.reshape(patches, [-1, in_dim])

            # 3) Reaction MLP
            h = self.dense1(flat)                     # (batch*H*W, hidden_dim)
            delta = self.dense2(h)                    # (batch*H*W, Cp)
            delta = tf.reshape(delta, [batch, self.H, self.W, self.Cp])

            # 4) Residual update of the pattern channels
            pat = state[..., :self.Cp] + self.alpha * delta
            pat = tf.clip_by_value(pat, 0.0, 1.0)

            # 5) Reconstruct new state by concatenating fixed coords
            state = tf.concat([pat, state[..., self.Cp:]], axis=-1)

        return state

# Usage example:

# — parameters —
H, W = 128, 128
C_pattern = 1
k = 3
hidden_dim = 256
alpha = 0.5
T_steps = 128

# Build the model
nca = RadialNCA(H, W, C_pattern, k, hidden_dim, alpha, T_steps)

# Prepare an initial state (batch of 1)
# state[..., :1] zeros, state[...,1] = r_norm, state[...,2] = theta_norm
import numpy as np
xs = np.linspace(-1, 1, W)
ys = np.linspace(-1, 1, H)
X, Y = np.meshgrid(xs, ys)
r = np.sqrt(X**2 + Y**2)
r_norm = np.clip(r / np.sqrt(2), 0, 1)[..., None]
theta = np.arctan2(Y, X)
theta_norm = ((theta + np.pi) / (2*np.pi))[..., None]

init_state = np.concatenate([
    np.zeros((H, W, 1), dtype=np.float32),
    r_norm.astype(np.float32),
    theta_norm.astype(np.float32)
], axis=-1)
init_state = init_state[None]  # add batch dim

# Run 64 iterations
output_state = nca(init_state)
pattern = output_state.numpy()[0, ..., 0]  # the learned pattern channel

# Visualize
import matplotlib.pyplot as plt
from skimage.color import hsv2rgb

hue = pattern
sat = np.ones_like(hue) * 0.9
val = 1.0 - r_norm[...,0]
hsv = np.stack([hue, sat, val], axis=2)
rgb = hsv2rgb(hsv)

plt.figure(figsize=(5,5))
plt.imshow(rgb)
plt.axis('off')
plt.show()
