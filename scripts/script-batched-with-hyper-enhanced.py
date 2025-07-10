#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#   • Fourier feature embedding
#   • Coordinate warping via small SIREN
#   • Hypernetwork generates pixel-MLP weights from a seed
#   • Fully NumPy-based pixel pass for simplicity/memory

import os
import time
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ----- Helpers -----

def save_image(img: np.ndarray, out_dir: str, postfix: str="") -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    name = f"img.{ts}{postfix}.png"
    path = os.path.join(out_dir, name)
    arr = np.clip(img, 0.0, 1.0)
    arr = (arr * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)
    return path


def fourier_embed(xy: np.ndarray, bands: int) -> np.ndarray:
    """
    xy: (N,2) coords in [-1,1]
    returns (N, 2 + 4*bands)
    """
    feats = [xy]
    for i in range(bands):
        freq = (2.0 ** i) * np.pi
        feats.append(np.sin(freq * xy))
        feats.append(np.cos(freq * xy))
    return np.concatenate(feats, axis=1)

class Sine(layers.Layer):
    def __init__(self, w0=60.0, **kwargs):
        super().__init__(**kwargs)
        self.w0 = w0
    def call(self, x):
        return tf.sin(self.w0 * x)

# ----- Models -----

class WarpNet(keras.Model):
    """
    Small SIREN to warp x,y coords.
    """
    def __init__(self, hidden_dim=64):
        inp = layers.Input((2,))
        h = layers.Dense(hidden_dim)(inp)
        h = Sine()(h)
        h = layers.Dense(hidden_dim)(h)
        h = Sine()(h)
        h = layers.Dense(hidden_dim)(h)
        h = Sine()(h)
        off = layers.Dense(2, activation=None)(h)
        super().__init__(inp, off, name="warp_net")

class HyperMLP(layers.Layer):
    """
    Generates weights for a 2-layer MLP:
      - features → [hidden_dim] → [out_dim]
    """
    def __init__(self, seed_dim, feat_dim, hidden_dim, out_dim, **kwargs):
        super().__init__(**kwargs)
        self.seed_dim = seed_dim
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        # weight generators
        self.w1_gen = layers.Dense(feat_dim * hidden_dim)
        self.b1_gen = layers.Dense(hidden_dim)
        self.w2_gen = layers.Dense(hidden_dim * out_dim)
        self.b2_gen = layers.Dense(out_dim)

    def call(self, seed: tf.Tensor):
        # seed: (1, seed_dim)
        w1_flat = self.w1_gen(seed)     # (1, feat_dim*hidden_dim)
        b1      = self.b1_gen(seed)     # (1, hidden_dim)
        w2_flat = self.w2_gen(seed)     # (1, hidden_dim*out_dim)
        b2      = self.b2_gen(seed)     # (1, out_dim)
        return w1_flat, b1, w2_flat, b2

# ----- Generation Pipeline -----

def create_image(
    warp_net: WarpNet,
    hyper: HyperMLP,
    x_dim: int,
    y_dim: int,
    seed_vector: np.ndarray,
    fourier_bands: int,
    batch_size: int = 65536,
) -> np.ndarray:
    """
    Generates one image (y_dim, x_dim, 3) using:
     - warp_net to deform coords
     - hyper to produce pixel-MLP weights
     - NumPy pixel pass
    """
    num_px = x_dim * y_dim
    # grid coords in [-1,1]
    xs = np.linspace(-1,1,x_dim)
    ys = np.linspace(-1,1,y_dim)
    grid = np.stack(np.meshgrid(xs, ys), -1).reshape(-1,2)  # (num_px,2)

    # Generate static weights once via hypernetwork
    seed_tf = tf.convert_to_tensor(seed_vector.astype(np.float32))  # (1, seed_dim)
    w1_flat, b1, w2_flat, b2 = hyper(seed_tf)
    # reshape & squeeze to NumPy
    W1 = tf.reshape(w1_flat, (hyper.feat_dim, hyper.hidden_dim)).numpy()
    b1 = b1.numpy().reshape(hyper.hidden_dim)
    W2 = tf.reshape(w2_flat, (hyper.hidden_dim, hyper.out_dim)).numpy()
    b2 = b2.numpy().reshape(hyper.out_dim)

    # Prepare output
    out = np.zeros((num_px, hyper.out_dim), dtype=np.float32)

    # Batched pixel pass
    for i in tqdm(range(0, num_px, batch_size), desc="Generating image"):
        j = min(i+batch_size, num_px)
        coords = grid[i:j].astype(np.float32)
        # warp coords via Keras
        offs = warp_net(coords).numpy()  # (batch,2)
        warped = coords + 0.3 * offs

        feats = fourier_embed(warped, fourier_bands)  # (batch, feat_dim)

        # pixel-MLP in NumPy: sin hidden, sigmoid output
        h_act = np.sin(feats.dot(W1) + b1)
        c = 1.0 / (1.0 + np.exp(-(h_act.dot(W2) + b2)))
        out[i:j] = c

    # reshape to H×W×C
    return out.reshape((y_dim, x_dim, hyper.out_dim))

# ----- Main -----

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--x", type=int, default=512)
    p.add_argument("--y", type=int, default=512)
    p.add_argument("--seed-dim", type=int, default=8)
    p.add_argument("--fourier-bands", type=int, default=14)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=65536)
    p.add_argument("--out-dir",   type=str, default="results")
    args = p.parse_args()

    # Build warp net & hypernetwork
    feat_dim = 2 + 4 * args.fourier_bands
    warp_net = WarpNet(hidden_dim=64)
    hyper    = HyperMLP(
        seed_dim=args.seed_dim,
        feat_dim=feat_dim,
        hidden_dim=args.hidden_dim,
        out_dim=3,
    )

    # Random seed vector
    seed_vec = np.random.randn(1, args.seed_dim).astype(np.float32)

    # Generate
    img = create_image(
        warp_net,
        hyper,
        args.x,
        args.y,
        seed_vec,
        args.fourier_bands,
        batch_size=args.batch_size,
    )

    # Save
    path = save_image(img, args.out_dir)
    print("Saved →", path)

if __name__ == "__main__":
    main()
