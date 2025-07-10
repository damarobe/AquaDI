#!/usr/bin/env python3
"""
batched_cppn_generator.py

Generate a batch of colorful CPPN‐style images with **randomized**
hyperparameters per image, to maximize pattern diversity.
"""

import os
import argparse
import numpy as np
from PIL import Image
from skimage.color import hsv2rgb
from skimage.filters import sobel

# -------------------
# CPPN CONSTRUCTION
# -------------------

def build_cppn(input_dim, hidden_width, hidden_depth, variance, activations, seed):
    """
    Build weight/bias lists for a small MLP (the CPPN).
    input_dim: dimensionality of per‐pixel input vector
    hidden_width: number of units per hidden layer
    hidden_depth: number of hidden layers
    variance: scale factor for weight init
    activations: list of strings ('sin','tanh','relu') of length hidden_depth
    seed: random seed for reproducibility
    """
    rng = np.random.RandomState(seed)
    Ws, bs = [], []
    in_dim = input_dim
    for act in activations:
        W = rng.randn(in_dim, hidden_width).astype(np.float32) * np.sqrt(variance/in_dim)
        b = np.zeros((hidden_width,), dtype=np.float32)
        Ws.append(W); bs.append(b)
        in_dim = hidden_width
    # final layer → 1 pattern channel
    Wf = rng.randn(hidden_width,1).astype(np.float32) * np.sqrt(variance/hidden_width)
    bf = np.zeros((1,), dtype=np.float32)
    Ws.append(Wf); bs.append(bf)
    return Ws, bs

def apply_activation(x, act):
    """In‐place numpy activations."""
    if act == 'sin':
        return np.sin(x)
    elif act == 'tanh':
        return np.tanh(x)
    elif act == 'relu':
        return np.maximum(0, x)
    else:
        # fallback
        return np.tanh(x)

def evaluate_cppn(Ws, bs, activations, grid):
    """
    Forward‐pass: grid has shape (H*W, input_dim).
    activations is length hidden_depth.
    Returns a (H*W,) pattern in [-1,1].
    """
    x = grid.dot(Ws[0]) + bs[0]
    x = apply_activation(x, activations[0])
    for i, act in enumerate(activations[1:], start=1):
        x = x.dot(Ws[i]) + bs[i]
        x = apply_activation(x, act)
    # final linear
    out = x.dot(Ws[-1]) + bs[-1]
    return out[:,0]

# -------------------
# COORDINATE WARP
# -------------------

def warp_coords(X, Y, alpha, freq):
    """
    A simple swirling warp.
    alpha controls strength, freq controls frequency.
    """
    Xw = X + alpha * np.sin(freq * Y)
    Yw = Y + alpha * np.sin(freq * X)
    return Xw, Yw

# -------------------
# IMAGE GENERATION
# -------------------

def generate_image(params):
    """
    params: dict containing
      width, height, seed,
      hidden_depth, hidden_width,
      variance, warp_alpha, warp_freq,
      latent_dim
    """
    W, H = params['width'], params['height']
    seed = params['seed']
    np.random.seed(seed)

    # 1) sample a random sequence of activations
    acts = np.random.choice(['sin','tanh','relu'], size=params['hidden_depth']).tolist()

    # 2) build the CPPN
    input_dim = 3 + params['latent_dim']      # x,y,r + latent
    Ws, bs = build_cppn(
        input_dim=input_dim,
        hidden_width=params['hidden_width'],
        hidden_depth=params['hidden_depth'],
        variance=params['variance'],
        activations=acts,
        seed=seed+1
    )

    # 3) create & warp grid
    xs = np.linspace(-1,1,W, dtype=np.float32)
    ys = np.linspace(-1,1,H, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    Xw, Yw = warp_coords(X, Y, params['warp_alpha'], params['warp_freq'])
    r = np.sqrt(Xw**2 + Yw**2)

    # 4) assemble per‐pixel input
    grid = np.stack([Xw, Yw, r], axis=-1).reshape(-1,3)
    if params['latent_dim'] > 0:
        z = np.random.RandomState(seed+2).randn(params['latent_dim']).astype(np.float32)
        Z = np.tile(z, (H*W,1))
        grid = np.concatenate([grid, Z], axis=1)

    # 5) evaluate CPPN and normalize
    pat = evaluate_cppn(Ws, bs, acts, grid)
    pat = (pat - pat.min())/(pat.max()-pat.min()+1e-8)
    pat_img = pat.reshape(H, W)

    # 6) map into HSV→RGB
    hue = pat_img
    sat = np.ones_like(hue)*0.8
    val = np.clip(1 - r, 0,1)
    hsv = np.stack([hue, sat, val], axis=2)
    rgb = hsv2rgb(hsv)

    # 7) overlay neon edges
    edges = sobel(val)
    mask = (edges > 0.1)[...,None]
    edge_col = np.array([0.1,1.0,0.2])[None,None,:]
    rgb = np.where(mask, edge_col, rgb)

    return (rgb*255).astype(np.uint8)

# -------------------
# BATCH DRIVER
# -------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir",   required=True,         help="Where to save")
    p.add_argument("--count",     type=int, default=20,  help="Num images")
    p.add_argument("--width",     type=int, default=512)
    p.add_argument("--height",    type=int, default=512)
    p.add_argument("--latent_dim",type=int, default=2)
    p.add_argument("--seed",      type=int, default=0)
    p.add_argument("--var_min",   type=float, default=50.0)
    p.add_argument("--var_max",   type=float, default=300.0)
    p.add_argument("--warp_min",  type=float, default=0.1)
    p.add_argument("--warp_max",  type=float, default=1.5)
    p.add_argument("--freq_min",  type=float, default=1.0)
    p.add_argument("--freq_max",  type=float, default=10.0)
    p.add_argument("--depth_min", type=int, default=4)
    p.add_argument("--depth_max", type=int, default=8)
    p.add_argument("--width_min", type=int, default=64)
    p.add_argument("--width_max", type=int, default=256)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for i in range(args.count):
        seed = args.seed + i
        # randomize hyperparams
        variance   = np.random.uniform(args.var_min, args.var_max)
        warp_alpha = np.random.uniform(args.warp_min, args.warp_max)
        warp_freq  = np.random.uniform(args.freq_min, args.freq_max)
        hidden_depth = np.random.randint(args.depth_min, args.depth_max+1)
        hidden_width = np.random.randint(args.width_min, args.width_max+1)

        params = dict(
            width=args.width, height=args.height,
            seed=seed,
            latent_dim=args.latent_dim,
            variance=variance,
            warp_alpha=warp_alpha,
            warp_freq=warp_freq,
            hidden_depth=hidden_depth,
            hidden_width=hidden_width
        )
        img = generate_image(params)
        fname = (f"cppn_{i:03d}_s{seed}"
                 f"_var{int(variance)}"
                 f"_warp{warp_alpha:.2f}"
                 f"_freq{warp_freq:.1f}"
                 f"_d{hidden_depth}"
                 f"_w{hidden_width}.png")
        path = os.path.join(args.out_dir, fname)
        Image.fromarray(img).save(path)
        print(f"Saved {path}")

if __name__=="__main__":
    main()
