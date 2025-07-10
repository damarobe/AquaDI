#!/usr/bin/env python3
"""
cppn_generator.py

Generate colorful, mandala‐like patterns with edge outlines,
using a modified CPPN architecture and coordinate warping.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import hsv2rgb
from skimage.filters import sobel
from PIL import Image

def build_cppn(input_dim=4, hidden_width=128, hidden_depth=6, variance=100.0):
    """
    Build a small MLP with sinusoidal and tanh activations.
    input_dim: number of per‐pixel inputs (x,y,r,latent)
    hidden_width: neurons per hidden layer
    hidden_depth: number of hidden layers
    variance: scale for VarianceScaling init
    Returns: list of weight matrices Ws and biases bs.
    """
    rng = np.random.RandomState(0)
    init_scale = variance
    Ws, bs = [], []
    in_dim = input_dim
    for i in range(hidden_depth):
        W = rng.randn(in_dim, hidden_width).astype(np.float32) * np.sqrt(init_scale/in_dim)
        b = np.zeros((hidden_width,), dtype=np.float32)
        Ws.append(W); bs.append(b)
        in_dim = hidden_width
    # final layer to 1 channel (grayscale pattern)
    W = rng.randn(hidden_width, 1).astype(np.float32) * np.sqrt(init_scale/hidden_width)
    b = np.zeros((1,), dtype=np.float32)
    Ws.append(W); bs.append(b)
    return Ws, bs

def warp_coords(X, Y, alpha=0.5):
    """
    Apply a fluid‐like sinusoidal warp to coordinates.
    """
    Xw = X + alpha * np.sin(3*Y + 2*X)
    Yw = Y + alpha * np.sin(3*X + 2*Y)
    return Xw, Yw

def evaluate_cppn(Ws, bs, grid):
    """
    Forward‐pass the grid through MLP with sine first layer, tanh elsewhere.
    grid: (H*W, input_dim)
    Returns: (H*W,) grayscale in [-1,1]
    """
    x = grid
    # first layer: sine activation
    W0, b0 = Ws[0], bs[0]
    x = np.sin(x.dot(W0) + b0)
    # remaining hidden layers: tanh
    for W, b in zip(Ws[1:-1], bs[1:-1]):
        x = np.tanh(x.dot(W) + b)
    # final linear
    Wf, bf = Ws[-1], bs[-1]
    out = x.dot(Wf) + bf
    return out[:,0]

def generate_image(
    width=512, height=512,
    latent_dim=2,
    variance=200.0,
    warp_alpha=0.8
):
    # 1) build CPPN
    input_dim = 3 + latent_dim  # x,y,r + latent
    Ws, bs = build_cppn(input_dim=input_dim,
                        hidden_width=128,
                        hidden_depth=6,
                        variance=variance)

    # 2) prepare grid
    xs = np.linspace(-1,1,width, dtype=np.float32)
    ys = np.linspace(-1,1,height, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    # warp coords
    Xw, Yw = warp_coords(X, Y, alpha=warp_alpha)
    r = np.sqrt(Xw**2 + Yw**2)
    # stack inputs
    grid = np.stack([Xw, Yw, r], axis=-1).reshape(-1,3)
    if latent_dim>0:
        z = np.random.RandomState(1).randn(latent_dim).astype(np.float32)
        Z = np.tile(z, (width*height,1))
        grid = np.concatenate([grid, Z], axis=1)  # (H*W, input_dim)

    # 3) evaluate
    pat = evaluate_cppn(Ws, bs, grid)
    pat = (pat - pat.min())/(pat.max()-pat.min()+1e-8)  # [0,1]
    pat_img = pat.reshape(height, width)

    # 4) color mapping: hue from pattern, fixed sat & val
    hue = pat_img
    sat = np.ones_like(hue)*0.8
    val = np.clip(1 - r, 0,1)
    hsv = np.stack([hue, sat, val], axis=2)
    rgb = hsv2rgb(hsv)

    # 5) edge outline: sobel on value channel
    edges = sobel(val)
    edges = (edges>0.1).astype(np.float32)[...,None]
    # neon‐green RGBA for edges
    edge_color = np.array([0.1,1.0,0.2])[None,None,:]
    # overlay edges
    rgb = np.where(edges==1, edge_color, rgb)

    return (rgb*255).astype(np.uint8)

if __name__=="__main__":
    img = generate_image(width=512, height=512,
                         latent_dim=2,
                         variance=200.0,
                         warp_alpha=0.8)
    im = Image.fromarray(img)
    im.save("cppn_style.png")
    im.show()
