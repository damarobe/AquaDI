import numpy as np
import matplotlib.pyplot as plt

def radial_rule_nca_color(H=256, W=256, C_pattern=3, k=3, hidden_dim=128,
                          alpha=0.5, T_steps=64, seed=10):
    """
    Neural Cellular Automaton with radial rules producing a 3-channel color pattern.
    Returns an (H, W, 3) array of RGB values in [0,1].
    """
    # 1) Prepare normalized coordinate grids
    xs = np.linspace(-1, 1, W, dtype=np.float32)
    ys = np.linspace(-1, 1, H, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    r = np.sqrt(X**2 + Y**2)
    r_norm = np.clip(r/np.sqrt(2), 0, 1)
    theta = np.arctan2(Y, X)
    theta_norm = (theta + np.pi) / (2*np.pi)

    # 2) Initialize state: [R,G,B, r, theta]
    np.random.seed(seed)
    state = np.zeros((H, W, C_pattern + 2), dtype=np.float32)
    # Start RGB channels with small random noise
    state[..., 0:3] = np.random.rand(H, W, 3) * 0.1
    state[..., 3] = r_norm
    state[..., 4] = theta_norm

    # 3) Precompute neighbor offsets for k×k patch
    offsets = [(i, j) for i in range(-(k//2), k//2+1)
                       for j in range(-(k//2), k//2+1)]

    # 4) Initialize reaction MLP weights
    in_dim = (C_pattern + 2) * k * k
    rng = np.random.RandomState(seed + 1)
    W1 = rng.randn(in_dim, hidden_dim).astype(np.float32) * (2/np.sqrt(in_dim))
    b1 = np.zeros((hidden_dim,), dtype=np.float32)
    W2 = rng.randn(hidden_dim, C_pattern).astype(np.float32) * (2/np.sqrt(hidden_dim))
    b2 = np.zeros((C_pattern,), dtype=np.float32)

    # 5) NCA iteration loop
    for t in range(T_steps):
        # 5a) Gather k×k neighborhood patches
        patches = []
        for di, dj in offsets:
            rolled = np.roll(np.roll(state, di, axis=0), dj, axis=1)
            patches.append(rolled)
        percep = np.concatenate(patches, axis=2)  # shape (H,W,in_dim)

        # 5b) Flatten to (H*W, in_dim)
        flat = percep.reshape(-1, in_dim)

        # 5c) Reaction MLP: two‐layer tanh network
        h = np.tanh(flat.dot(W1) + b1)             # (H*W, hidden_dim)
        delta = np.tanh(h.dot(W2) + b2)            # (H*W, 3)
        delta = delta.reshape(H, W, C_pattern)     # (H, W, 3)

        # 5d) Residual update & clamp color channels
        state[..., 0:3] += alpha * delta
        state[..., 0:3] = np.clip(state[..., 0:3], 0.0, 1.0)

    # 6) Return the final RGB pattern
    return state[..., 0:3]


if __name__ == "__main__":
    # Generate and plot the NCA color pattern
    pattern_rgb = radial_rule_nca_color(
        H=256, W=256,
        C_pattern=3, k=3,
        hidden_dim=128, alpha=0.5,
        T_steps=64, seed=1234
    )

    plt.figure(figsize=(6,6))
    plt.imshow(pattern_rgb, interpolation="bilinear")
    plt.axis("off")
    plt.title("Radial‐Rule NCA Color Output after 64 Iterations")
    plt.show()
