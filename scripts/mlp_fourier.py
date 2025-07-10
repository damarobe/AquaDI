import numpy as np
import matplotlib.pyplot as plt
from skimage.color import hsv2rgb

def generate_fourier_mlp_color(
    H=256, W=256,
    n_f=128, sigma=5.0,
    D=6, W_hidden=128,
    variance=4.0,
    include_radius=True,
    include_latent=True,
    random_seed=11
):
    """
    Generates a color pattern via a Fourier‐Feature MLP.
    Returns an (H, W, 3) RGB image in [0,1].
    """
    np.random.seed(random_seed)

    # 1) Create normalized coordinate grid
    xs = np.linspace(-1.0, 1.0, W, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, H, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    coords = np.stack([X, Y], axis=-1).reshape(-1, 2)  # (H*W, 2)

    # 2) Random Gaussian projection B ~ N(0, σ²)
    B = np.random.normal(0.0, sigma, size=(n_f, 2)).astype(np.float32)  # (n_f,2)
    proj = 2 * np.pi * coords.dot(B.T)                                   # (H*W, n_f)

    # 3) Fourier features: [cos, sin]
    feat_cos = np.cos(proj)
    feat_sin = np.sin(proj)
    features = np.concatenate([feat_cos, feat_sin], axis=1)             # (H*W, 2*n_f)
    features *= np.sqrt(2.0 / n_f)

    # 4) Optional extras: radius and latent bias
    extras = []
    if include_radius:
        r = np.sqrt(X**2 + Y**2).reshape(-1, 1).astype(np.float32)
        extras.append(r)
    if include_latent:
        z = np.random.normal(0.0, 1.0, size=(1,)).astype(np.float32)
        extras.append(np.full((H*W, 1), z, dtype=np.float32))
    if extras:
        features = np.concatenate([features] + extras, axis=1)           # (H*W, N)

    # 5) Build MLP weights with Variance‐Scaling init
    Ws, bs = [], []
    in_dim = features.shape[1]
    for i in range(D):
        fan_in = in_dim if i == 0 else W_hidden
        W_layer = (np.random.randn(fan_in, W_hidden).astype(np.float32) *
                   np.sqrt(variance / fan_in))
        b_layer = np.zeros((W_hidden,), dtype=np.float32)
        Ws.append(W_layer); bs.append(b_layer)
    # Final output layer → 3 channels
    W_out = (np.random.randn(W_hidden, 3).astype(np.float32) *
             np.sqrt(variance / W_hidden))
    b_out = np.zeros((3,), dtype=np.float32)
    Ws.append(W_out); bs.append(b_out)

    # 6) Forward pass through MLP
    x = features
    for i in range(D):
        x = np.tanh(x.dot(Ws[i]) + bs[i])
    out = np.tanh(x.dot(Ws[-1]) + bs[-1])  # shape (H*W, 3)

    # 7) Convert to HSV→RGB
    hsv = (out.reshape(H, W, 3) + 1.0) / 2.0
    rgb = hsv2rgb(hsv)
    return rgb

if __name__ == "__main__":
    img = generate_fourier_mlp_color()
    plt.figure(figsize=(6,6))
    plt.imshow(img, interpolation='bilinear')
    plt.axis('off')
    plt.title("Fourier‐Feature MLP Color Pattern (256×256)")
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import hsv2rgb

def generate_fourier_mlp_color(
    H=256, W=256,
    n_f=128, sigma=5.0,
    D=6, W_hidden=128,
    variance=100.0,
    include_radius=True,
    include_latent=True,
    random_seed=0
):
    """
    Generates a color pattern via a Fourier‐Feature MLP.
    Returns an (H, W, 3) RGB image in [0,1].
    """
    np.random.seed(random_seed)

    # 1) Create normalized coordinate grid
    xs = np.linspace(-1.0, 1.0, W, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, H, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    coords = np.stack([X, Y], axis=-1).reshape(-1, 2)  # (H*W, 2)

    # 2) Random Gaussian projection B ~ N(0, σ²)
    B = np.random.normal(0.0, sigma, size=(n_f, 2)).astype(np.float32)  # (n_f,2)
    proj = 2 * np.pi * coords.dot(B.T)                                   # (H*W, n_f)

    # 3) Fourier features: [cos, sin]
    feat_cos = np.cos(proj)
    feat_sin = np.sin(proj)
    features = np.concatenate([feat_cos, feat_sin], axis=1)             # (H*W, 2*n_f)
    features *= np.sqrt(2.0 / n_f)

    # 4) Optional extras: radius and latent bias
    extras = []
    if include_radius:
        r = np.sqrt(X**2 + Y**2).reshape(-1, 1).astype(np.float32)
        extras.append(r)
    if include_latent:
        z = np.random.normal(0.0, 1.0, size=(1,)).astype(np.float32)
        extras.append(np.full((H*W, 1), z, dtype=np.float32))
    if extras:
        features = np.concatenate([features] + extras, axis=1)           # (H*W, N)

    # 5) Build MLP weights with Variance‐Scaling init
    Ws, bs = [], []
    in_dim = features.shape[1]
    for i in range(D):
        fan_in = in_dim if i == 0 else W_hidden
        W_layer = (np.random.randn(fan_in, W_hidden).astype(np.float32) *
                   np.sqrt(variance / fan_in))
        b_layer = np.zeros((W_hidden,), dtype=np.float32)
        Ws.append(W_layer); bs.append(b_layer)
    # Final output layer → 3 channels
    W_out = (np.random.randn(W_hidden, 3).astype(np.float32) *
             np.sqrt(variance / W_hidden))
    b_out = np.zeros((3,), dtype=np.float32)
    Ws.append(W_out); bs.append(b_out)

    # 6) Forward pass through MLP
    x = features
    for i in range(D):
        x = np.tanh(x.dot(Ws[i]) + bs[i])
    out = np.tanh(x.dot(Ws[-1]) + bs[-1])  # shape (H*W, 3)

    # 7) Convert to HSV→RGB
    hsv = (out.reshape(H, W, 3) + 1.0) / 2.0
    rgb = hsv2rgb(hsv)
    return rgb

if __name__ == "__main__":
    img = generate_fourier_mlp_color()
    plt.figure(figsize=(6,6))
    plt.imshow(img, interpolation='bilinear')
    plt.axis('off')
    plt.title("Fourier‐Feature MLP Color Pattern (256×256)")
    plt.savefig("mlp_fourier.png",dpi=(300), bbox_inches='tight')
    plt.show()
