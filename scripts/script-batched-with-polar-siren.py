# -*- coding: utf-8 -*-
# 
# Specify a seed via cli argument --seed 349348
# Specify `--save --path folder/` to save the images

import os
import time
import math 
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
from skimage.color import hsv2rgb
import matplotlib.pyplot as plt
import keras
from keras import models, layers, initializers
from keras.layers import Input
import tensorflow as tf
import numpy as np
from skimage.color import hsv2rgb


keras.backend.set_learning_phase(1)


from keras import models, layers, initializers
import tensorflow as tf

import tensorflow as tf
from keras import models, layers, initializers


import tensorflow as tf
import numpy as np
from keras import models, layers

def build_model_polar_siren(n_extra, k=8, width=128, depth=6, ω0=30.0):
    """
    - n_extra: number of non-(x,y) inputs (e.g. lat, r_bias, etc).
    - k: desired rotational symmetry (e.g. 8 for octagonal mandala).
    - width, depth, ω0: SIREN hyperparams.
    """

    # 1) Input: [x, y, *extras]
    inp = layers.Input(shape=(2 + n_extra,))

    # 2) Split coordinates vs extras
    xy    = layers.Lambda(lambda z: z[:, :2],      name="xy")(inp)       # (batch,2)
    extras= layers.Lambda(lambda z: z[:, 2:],      name="extras")(inp)   # (batch,n_extra)

    # 3) Polar transform
    def to_polar(z):
        x, y = z[:,0], z[:,1]
        r = tf.sqrt(x*x + y*y)
        θ = tf.atan2(y, x)            # in [−π, +π]
        return tf.stack([r, θ], axis=1)
    polar = layers.Lambda(to_polar, name="to_polar")(xy)  # (batch,2)

    # 4) Build k-fold harmonics
    def harmonics(z):
        r, θ = z[:,0:1], z[:,1:2]
        feats = [r]  # always include radius
        for m in range(1, depth):
            if m % k == 0:
                feats.append(tf.sin(m*θ))
                feats.append(tf.cos(m*θ))
        return tf.concat(feats, axis=1)
    poly = layers.Lambda(harmonics, name="kfold_harmonics")(polar)

    # 5) Concatenate harmonics + extras
    x = layers.Concatenate(name="features")([poly, extras])

    # 6) SIREN‐style MLP
    def sine_dense(x, is_first=False):
        init = tf.random_uniform_initializer(
            minval=-1.0/(width if is_first else np.sqrt(width)),
            maxval=+1.0/(width if is_first else np.sqrt(width))
        )
        w = layers.Dense(width, kernel_initializer=init, use_bias=True)(x)
        return tf.sin(ω0 * w) if is_first else tf.sin(w)

    x = sine_dense(x, is_first=True)
    for _ in range(depth-1):
        x = sine_dense(x, is_first=False)

    # 7) Final tanh → 3 channels (HSV) or 1 channel
    out = layers.Dense(3, activation="tanh")(x)
    model = models.Model(inp, out)
    return model



def create_grid(size, lat, scale = 1.0):
    x_dim, y_dim = size
    N = np.mean((x_dim, y_dim))
    x = np.linspace(- x_dim / N * scale, x_dim / N * scale, x_dim)
    y = np.linspace(- y_dim / N * scale, y_dim / N * scale, y_dim)

    X, Y = np.meshgrid(x, y)

    x = np.ravel(X).reshape(-1, 1)
    y = np.ravel(Y).reshape(-1, 1)
    r = np.sqrt(x ** 2 + y ** 2)

    # lat = np.random.normal(0,1,1)
    Z = np.repeat(lat, x.shape[0]).reshape(-1, x.shape[0])

   #  h = x * y * x

    return x, y, Z.T, r # r, Z.T, alpha


def create_image_batched(model, params, size, batch_size=65536):
    """
    Generate a single image by streaming pixel coordinates through `model.predict`
    in batches of `batch_size`. Performs two passes:
      1) find global per‐channel min/max across all pixels
      2) normalize per‐channel and write into the final 2D/3D array
    
    Args:
      model      : a compiled Keras model that takes shape (n_features,) → (1 or 3,)
      params     : tuple (x_flat, y_flat, lat_flat, r_flat), each of shape (num_pixels, 1)
      size       : (x_dim, y_dim)
      batch_size : int, maximum number of pixels to process at once
    
    Returns:
      img_uint8 : np.ndarray of shape (y_dim, x_dim, channels) with dtype uint8
    """
    x_flat, y_flat, lat_flat, r_flat = params
    x_dim, y_dim = size
    num_pixels = x_flat.shape[0]
    channels = model.output_shape[-1]  # should be 3 or 1

    # ——— PASS 1: compute global min/max for each channel ———
    # initialize min/max arrays
    global_min = np.full((channels,), np.inf, dtype=np.float32)
    global_max = np.full((channels,), -np.inf, dtype=np.float32)

    # iterate in batches
    for start in range(0, num_pixels, batch_size):
        end = min(start + batch_size, num_pixels)
        # slice out the batch of coordinates
        X_batch = np.concatenate([
            x_flat[start:end],
            y_flat[start:end],
            lat_flat[start:end],
            r_flat[start:end]
        ], axis=1)  # shape (batch_size, n_features)

        preds = model.predict(X_batch, verbose=0)  # shape (batch_batch, channels)
        # update per‐channel mins and maxs
        batch_min = preds.min(axis=0)  # shape (channels,)
        batch_max = preds.max(axis=0)
        global_min = np.minimum(global_min, batch_min)
        global_max = np.maximum(global_max, batch_max)
        # discard `preds` to free memory, move to next batch

    # ——— PASS 2: normalize each batch using global_min/global_max ———
    # preallocate output image array
    if channels == 1:
        img = np.zeros((y_dim, x_dim), dtype=np.float32)
    else:
        img = np.zeros((y_dim, x_dim, channels), dtype=np.float32)

    for start in range(0, num_pixels, batch_size):
        end = min(start + batch_size, num_pixels)
        X_batch = np.concatenate([
            x_flat[start:end],
            y_flat[start:end],
            lat_flat[start:end],
            r_flat[start:end]
        ], axis=1)  # shape (batch_subsize, n_features)

        preds = model.predict(X_batch, verbose=0)  # shape (batch_subsize, channels)

        # normalize each channel: (val - min) / (max - min)
        for c in range(channels):
            c_min = global_min[c]
            c_max = global_max[c]
            # avoid division by zero if c_max == c_min (unlikely for a CPPN, but just in case)
            denom = c_max - c_min if (c_max > c_min) else 1.0
            preds[:, c] = (preds[:, c] - c_min) / denom

        # now preds is in [0, 1] for each channel
        # reshape and write into `img`
        batch_height = end - start  # number of pixels here
        # convert flat indices [start:end] → 2D coords
        # “row” = (flat_index) // x_dim,  “col” = (flat_index) % x_dim
        flat_indices = np.arange(start, end)
        rows = flat_indices // x_dim
        cols = flat_indices % x_dim

        if channels == 1:
            img[rows, cols] = preds[:, 0]
        else:
            img[rows, cols, :] = preds  # shape matches

    # If 3 channels, interpret as HSV → RGB
    if channels == 3:
        img = hsv2rgb(img)

    # Finally convert [0,1] float32 into [0,255] uint8
    img_uint8 = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img_uint8


def create_image(model, params, size):
    x_dim, y_dim = size
    X = np.concatenate(np.array(params), axis=1)

    pred = model.predict(X)

    img = []
    channels = pred.shape[1]
    for channel in range(channels):
        yp = pred[:, channel]
        yp = (yp - yp.min()) / (yp.max()-yp.min())
        img.append(yp.reshape(y_dim, x_dim))
    img = np.dstack(img)

    if channels == 3: img = hsv2rgb(img)
    img = (img * 255).astype(np.uint8)

    return img


def plot_images(images):
    """Plots the given images with pyplot (max 9)."""
    n = min(len(images), 9)
    rows = int(math.sqrt(n))
    cols = n // rows
    fig = plt.figure()
    for i in range(1, n+1):
        image = images[i-1]
        fig.add_subplot(rows, cols, i)
        plt.axis("off")
        plt.imshow(image)
    plt.show()


def save_image(image, results_dir, postfix = ""):
    """Saves given image-array under the given path."""
    if not os.path.exists(results_dir): os.makedirs(results_dir)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    image_name = "img.{}{}.png".format(timestr, postfix)
    image_path = os.path.join(results_dir, image_name)
    file = Image.fromarray(image)
    file.save(image_path)
    return image_path



if __name__ == "__main__":
    import argparse
    import os
    import time
    import numpy as np
    import tensorflow as tf
    from tqdm import tqdm
    from PIL import Image
    from skimage.color import hsv2rgb

    def save_image(image, results_dir, postfix=""):
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        img_name = f"mandala.{timestr}{postfix}.png"
        path = os.path.join(results_dir, img_name)
        Image.fromarray(image).save(path)
        return path

    parser = argparse.ArgumentParser(description="Generate mandala‐style images with Polar‐SIREN")
    parser.add_argument('--x',       type=int,   default=800, help="Image width")
    parser.add_argument('--y',       type=int,   default=800, help="Image height")
    parser.add_argument('-n', '--num',    type=int,   default=1,     help="Number of seeds to run")
    parser.add_argument('--series', type=int,   default=2,     help="Number of latitude steps per seed")
    parser.add_argument('--k',       type=int,   default=8,     help="k‐fold rotational symmetry")
    parser.add_argument('--width',   type=int,   default=128,   help="Hidden layer width")
    parser.add_argument('--depth',   type=int,   default=6,     help="Number of SIREN layers")
    parser.add_argument('--omega',   type=float, default=30.0,  help="First‐layer frequency ω₀")
    parser.add_argument('--seed',    type=int,   default=100,   help="Base RNG seed")
    parser.add_argument('-s', '--save', action='store_true',     help="Save images to disk")
    parser.add_argument('-p', '--plot', action='store_true',     help="Plot output grid at end")
    parser.add_argument('--path',    type=str,   default="./results", help="Save directory")
    args = parser.parse_args()

    images = []
    for i in range(args.num):
        tf.random.set_seed(args.seed + i)
        np.random.seed(args.seed + i)

        # random “latitude” offset for each seed
        lat0 = np.random.normal(0, 1, 1)

        # build the Polar‐SIREN with k‐fold symmetry, extras = [lat, r]
        model = build_model_polar_siren(
            n_extra=2,
            k=args.k,
            width=args.width,
            depth=args.depth,
            ω0=args.omega
        )

        for j in range(args.series):
            lat = lat0 + 2.0 * j / args.series
            # create_grid returns (x_flat, y_flat, lat_flat, r_flat)
            params = create_grid((args.x, args.y), lat)
            # create_image flattens & streams through model + hsv2rgb + normalize
            img = create_image(model, params, (args.x, args.y)).squeeze()
            images.append(img)

            if args.save:
                postfix = f".seed{args.seed+i}.step{j}"
                path = save_image(img, args.path, postfix)
                print(f"Saved: {path}")

    if args.plot:
        plot_images(images)
