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
from keras.layers import Input, Dense
import tensorflow as tf
import numpy as np
from skimage.color import hsv2rgb


keras.backend.set_learning_phase(1)


from keras import models, layers, initializers
import tensorflow as tf

import tensorflow as tf
from keras import models, layers, initializers



def build_model_fractal(n, variance, bw=False, depth=5, swirl=2.0):
    inp = layers.Input((n,))         # [x,y, …]
    x = inp
    outputs = []
    for ℓ in range(depth):
        # 1) apply your standard CPPN block
        y = Dense(64, activation='tanh')(x)
        y = Dense(3, activation='tanh')(y)
        outputs.append(y)

        # 2) swirl the coordinates for next level:
        coords = x[:, :2]
        r = tf.sqrt(tf.reduce_sum(coords**2, axis=1, keepdims=True))
        θ = tf.atan2(coords[:,1:2], coords[:,0:1]) + swirl * (ℓ+1)
        x = tf.concat([r * tf.cos(θ), r * tf.sin(θ), x[:,2:]], axis=1)

    # Sum or multiply all levels:
    merged = layers.Add()(outputs)
    return models.Model(inp, merged)



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
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save', action='store_const', const=True)
    parser.add_argument('-p', '--plot', action='store_const', const=True)
    parser.add_argument('--n', type=int, nargs='?', default=1)
    parser.add_argument('--path', type=str, nargs='?', default="./results")
    parser.add_argument('--x', type=int, nargs='?', default=512)
    parser.add_argument('--y', type=int, nargs='?', default=512)
    parser.add_argument('--bw', action='store_const', const=True)
    parser.add_argument('--variance', type=float, nargs='?')
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--series', type=int, default=1)
    args = parser.parse_args()

    params = create_grid((args.x, args.y), 1.0)

    images = []
    
    for i in tqdm(range(args.n)):
        tf.random.set_seed(args.seed + i)
        np.random.seed(args.seed + i)

        variance = args.variance or np.random.uniform(50, 150)
        lat = np.random.normal(0, 1, 1)
        model = build_model_fractal(len(create_grid((args.x, args.y), lat)), variance, bw=args.bw)

        for j in tqdm(range(args.series), leave=False):
            params = create_grid((args.x, args.y), lat + 2 * j / args.series, 1.0)
            
            # Here’s the only change:
            # Specify a batch_size—e.g. 65 536 pixels per batch
            image = create_image_batched(model, params, (args.x, args.y), batch_size=65536)

            image = image.squeeze()
            images.append(image)

            if args.save:
                postfix = f".var{variance:.0f}.seed{args.seed + i}"
                image_path = save_image(image, args.path, postfix)
                tqdm.write(f"Image saved under {image_path}")

    if args.plot:
        plot_images(images)
