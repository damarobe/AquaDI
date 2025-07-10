# -*- coding: utf-8 -*-
#
# Abstract Art with ML - Batched Hypernetwork Script
# Refactored hypernetwork as subclassed Layer to track Variables
# Based on https://github.com/wottpal/cppn-keras

import os
import time
import math
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
from skimage.color import hsv2rgb
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

class HyperMLP(layers.Layer):
    def __init__(self, input_dim, hidden_dims, out_dim, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        # hyper-networks to generate weight matrices and biases
        self.weight_nets = []
        self.bias_nets = []
        prev_dim = input_dim
        for i, h in enumerate(hidden_dims + [out_dim]):
            # weight generator: maps seed (1) -> prev_dim * h
            self.weight_nets.append(layers.Dense(prev_dim * h, name=f"w{i}_net"))
            # bias generator: maps seed -> h
            self.bias_nets.append(layers.Dense(h, name=f"b{i}_net"))
            prev_dim = h

    def call(self, inputs):
        seed, x = inputs  # seed: (batch,1), x: (batch, input_dim)
        batch = tf.shape(x)[0]
        out = x
        prev_dim = self.input_dim
        # iterate through layers
        for i, (w_net, b_net) in enumerate(zip(self.weight_nets, self.bias_nets)):
            # generate weights and biases
            w = w_net(seed)           # (batch, prev_dim*h)
            b = b_net(seed)           # (batch, h)
            h = self.hidden_dims[i] if i < len(self.hidden_dims) else self.out_dim
            # reshape w to (batch, prev_dim, h)
            w = tf.reshape(w, (batch, prev_dim, h))
            # apply to out: batch matmul
            out = tf.matmul(tf.expand_dims(out, 1), w)  # (batch,1,h)
            out = tf.squeeze(out, 1) + b                 # (batch,h)
            # activation except last
            if i < len(self.hidden_dims):
                out = tf.nn.relu(out)
            prev_dim = h
        return out  # (batch, out_dim)


def build_model_hyper(input_dim, hidden_dims, out_dim):
    # Inputs
    seed_in = layers.Input(shape=(1,), name="hyper_seed")
    pixel_in = layers.Input(shape=(input_dim,), name="pixel_input")
    # Hyper MLP
    color_out = HyperMLP(input_dim=input_dim,
                         hidden_dims=hidden_dims,
                         out_dim=out_dim,
                         name="hyper_mlp")([seed_in, pixel_in])
    return models.Model(inputs=[seed_in, pixel_in], outputs=color_out, name="hyper_model")


def create_image_batched(model, size, seed_value, batch_size=65536):
    x_dim, y_dim = size
    num_pixels = x_dim * y_dim
    # Prepare output buffer
    preds = np.zeros((num_pixels, model.output_shape[-1]), dtype=np.float32)
    # process in batches
    for start in range(0, num_pixels, batch_size):
        end = min(start + batch_size, num_pixels)
        idx = np.arange(start, end, dtype=np.int32)
        # compute pixel coords
        xs = (idx % x_dim) / (x_dim - 1) * 2 - 1
        ys = (idx // x_dim) / (y_dim - 1) * 2 - 1
        rs = np.sqrt(xs**2 + ys**2)[:, None]
        lats = np.arctan2(ys, xs)[:, None] / math.pi
        pixel_feats = np.stack([xs, ys, lats[:,0], rs[:,0]], axis=1).astype(np.float32)
        seed_batch = np.full((end-start, 1), seed_value, dtype=np.float32)
        preds[start:end] = model.predict([seed_batch, pixel_feats], verbose=0)
    # normalize
    mins = preds.min(axis=0)
    maxs = preds.max(axis=0)
    norm = (preds - mins) / (maxs - mins + 1e-8)
    # reshape to image
    img = norm.reshape((y_dim, x_dim, -1))
    return img


def save_image(image, results_dir, postfix = ""):
    """Saves given image-array under the given path."""
    if not os.path.exists(results_dir): os.makedirs(results_dir)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    image_name = "img.{}{}.png".format(timestr, postfix)
    image_path = os.path.join(results_dir, image_name)
    
    rgb = hsv2rgb(image)
    file = Image.fromarray((rgb * 255).astype(np.uint8))

    file.save(image_path)
    return image_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=int, default=800)
    parser.add_argument("--y", type=int, default=800)
    parser.add_argument("--variance", type=float, default=None)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--path", type=str, default="./my_results")
    parser.add_argument("--batch_size", type=int, default=9128)
    args = parser.parse_args()

    seed_val = args.variance or np.random.uniform(50, 150)
    model = build_model_hyper(input_dim=4,
                              hidden_dims=[128, 32, 1024, 32, 96],
                              out_dim=3)

    print(f"Generating with seed {seed_val:.2f}...")
    image = create_image_batched(model,
                                 (args.x, args.y),
                                 seed_value=seed_val,
                                 batch_size=args.batch_size)

    if args.save:
        os.makedirs(os.path.dirname(args.path), exist_ok=True)
        save_image(image, args.path)
        print(f"Saved to {args.path}")
    else:
        plt.figure(figsize=(6,6))
        plt.axis('off')
        plt.imshow(hsv2rgb(image))
        plt.show()

if __name__ == '__main__':
    main()
