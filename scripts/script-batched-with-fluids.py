#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fluid-based abstract art generator using a 2D Navier–Stokes “Stable Fluids” solver.
• Seeds an initial shape (white circle) in RGB density.
• Applies random jets + rotating swirl forces each step.
• Tracks 3‐channel density → full‐color output.
"""

import os
import time
import argparse
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 1) FluidSolver2D: 3‐channel “Stable Fluids” Navier–Stokes solver
# -------------------------------------------------------------------
class FluidSolver2D:
    def __init__(self, N, diffusion=1e-4, viscosity=1e-4, dt=0.1):
        self.N = N
        self.dt = dt
        self.diff = diffusion
        self.visc = viscosity
        self.size = N + 2
        # Velocity fields
        self.u  = np.zeros((self.size, self.size), dtype=np.float32)
        self.v  = np.zeros((self.size, self.size), dtype=np.float32)
        self.u0 = np.zeros_like(self.u)
        self.v0 = np.zeros_like(self.v)
        # 3‐channel density
        self.density  = np.zeros((self.size, self.size, 3), dtype=np.float32)
        self.density0 = np.zeros_like(self.density)

    def step(self):
        N, visc, diff, dt = self.N, self.visc, self.diff, self.dt
        # --- diffuse velocity ---
        self.u0[:] = self.u
        self.v0[:] = self.v
        self._diffuse(1, self.u, self.u0, visc, dt)
        self._diffuse(2, self.v, self.v0, visc, dt)
        # --- project velocity ---
        self._project(self.u, self.v, self.u0, self.v0)
        # --- advect velocity ---
        self.u0[:] = self.u
        self.v0[:] = self.v
        self._advect(1, self.u, self.u0, self.u0, self.v0, dt)
        self._advect(2, self.v, self.v0, self.u0, self.v0, dt)
        # --- project again ---
        self._project(self.u, self.v, self.u0, self.v0)
        # --- diffuse & advect density per channel ---
        self.density0[:] = self.density
        for c in range(3):
            self._diffuse(0, self.density[:,:,c], self.density0[:,:,c], diff, dt)
        for c in range(3):
            self._advect(0, self.density[:,:,c], self.density0[:,:,c], self.u, self.v, dt)

    def add_density(self, x, y, amount):
        """Add a 3‐vector or scalar to density at (x,y)."""
        if hasattr(amount, "__len__"):
            self.density[y, x, :] += amount
        else:
            self.density[y, x, :] += amount

    def add_velocity(self, x, y, amount_u, amount_v):
        self.u[y, x] += amount_u
        self.v[y, x] += amount_v

    def _diffuse(self, b, x, x0, diff, dt):
        a = dt * diff * self.N * self.N
        self._lin_solve(b, x, x0, a, 1 + 4 * a)

    def _lin_solve(self, b, x, x0, a, c):
        N = self.N
        for _ in range(20):
            x[1:N+1,1:N+1] = (
                x0[1:N+1,1:N+1]
                + a * ( x[1:N+1,2:N+2] + x[1:N+1,0:N]
                      + x[2:N+2,1:N+1] + x[0:N,  1:N+1] )
            ) / c
            self._set_bnd(b, x)

    def _advect(self, b, d, d0, u, v, dt):
        N, dt0 = self.N, dt * self.N
        for i in range(1, N+1):
            for j in range(1, N+1):
                x = i - dt0 * u[j, i]
                y = j - dt0 * v[j, i]
                x = min(max(x, 0.5), N + 0.5)
                y = min(max(y, 0.5), N + 0.5)
                i0, j0 = int(np.floor(x)), int(np.floor(y))
                i1, j1 = i0 + 1, j0 + 1
                s1, t1 = x - i0, y - j0
                s0, t0 = 1 - s1, 1 - t1
                d[j, i] = (
                    s0 * (t0 * d0[j0, i0] + t1 * d0[j1, i0]) +
                    s1 * (t0 * d0[j0, i1] + t1 * d0[j1, i1])
                )
        self._set_bnd(b, d)

    def _project(self, u, v, p, div):
        N, h = self.N, 1.0/self.N
        div[1:N+1,1:N+1] = -0.5*h * (
            u[1:N+1,2:N+2] - u[1:N+1,0:N] +
            v[2:N+2,1:N+1] - v[0:N,  1:N+1]
        )
        p[1:N+1,1:N+1] = 0
        self._set_bnd(0, div); self._set_bnd(0, p)
        self._lin_solve(0, p, div, 1, 4)
        u[1:N+1,1:N+1] -= 0.5*( p[1:N+1,2:N+2] - p[1:N+1,0:N] )/h
        v[1:N+1,1:N+1] -= 0.5*( p[2:N+2,1:N+1] - p[0:N,  1:N+1] )/h
        self._set_bnd(1, u); self._set_bnd(2, v)

    def _set_bnd(self, b, x):
        N = self.N
        # handle 2D or 3D arrays
        if x.ndim == 3:
            for c in range(x.shape[2]):
                self._set_bnd(b, x[:,:,c])
            return
        for i in range(1, N+1):
            if b==1:
                x[0,i]   = -x[1,i]
                x[N+1,i] = -x[N,i]
            else:
                x[0,i]   = x[1,i]
                x[N+1,i] = x[N,i]
            if b==2:
                x[i,0]   = -x[i,1]
                x[i,N+1] = -x[i,N]
            else:
                x[i,0]   = x[i,1]
                x[i,N+1] = x[i,N]
        x[0,0]       = 0.5*(x[1,0]   + x[0,1])
        x[0,N+1]     = 0.5*(x[1,N+1] + x[0,N])
        x[N+1,0]     = 0.5*(x[N,0]   + x[N+1,1])
        x[N+1,N+1]   = 0.5*(x[N,N+1] + x[N+1,N])


# -------------------------------------------------------------------
# 2) build_model: returns solver + config
# -------------------------------------------------------------------
def build_model(n, variance, bw=False,
                viscosity=1e-4, diffusion=1e-4, dt=0.1, steps=200):
    """
    n        : grid resolution (n×n)
    variance : ignored (placeholder)
    bw       : if True, output single‐channel RGB=gray
    viscosity, diffusion, dt : solver params
    steps    : number of fluid timesteps to run
    """
    solver = FluidSolver2D(N=n,
                           diffusion=diffusion,
                           viscosity=viscosity,
                           dt=dt)
    return solver, steps, bw


# -------------------------------------------------------------------
# 3) create_image: initialize mask + random jets + step solver
# -------------------------------------------------------------------
def create_image(model_tuple, params, size):
    solver, steps, bw = model_tuple
    N = solver.N
    x_dim, y_dim = size

    # 1) Seed an initial white circle at center in all 3 channels
    mask = Image.new("L", (N, N), 0)
    draw = ImageDraw.Draw(mask)
    r = N // 3
    draw.ellipse((N//2-r, N//2-r, N//2+r, N//2+r), fill=255)
    mask_arr = np.array(mask, dtype=np.float32)/255.0 * 50.0
    for c in range(3):
        solver.density[1:N+1,1:N+1,c] = mask_arr

    # 2) Run timesteps with a mix of rotating swirl + random jets
    rng = np.random.RandomState(int(time.time()))
    for t in range(steps):
        # 2a) rotating swirl around center
        angle = 2*np.pi * (t/steps)
        fx = 50.0*np.cos(angle)
        fy = 50.0*np.sin(angle)
        cx, cy, rad = N//2, N//2, N//4
        for i in range(cx-rad, cx+rad):
            for j in range(cy-rad, cy+rad):
                if 1<=i<=N and 1<=j<=N and (i-cx)**2+(j-cy)**2 < rad*rad:
                    solver.add_velocity(i, j, fx, fy)

        # 2b) random jets at boundary
        for _ in range(5):
            side = rng.choice(['left','right','top','bottom'])
            if side=='left':
                i, j = 1, rng.randint(1, N+1)
                solver.add_velocity(i, j, 100.0,  0.0)
            elif side=='right':
                i, j = N, rng.randint(1, N+1)
                solver.add_velocity(i, j, -100.0, 0.0)
            elif side=='top':
                i, j = rng.randint(1, N+1), 1
                solver.add_velocity(i, j, 0.0,  100.0)
            else:
                i, j = rng.randint(1, N+1), N
                solver.add_velocity(i, j, 0.0, -100.0)

        # 2c) occasional colored puffs
        if t % 40 == 0:
            x0 = rng.randint(1, N+1)
            y0 = rng.randint(1, N+1)
            color = rng.rand(3) * 50.0
            solver.add_density(x0, y0, color)

        # Step the solver
        solver.step()

    # 3) Extract density and normalize to [0,1]
    dens = solver.density[1:N+1, 1:N+1, :]  # (N, N, 3)
    maxv = dens.max() if dens.max()>0 else 1.0
    img_f = np.clip(dens / maxv, 0.0, 1.0)

    # 4) Resize if needed
    if not (x_dim==N and y_dim==N):
        pil = Image.fromarray((img_f*255).astype(np.uint8))
        pil = pil.resize((x_dim, y_dim), resample=Image.BILINEAR)
        img_f = np.array(pil, dtype=np.float32)/255.0

    # 5) Convert to uint8 RGB or grayscale
    if bw:
        gray8 = (img_f.mean(axis=2)*255).astype(np.uint8)
        return gray8[..., np.newaxis]
    else:
        return (img_f*255).astype(np.uint8)


# -------------------------------------------------------------------
# 4) plot_images: same as before
# -------------------------------------------------------------------
def plot_images(images):
    n = min(len(images), 9)
    rows = int(np.floor(np.sqrt(n)))
    cols = int(np.ceil(n/rows))
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = np.atleast_2d(axes)
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        ax.axis('off')
        if img.ndim == 2:
            ax.imshow(img, cmap='inferno', origin='lower')
        else:
            ax.imshow(img, origin='lower')
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# 5) Main loop
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save', action='store_true')
    parser.add_argument('-p', '--plot', action='store_true')
    parser.add_argument('--n',      type=int,   default=1)
    parser.add_argument('--path',   type=str,   default="results")
    parser.add_argument('--x',      type=int,   default=256)
    parser.add_argument('--y',      type=int,   default=256)
    parser.add_argument('--bw',     action='store_true')
    parser.add_argument('--seed',   type=int,   default=None)
    parser.add_argument('--steps',  type=int,   default=200)
    parser.add_argument('--visc',   type=float, default=1e-4)
    parser.add_argument('--diff',   type=float, default=1e-4)
    parser.add_argument('--dt',     type=float, default=0.1)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    images = []
    for i in tqdm(range(args.n), desc="Generating images"):
        solver, steps, bw_flag = build_model(
            n=args.x,
            variance=0.0,
            bw=args.bw,
            viscosity=args.visc,
            diffusion=args.diff,
            dt=args.dt,
            steps=args.steps
        )
        # params unused in this physics‐based version
        img = create_image((solver, steps, bw_flag), None, (args.x, args.y))

        if args.save:
            os.makedirs(args.path, exist_ok=True)
            fname = f"fluid_{time.strftime('%Y%m%d-%H%M%S')}_{i}.png"
            out_p = os.path.join(args.path, fname)
            if img.ndim == 3:
                Image.fromarray(img).save(out_p)
            else:
                Image.fromarray(img.squeeze(), mode='L').save(out_p)
            tqdm.write(f"Saved → {out_p}")

        images.append(img)

        exit;
