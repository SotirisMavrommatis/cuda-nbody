"""Render frames.bin (written by nbody.exe) into an animated GIF and a still PNG.

Usage:  python render.py [frames.bin] [output.gif] [title]
Needs numpy, matplotlib and pillow (all included with Anaconda).
"""
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")  # render to files, no window
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

src = sys.argv[1] if len(sys.argv) > 1 else "frames.bin"
out = sys.argv[2] if len(sys.argv) > 2 else "simulation.gif"
title = sys.argv[3] if len(sys.argv) > 3 else ""

GALAXY, CLUSTER, BODY = 0, 1, 2

with open(src, "rb") as f:
    n, frames, num_objects = (int(v) for v in np.fromfile(f, dtype=np.int32, count=3))
    view = float(np.fromfile(f, dtype=np.float32, count=1)[0])
    types = np.fromfile(f, dtype=np.int32, count=num_objects)
    group = np.fromfile(f, dtype=np.int32, count=n)
    data = np.fromfile(f, dtype=np.float32).reshape(-1, n, 2)
frames = data.shape[0]
print(f"Loaded {frames} frames of {n} bodies ({num_objects} objects) from {src}")

# One colour per object. Single heavy bodies are drawn as larger white dots.
palette = [
    (0.45, 0.75, 1.00),  # blue
    (1.00, 0.65, 0.30),  # orange
    (0.55, 1.00, 0.55),  # green
    (1.00, 0.45, 0.70),  # pink
    (0.80, 0.60, 1.00),  # purple
    (1.00, 0.95, 0.45),  # yellow
    (0.40, 1.00, 0.95),  # cyan
    (1.00, 0.55, 0.50),  # red
]
is_body = types[group] == BODY
stars = ~is_body
colors = np.array([palette[g % len(palette)] + (0.7,) for g in group])

# Smaller dots when there are many bodies, so the picture doesn't turn into a blob.
dot = float(np.clip(6500.0 / max(n, 1), 0.3, 2.5))

fig = plt.figure(figsize=(6, 6), dpi=100, facecolor="black")
ax = fig.add_axes([0, 0, 1, 1], facecolor="black")
ax.set_xlim(-view, view)
ax.set_ylim(-view, view)
ax.set_aspect("equal")
ax.axis("off")

scat = ax.scatter(data[0, stars, 0], data[0, stars, 1], s=dot, c=colors[stars], linewidths=0)
heavy = ax.scatter(data[0, is_body, 0], data[0, is_body, 1], s=28, c="white",
                   edgecolors=(1, 1, 1, 0.35), linewidths=4)
label = ax.text(0.03, 0.97, "", transform=ax.transAxes, color="white",
                fontsize=10, va="top", family="monospace")


def update(k):
    scat.set_offsets(data[k, stars])
    heavy.set_offsets(data[k, is_body] if is_body.any() else np.empty((0, 2)))
    name = f"{title}   " if title else ""
    label.set_text(f"{name}N = {n}   frame {k + 1}/{frames}")
    return scat, heavy, label


anim = FuncAnimation(fig, update, frames=frames, interval=33, blit=True)
anim.save(out, writer=PillowWriter(fps=30))
print(f"Saved {out}")

# A still frame from two thirds of the way through, handy for the README.
update(frames * 2 // 3)
still = out.rsplit(".", 1)[0] + ".png"
fig.savefig(still, facecolor="black")
print(f"Saved {still}")
