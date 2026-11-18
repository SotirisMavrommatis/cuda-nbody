"""Render frames.bin (written by nbody.exe) into an animated GIF and a still PNG.

Usage:  python render.py [frames.bin] [galaxy_collision.gif]
Needs numpy, matplotlib and pillow (all included with Anaconda).
"""
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")  # render to files, no window
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

src = sys.argv[1] if len(sys.argv) > 1 else "frames.bin"
out = sys.argv[2] if len(sys.argv) > 2 else "galaxy_collision.gif"

with open(src, "rb") as f:
    n, frames = (int(v) for v in np.fromfile(f, dtype=np.int32, count=2))
    data = np.fromfile(f, dtype=np.float32).reshape(frames, n, 2)
print(f"Loaded {frames} frames of {n} bodies from {src}")

# First half of the bodies is galaxy 1 (blue), second half galaxy 2 (orange).
half = n // 2
colors = np.zeros((n, 4))
colors[:half] = (0.45, 0.75, 1.0, 0.7)
colors[half:] = (1.0, 0.65, 0.3, 0.7)

fig = plt.figure(figsize=(6, 6), dpi=100, facecolor="black")
ax = fig.add_axes([0, 0, 1, 1], facecolor="black")
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_aspect("equal")
ax.axis("off")

scat = ax.scatter(data[0, :, 0], data[0, :, 1], s=0.8, c=colors, linewidths=0)
label = ax.text(0.03, 0.97, "", transform=ax.transAxes, color="white",
                fontsize=10, va="top", family="monospace")


def update(k):
    scat.set_offsets(data[k])
    label.set_text(f"N = {n}   frame {k + 1}/{frames}")
    return scat, label


anim = FuncAnimation(fig, update, frames=frames, interval=33, blit=True)
anim.save(out, writer=PillowWriter(fps=30))
print(f"Saved {out}")

# A still frame from the middle of the collision, handy for the README.
update(frames * 2 // 3)
still = out.rsplit(".", 1)[0] + ".png"
fig.savefig(still, facecolor="black")
print(f"Saved {still}")
