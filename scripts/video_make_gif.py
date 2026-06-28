"""Render the escalator background-subtraction comparison as an animated GIF:
each frame shows original | classical bg | classical fg | unfolded bg |
unfolded fg, across all frames. Uses the trained model from Stage 2."""
import sys
sys.path.insert(0, ".")
from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image, ImageDraw

from deep_unfolded_rpca import DeepUnfoldedRPCA, classical_rpca
from deep_unfolded_rpca.video import load_escalator_matrix

OUT = Path("outputs")
DS, T, K, UP = 64, 100, 15, 3

D, _ = load_escalator_matrix("data/escalator_data.mat", ds=DS, n_frames=T)
model = DeepUnfoldedRPCA(n_layers=K, n=T)
model.load_state_dict(torch.load(OUT / "model_video_unfolded.pt", map_location="cpu"))
model.eval()
with torch.no_grad():
    Lu, Su = (x.squeeze(0) for x in model(D.unsqueeze(0)))
    Lc, Sc = (x.squeeze(0) for x in classical_rpca(
        D.unsqueeze(0), alpha=1.0, tau_L=10.0, tau_S=0.05, n_iter=200))

D, Lu, Su, Lc, Sc = (t.numpy() for t in (D, Lu, Su, Lc, Sc))
gray, hot = colormaps["gray"], colormaps["hot"]
fg_vmax = float(np.percentile(np.abs(np.concatenate([Sc, Su])), 99.0))
labels = ["original", "classical bg", "classical fg", "unfolded bg", "unfolded fg"]


def frame_img(t):
    def cell(v):
        return v[:, t].reshape(DS, DS)

    def g(a):
        return (gray(np.clip(a, 0, 1))[..., :3] * 255).astype(np.uint8)

    def h(a):
        return (hot(np.clip(np.abs(a) / (fg_vmax + 1e-9), 0, 1))[..., :3] * 255).astype(np.uint8)

    parts = [g(cell(D)), g(cell(Lc)), h(cell(Sc)), g(cell(Lu)), h(cell(Su))]
    sep = np.full((DS, 2, 3), 255, np.uint8)
    strip = parts[0]
    for p in parts[1:]:
        strip = np.concatenate([strip, sep, p], axis=1)
    img = Image.fromarray(strip).resize((strip.shape[1] * UP, strip.shape[0] * UP), Image.NEAREST)
    canvas = Image.new("RGB", (img.width, img.height + 18), (255, 255, 255))
    canvas.paste(img, (0, 18))
    d = ImageDraw.Draw(canvas)
    for i, lab in enumerate(labels):
        d.text((i * (DS + 2) * UP + 4, 4), lab, fill=(0, 0, 0))
    d.text((img.width - 58, 4), f"frame {t}", fill=(0, 0, 0))
    return canvas


frames = [frame_img(t) for t in range(T)]
gif = OUT / "video_compare.gif"
frames[0].save(gif, save_all=True, append_images=frames[1:], duration=100, loop=0)
frames[45].save(OUT / "video_compare_frame.png")
print(f"saved {gif} ({len(frames)} frames) and video_compare_frame.png")
