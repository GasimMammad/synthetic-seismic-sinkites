"""
Synthetic seismic model of a sinkite / floatite mound field.

Builds a 2D acoustic impedance (AI) section with:
  - layered overburden that drapes over the mounds,
  - floatites: ooze rafts pushed up into the mound cores,
  - sinkites: sand bodies that sank into the ooze below,
  - polygonal faults in the ooze-rich interval,
then makes a convolutional (Ricker) seismic section from it and a smoothed,
band-limited version of the AI that stands in for an inversion result.

It is a cartoon for illustration and teaching, not a calibrated model of any
real field. Units are nominal: 1 sample = 1 ms TWT, 1 trace = 4 m.

usage:
    python synthetic_sinkites.py figure            # 3-panel figure
    python synthetic_sinkites.py banner --style dark
    python synthetic_sinkites.py banner --style light
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import fftconvolve

DT_MS = 1.0   # ms per sample
DX_M = 4.0    # m per trace


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def smooth_noise(rng, n, sigma, amp=1.0):
    v = gaussian_filter1d(rng.normal(0, 1, n), sigma)
    return v / (v.std() + 1e-9) * amp


def dome(x, c, wl, wr):
    """Asymmetric bump: 1 at c, falling to 0 at c - wl and c + wr."""
    u = np.where(x < c, (x - c) / wl, (x - c) / wr)
    u = np.clip(u, -1, 1)
    return np.cos(0.5 * np.pi * u) ** 2


def relative_time(ifaces, nz):
    """Fractional layer index at every sample, so thin beds follow the layering."""
    nlay, nx = ifaces.shape
    T = np.empty((nz, nx))
    zs = np.arange(nz, dtype=float)
    idx = np.arange(nlay, dtype=float)
    for j in range(nx):
        T[:, j] = np.interp(zs, ifaces[:, j], idx, left=-1, right=nlay)
    return T


def ricker(f, half_len=45):
    """Ricker wavelet, f in cycles per sample."""
    t = np.arange(-half_len, half_len + 1)
    a = (np.pi * f * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
# mound centre (fraction of width), left / right half-widths (traces), relief
MOUNDS = [
    (0.285, 120, 165, 0.85),
    (0.405, 95, 80, 0.6),
    (0.52, 170, 130, 1.1),
    (0.655, 110, 140, 0.8),
    (0.77, 150, 120, 1.0),
    (0.905, 135, 170, 0.95),
]


def build_model(nx=3168, nz=792, seed=11):
    rng = np.random.default_rng(seed)
    x = np.arange(nx, dtype=float)
    Z = np.arange(nz, dtype=float)[:, None]

    top_shape = np.zeros(nx)
    keel_shape = np.zeros(nx)
    for c, wl, wr, amp in MOUNDS:
        c *= nx
        top_shape = np.maximum(top_shape, amp * dome(x, c, wl, wr))
        keel_shape = np.maximum(keel_shape, amp * dome(x, c, 0.7 * wl, 0.7 * wr) ** 0.7)

    dip = 0.010 * (x - nx / 2)
    zs, zb, zo = 455.0, 492.0, 640.0   # sand top, sand base, base of ooze
    relief, keel = 62.0, 105.0         # mound relief above, sinkite keel below

    mound_top = zs - relief * top_shape + dip + smooth_noise(rng, nx, 25, 1.5)
    sink_top = zs + 0.55 * keel * keel_shape + dip
    sink_base = zb + keel * keel_shape + dip + smooth_noise(rng, nx, 18, 2.0)

    # overburden, draped over the mounds with the drape dying out upwards
    n_ob = 30
    gaps = rng.uniform(0.6, 1.6, n_ob)
    t0 = 15 + np.cumsum(gaps) / gaps.sum() * (zs - 30)
    ob = np.empty((n_ob, nx))
    for k in range(n_ob):
        frac = k / (n_ob - 1)
        drape = -relief * top_shape * frac ** 1.8
        ob[k] = t0[k] + dip * frac + drape + smooth_noise(rng, nx, 140, 6 * (1 - frac) + 1)
    ob = np.minimum(ob, mound_top[None, :] - 3)
    ob = np.maximum.accumulate(ob, axis=0)

    # deeper section below the ooze
    n_dp = 14
    dp = np.empty((n_dp, nx))
    for k, d0 in enumerate(np.linspace(zo + 8, nz + 40, n_dp)):
        dp[k] = d0 + dip * 1.4 + smooth_noise(rng, nx, 200, 4)
    dp = np.maximum.accumulate(dp, axis=0)

    T_ob = relative_time(ob, nz)
    T_dp = relative_time(dp, nz)

    fine = gaussian_filter1d(rng.normal(0, 1, 4000), 1.3)
    fine /= fine.std()

    def thin_beds(T, scale):
        return np.interp(T * scale, np.arange(fine.size), fine)

    coarse_ob = 4.2 + np.linspace(0, 0.4, n_ob + 2) + rng.uniform(-0.22, 0.22, n_ob + 2)
    coarse_dp = 4.45 + np.linspace(0, 0.3, n_dp + 2) + rng.uniform(-0.25, 0.25, n_dp + 2)
    lateral = smooth_noise(rng, nx, 300, 0.05)[None, :]

    ai_ob = coarse_ob[np.clip(T_ob.astype(int) + 1, 0, n_ob + 1)] + 0.11 * thin_beds(T_ob, 22) + lateral
    ai_dp = coarse_dp[np.clip(T_dp.astype(int) + 1, 0, n_dp + 1)] + 0.12 * thin_beds(T_dp + 50, 18) + lateral

    # polygonal faults in the ooze: inclined, alternating dip, small throws
    fx = np.cumsum(rng.uniform(55, 115, 60)) - 60
    fdir = rng.choice([-1, 1], fx.size)
    throw = np.where(np.arange(fx.size) % 2 == 0, 1, -1) * rng.uniform(4, 9, fx.size)
    zref = zb + 40
    off = np.zeros((nz, nx))
    for xi, d, th in zip(fx, fdir, throw):
        off += th * (x[None, :] > xi + d * 0.55 * (Z - zref))
    off -= gaussian_filter1d(off[int(zref)], 200)[None, :]

    # floatites: ooze rafts broken into tilted blocks
    edges = np.cumsum(rng.uniform(45, 110, 80)).astype(int)
    block_tilt = np.zeros(nx)
    start = 0
    for e in edges:
        block_tilt[start:min(e, nx)] = rng.uniform(-0.45, 0.45)
        start = e
        if start >= nx:
            break
    ai_float = 3.75 + 0.09 * np.sin(2 * np.pi * (Z + block_tilt[None, :] * (x[None, :] % 97)) / 11.0)
    ai_float += gaussian_filter(rng.normal(0, 1, (nz, nx)), (1.5, 4)) * 0.08

    ai_ooze = 3.62 + 0.11 * np.sin(2 * np.pi * (Z - off) / 14.0)

    # put it together
    in_float = (Z >= mound_top[None, :]) & (Z < sink_top[None, :])
    in_sand = (Z >= np.minimum(mound_top, sink_top)[None, :]) & (Z < sink_base[None, :]) & ~in_float
    ooze_base = zo + dip[None, :] + 0.5 * off[int(zo) - 1][None, :]
    in_ooze = (Z >= sink_base[None, :]) & (Z < ooze_base)
    in_deep = Z >= ooze_base

    ai = ai_ob.copy()
    ai = np.where(in_float, ai_float, ai)
    ai = np.where(in_sand, 5.0 + 0.08 * gaussian_filter(rng.normal(0, 1, (nz, nx)), (2, 6)), ai)
    ai = np.where(in_ooze, ai_ooze, ai)
    ai = np.where(in_deep, ai_dp, ai)
    ai = gaussian_filter(ai, (0.6, 1.0))

    return {
        "ai": ai,
        "mound_top": mound_top,
        "sink_base": sink_base,
        "well_x": int(MOUNDS[-1][0] * nx),
        "rng": rng,
    }


def synthetic_seismic(ai, rng, f=0.042):
    """Normal-incidence reflectivity convolved with a Ricker wavelet."""
    refl = np.zeros_like(ai)
    refl[1:] = np.diff(ai, axis=0) / (ai[1:] + ai[:-1])
    seis = fftconvolve(refl, ricker(f)[:, None], mode="same")
    seis *= 1 + 0.25 * smooth_noise(rng, ai.shape[1], 60)[None, :]
    seis += gaussian_filter(rng.normal(0, 1, seis.shape), (1.8, 3.5)) * 0.004
    seis = gaussian_filter(seis, (0, 0.8))
    seis /= np.percentile(np.abs(seis), 99.2)
    return np.clip(seis, -1, 1)


def pseudo_inversion(ai, rng):
    """Smoothed, slightly noisy AI: roughly what a band-limited inversion recovers."""
    return gaussian_filter(ai, (3.5, 2.5)) + gaussian_filter(rng.normal(0, 1, ai.shape), (3, 8)) * 0.03


def ai_to_unit(ai_inv):
    """Map AI to 0..1, half linear and half histogram-equalised, so no band saturates."""
    lo, hi = np.percentile(ai_inv, [1, 99])
    lin = np.clip((ai_inv - lo) / (hi - lo), 0, 1)
    ranks = np.empty(ai_inv.size)
    ranks[np.argsort(ai_inv, axis=None)] = np.linspace(0, 1, ai_inv.size)
    return 0.45 * lin + 0.55 * ranks.reshape(ai_inv.shape)


# ---------------------------------------------------------------------------
# colours
# ---------------------------------------------------------------------------
PALETTES = {
    "dark": dict(
        seis=["#cfe6fa", "#4382c4", "#121a24", "#d49a4a", "#fae3b6"],
        ai=["#0d1f38", "#1d4a6e", "#22857f", "#bfae68", "#f0c47c"],
        pick_top="#ffd28a", pick_base="#8fd3ff", line="#f2f2f2",
    ),
    "light": dict(
        seis=["#173a87", "#7f9fd4", "#f5f3ef", "#d88d7e", "#8c1c1c"],
        ai=["#1f3b4d", "#2a9d8f", "#9cc5a1", "#e9c46a", "#e8a25e"],
        pick_top="#1a1a1a", pick_base="#14507d", line="#1a1a1a",
    ),
}


def cmaps(style):
    p = PALETTES[style]
    return (LinearSegmentedColormap.from_list("seis", p["seis"]),
            LinearSegmentedColormap.from_list("ai", p["ai"]), p)


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------
def plot_figure(model, seis, ai_inv, out="figures/synthetic_sinkites.png"):
    nz, nx = model["ai"].shape
    ext = (0, nx * DX_M / 1000, nz * DT_MS, 0)
    seis_cmap, ai_cmap, _ = cmaps("light")
    km = np.arange(nx) * DX_M / 1000

    fig, axes = plt.subplots(3, 1, figsize=(12, 9.5), sharex=True, constrained_layout=True)
    panels = [
        (model["ai"], "viridis", "a) Acoustic impedance model", "AI (nominal)"),
        (seis, seis_cmap, "b) Synthetic seismic (Ricker, ~42 Hz)", "Amplitude"),
        (ai_to_unit(ai_inv), ai_cmap, "c) Band-limited impedance (pseudo-inversion)", "Relative AI"),
    ]
    for ax, (img, cmap, title, clabel) in zip(axes, panels):
        vmin, vmax = (-1, 1) if img is seis else (None, None)
        im = ax.imshow(img, cmap=cmap, aspect="auto", extent=ext, vmin=vmin, vmax=vmax,
                       interpolation="bilinear")
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylabel("TWT (ms)")
        fig.colorbar(im, ax=ax, label=clabel, pad=0.01, shrink=0.9)

    ax = axes[1]
    ax.plot(km, model["mound_top"], color="#1a1a1a", lw=1.0, label="Mound top")
    ax.plot(km, model["sink_base"], color="#14507d", lw=1.0, ls="--", label="Sinkite base")
    ax.legend(loc="lower left", fontsize=8, frameon=True)

    axes[-1].set_xlabel("Distance (km)")
    fig.savefig(out, dpi=150)
    print("saved", out)


def render_banner(model, seis, ai_inv, style="dark", out=None, size=(1584, 396)):
    out = out or f"figures/banner_{style}.png"
    nz, nx = model["ai"].shape
    seis_cmap, ai_cmap, p = cmaps(style)
    x = np.arange(nx, dtype=float)

    rgb_s = seis_cmap(0.5 + 0.5 * seis)[..., :3]
    rgb_a = ai_cmap(ai_to_unit(ai_inv))[..., :3]

    # blend amplitude into impedance across a soft seam
    u = np.clip((x / nx - 0.60) / 0.13, 0, 1)
    blend = u * u * (3 - 2 * u)
    rgb = rgb_s * (1 - blend[None, :, None]) + rgb_a * blend[None, :, None]
    if style == "dark":
        yy = (np.arange(nz) / nz)[:, None]
        rgb *= np.clip(1 - 0.9 * (yy - 0.55) ** 2, 0.65, 1)[..., None]

    dpi = 200
    fig = plt.figure(figsize=(nx / dpi, nz / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(np.clip(rgb, 0, 1), aspect="auto", interpolation="bilinear", extent=(0, nx, nz, 0))

    # interpretation picks on the seismic side, fading out over the seam
    x0 = int(0.20 * nx)
    for i in range(x0, int(nx * 0.76), 6):
        a = (1 - blend[i]) * min(1, (i - x0) / 120)
        if a < 0.02:
            continue
        j = slice(i, i + 7)
        ax.plot(x[j], model["mound_top"][j] - 2, color=p["pick_top"], lw=1.5, alpha=0.8 * a)
        ax.plot(x[j], model["sink_base"][j] + 2, color=p["pick_base"], lw=1.5, alpha=0.8 * a)

    # a well through the last mound, with a GR-style curve next to it
    wx = model["well_x"]
    zz = np.arange(nz)
    keep = zz < nz * 0.9
    ax.plot([wx, wx], [0, zz[keep][-1]], color=p["line"], lw=1.3, alpha=0.75)
    gr = -(model["ai"][:, wx] - np.median(model["ai"][:, wx]))
    gr = gaussian_filter1d(gr, 1.5) * 60 + gaussian_filter1d(model["rng"].normal(0, 1, nz), 1.2) * 5
    gr = gr[keep]
    ax.plot(wx + 14 + gr - gr.min(), zz[keep], color=p["line"], lw=0.9, alpha=0.7)

    ax.set_xlim(0, nx)
    ax.set_ylim(nz, 0)
    ax.axis("off")
    tmp = out + ".tmp.png"
    fig.savefig(tmp, dpi=dpi)
    plt.close(fig)

    from PIL import Image
    Image.open(tmp).convert("RGB").resize(size, Image.LANCZOS).save(out, optimize=True)
    import os
    os.remove(tmp)
    print("saved", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=["figure", "banner"])
    ap.add_argument("--style", choices=["dark", "light"], default="dark")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("-o", "--out")
    args = ap.parse_args()

    model = build_model(seed=args.seed)
    seis = synthetic_seismic(model["ai"], model["rng"])
    ai_inv = pseudo_inversion(model["ai"], model["rng"])

    if args.what == "figure":
        plot_figure(model, seis, ai_inv, out=args.out or "figures/synthetic_sinkites.png")
    else:
        render_banner(model, seis, ai_inv, style=args.style, out=args.out)


if __name__ == "__main__":
    main()
