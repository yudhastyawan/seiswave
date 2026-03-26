#!/usr/bin/env python3
"""
PySeisSynth — Example Driver Script

Reproduces the user's CPS workflow using the pure Python library:
  sprep96 → sdisp96 → slegn96/sregn96 → spulse96

Uses the exact earth model and acquisition parameters provided
by the user.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import the library
from PySeisSynth import LayeredModel, compute_greens


def main():
    # ==========================================================
    #  1. DEFINE THE EARTH MODEL
    # ==========================================================
    nu = 0.40  # Poisson's ratio

    layers = [
        (0.0005, 0.0500),
        (0.0008, 0.1000),
        (0.0020, 0.1500),
        (0.0017, 0.2000),
        (0.0010, 0.1500),
        (0.0020, 0.2500),
        (0.0030, 0.4500),
        (0.0050, 0.3000),
        (0.0030, 0.2500),
        (0.0120, 0.4000),
        (0.0000, 1.2000),  # halfspace (thickness=0)
    ]

    model = LayeredModel.from_h_vs(layers, nu=nu)
    print(model)
    print()

    # Print model table
    print(" H(KM)   VP(KM/S)  VS(KM/S)  RHO(GM/CC)")
    for i in range(model.nlayers):
        print(f"{model.h[i]:7.4f}  {model.vp[i]:8.4f}  {model.vs[i]:8.4f}  "
              f"{model.rho[i]:8.2f}")
    print()

    # ==========================================================
    #  2. DEFINE ACQUISITION PARAMETERS
    # ==========================================================
    start = 0.004     # first offset (km)
    step = 0.003      # offset step (km)
    noffsets = 24      # number of receivers

    offsets = np.arange(noffsets) * step + start  # km

    dt = 0.002        # sampling interval (s)
    npts = 451        # number of time samples

    print(f"Offsets: {offsets[0]*1000:.1f} m to {offsets[-1]*1000:.1f} m  "
          f"(step={step*1000:.1f} m, N={noffsets})")
    print(f"Time:   dt={dt} s, npts={npts}, T_max={dt*(npts-1):.3f} s")
    print()

    # ==========================================================
    #  3. COMPUTE SYNTHETIC SEISMOGRAMS
    # ==========================================================
    data = compute_greens(
        model,
        offsets,
        dt=dt,
        npts=npts,
        nmodes=100,
        wave_types=('rayleigh',),
        pulse_type='parabolic',
        ntau=1,
    )

    print(f"\nOutput matrix shape: {data.shape}  (time × offset)")
    print(f"  npts = {data.shape[0]}")
    print(f"  noffsets = {data.shape[1]}")

    # ==========================================================
    #  4. PLOT: Vs PROFILE
    # ==========================================================
    depth_top, depth_bot = model.depth_profile
    depth_plot = []
    vs_plot = []
    for i in range(model.nlayers):
        depth_plot += [depth_top[i] * 1000, depth_bot[i] * 1000]  # km → m
        vs_plot += [model.vs[i] * 1000, model.vs[i] * 1000]       # km/s → m/s

    plt.figure(figsize=(5, 8))
    plt.plot(vs_plot, depth_plot, '-b', linewidth=3)
    plt.ylim(0, 40)
    plt.gca().invert_yaxis()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.xlabel("Vs (m/s)", fontsize=12)
    plt.ylabel("Depth (m)", fontsize=12)
    plt.title("Vs Vertical Profile", fontsize=14)
    plt.tight_layout()
    plt.savefig("vs_profile.png", dpi=150)
    plt.show()

    # ==========================================================
    #  5. PLOT: NORMALIZED SHOT GATHER
    # ==========================================================
    x = offsets * 1000  # km → m
    t = np.arange(npts) * dt

    # Normalize per trace
    max_per_trace = np.max(np.abs(data), axis=0) + 1e-20
    data_norm = data / max_per_trace

    plt.figure(figsize=(12, 6))
    plt.imshow(data_norm, aspect='auto', cmap='gray',
               extent=[x[0], x[-1], t[-1], 0])
    plt.title("Shot Gather — PySeisSynth (Normalized per Trace)")
    plt.xlabel("Offset (m)")
    plt.ylabel("Time (s)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("shot_gather.png", dpi=150)
    plt.show()

    # ==========================================================
    #  6. PLOT: WIGGLE PLOT
    # ==========================================================
    plt.figure(figsize=(12, 6))
    for i in range(noffsets):
        plt.plot(data_norm[:, i] + x[i], t, 'k', linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.xlabel("Offset (m)")
    plt.ylabel("Time (s)")
    plt.title("Wiggle Plot — PySeisSynth (Normalized per Trace)")
    plt.tight_layout()
    plt.savefig("wiggle_plot.png", dpi=150)
    plt.show()

    print("\nDone! Plots saved: vs_profile.png, shot_gather.png, wiggle_plot.png")
    return data


if __name__ == "__main__":
    data = main()
