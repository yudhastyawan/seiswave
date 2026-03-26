"""
Dispersion Image Module — Phase-Shift Method (Park et al., 1999)

Transforms multichannel seismic data from the time-offset (t-x) domain
to the frequency–phase velocity (f-c) domain, producing an overtone
image suitable for MASW dispersion curve picking or Full Wavefield
Inversion.

Algorithm:
    1. FFT each trace along the time axis → U(x, f)
    2. Normalize to unit amplitude  → U_norm = U / |U|
    3. For each trial phase velocity c and frequency f, compute
       the slant-stack (phase-shift summation):
           E(f, c) = | Σ_x  U_norm(x, f) · exp(i 2π f x / c) |
    4. Normalize each frequency column by its maximum.

Reference:
    Park, C.B., Miller, R.D., and Xia, J. (1999).
    Multichannel analysis of surface waves.
    Geophysics, 64(3), 800–808.
"""

import numpy as np


def calculate_dispersion_image(data, x, fs, c_min, c_max, dc,
                               f_min=1.0, f_max=50.0):
    """
    Compute a dispersion image using the phase-shift method.

    Parameters
    ----------
    data : np.ndarray, shape (npts, N)
        Multichannel seismic data (shot gather).
        Rows = time samples, Columns = receiver channels.
    x : np.ndarray, shape (N,)
        Offset distances of each receiver in **metres**.
    fs : float
        Sampling rate in Hz.
    c_min : float
        Minimum trial phase velocity (m/s).
    c_max : float
        Maximum trial phase velocity (m/s).
    dc : float
        Phase velocity increment (m/s).
    f_min : float, optional
        Minimum frequency of interest (Hz). Default 1.0.
    f_max : float, optional
        Maximum frequency of interest (Hz). Default 50.0.

    Returns
    -------
    freqs : np.ndarray, shape (Nf,)
        Frequency axis (Hz), trimmed to [f_min, f_max].
    c_arr : np.ndarray, shape (Nc,)
        Phase velocity axis (m/s).
    E : np.ndarray, shape (Nc, Nf)
        Normalized dispersion energy matrix.
        Rows = phase velocity, Columns = frequency.
    """
    npts, ntraces = data.shape
    x = np.asarray(x, dtype=np.float64)

    # ------------------------------------------------------------------
    # 1. FFT along time axis → U(x, f)
    # ------------------------------------------------------------------
    nfft = npts                         # can pad to next power-of-2 if desired
    U = np.fft.rfft(data, n=nfft, axis=0)          # shape (nfft//2+1, N)

    # Corresponding frequency bins
    freqs_full = np.fft.rfftfreq(nfft, d=1.0 / fs) # shape (nfft//2+1,)

    # Trim to [f_min, f_max]
    f_mask = (freqs_full >= f_min) & (freqs_full <= f_max)
    freqs = freqs_full[f_mask]                      # (Nf,)
    U = U[f_mask, :]                                # (Nf, N)

    Nf = len(freqs)

    # ------------------------------------------------------------------
    # 2. Normalize amplitude → pure phase: U_norm = U / |U|
    # ------------------------------------------------------------------
    amp = np.abs(U)
    amp[amp < 1.0e-30] = 1.0e-30       # avoid division by zero
    U_norm = U / amp                    # (Nf, N)

    # ------------------------------------------------------------------
    # 3. Phase velocity array
    # ------------------------------------------------------------------
    c_arr = np.arange(c_min, c_max + dc * 0.5, dc, dtype=np.float64)  # (Nc,)
    Nc = len(c_arr)

    # ------------------------------------------------------------------
    # 4. Vectorised phase-shift summation
    #
    #    E(f, c) = | Σ_x  U_norm(x, f) · exp(i 2π f x / c) |
    #
    #    Shapes used for broadcasting:
    #       freqs   → (Nf, 1, 1)
    #       x       → (1,  N, 1)      — offset axis
    #       c_arr   → (1,  1, Nc)     — velocity axis
    #       U_norm  → (Nf, N, 1)
    #
    #    phase_shift → (Nf, N, Nc)
    #    Summation over axis=1 (offsets) → (Nf, Nc)
    #    Transpose → (Nc, Nf)
    # ------------------------------------------------------------------
    freqs_3d = freqs[:, np.newaxis, np.newaxis]      # (Nf, 1,  1)
    x_3d     = x[np.newaxis, :, np.newaxis]          # (1,  N,  1)
    c_3d     = c_arr[np.newaxis, np.newaxis, :]      # (1,  1,  Nc)

    phase_shift = np.exp(1j * 2.0 * np.pi * freqs_3d * x_3d / c_3d)
    # → (Nf, N, Nc)

    U_norm_3d = U_norm[:, :, np.newaxis]             # (Nf, N, 1)

    E_raw = np.abs(np.sum(U_norm_3d * phase_shift, axis=1))
    # → (Nf, Nc)

    E = E_raw.T                                      # → (Nc, Nf)

    # ------------------------------------------------------------------
    # 5. Normalize each frequency column by its maximum
    # ------------------------------------------------------------------
    col_max = E.max(axis=0)
    col_max[col_max < 1.0e-30] = 1.0e-30
    E = E / col_max[np.newaxis, :]

    return freqs, c_arr, E


def plot_dispersion_image(freqs, c_arr, E, ax=None,
                          cmap='jet', title='Dispersion Image'):
    """
    Plot a dispersion image as a colour map.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency axis (Hz).
    c_arr : np.ndarray
        Phase velocity axis (m/s).
    E : np.ndarray, shape (Nc, Nf)
        Normalised dispersion energy matrix.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.  Created if *None*.
    cmap : str
        Colour map name.
    title : str
        Plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax  : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    extent = [freqs[0], freqs[-1], c_arr[0], c_arr[-1]]
    im = ax.imshow(E, aspect='auto', origin='lower', extent=extent,
                   cmap=cmap, vmin=0, vmax=1, interpolation='bilinear')

    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Phase Velocity (m/s)', fontsize=12)
    ax.set_title(title, fontsize=14)
    fig.colorbar(im, ax=ax, label='Normalised Energy')
    fig.tight_layout()

    return fig, ax
