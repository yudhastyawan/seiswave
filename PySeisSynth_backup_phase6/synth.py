"""
Seismogram Synthesis Module — Modal Summation + IFFT

Computes synthetic seismograms using modal summation of surface waves
in the frequency domain, then transforms to time domain via IFFT.

Equivalent to CPS spulse96.

Key formula for ZVF (vertical force, vertical component):
    For each mode n at frequency ω:
      v_n = sqrt(are_src * are_rec) / sqrt(wvno_src * r)
      d_n = uz_src                 (source excitation)
      w_n = d_n * v_n * uz_rec     (receiver response)
    
    where are = k / (2 ω U I0)  from sregn96 energy()
    
    phase = exp(i*(ω*tshift - k*r - π/4))
    
    ZVF(ω) = Σ_n  xmom * w_n * atten * phase * S(ω)
    
    where xmom = moment / sqrt(2π)

References:
    - Herrmann (1979), BSSA 69, 1-16
    - CPS spulse96.f subroutine excitr
"""

import numpy as np
from .earth_model import LayeredModel
from .propagator import find_dispersion
from .eigen import love_eigen, rayleigh_eigen


def source_pulse(pulse_type, dt, npts, ntau=1, alpha=1.0):
    """
    Generate a source time function.

    Parameters
    ----------
    pulse_type : str
        'parabolic', 'triangular', 'ohnaka', 'dirac'.
    dt : float
        Time step in seconds.
    npts : int
        Number of samples.
    ntau : int
        Duration parameter (for parabolic / triangular).
    alpha : float
        Shape parameter (for Ohnaka pulse).

    Returns
    -------
    src : np.ndarray
        Source time function array of length npts.
    """
    src = np.zeros(npts)

    if pulse_type == 'parabolic':
        tau = ntau * dt
        t1 = 0.01 * dt
        t2 = t1 + tau
        t3 = t2 + tau
        t4 = t3 + tau
        t5 = t4 + tau
        for i in range(npts):
            t = i * dt
            z = t - t1
            if t1 <= t < t2:
                src[i] = 0.5 * (z / tau) ** 2
            elif t2 <= t <= t3:
                src[i] = -0.5 * (z / tau) ** 2 + 2.0 * (z / tau) - 1.0
            elif t3 < t <= t4:
                src[i] = -0.5 * (z / tau) ** 2 + 2.0 * (z / tau) - 1.0
            elif t4 < t <= t5:
                src[i] = 0.5 * (z / tau) ** 2 - 4.0 * (z / tau) + 8.0
        src /= (2.0 * tau)

    elif pulse_type == 'triangular':
        tau = ntau * dt
        fac = 1.0 / tau
        t1 = tau
        t2 = 2.0 * tau
        for i in range(npts):
            t = i * dt
            if t <= t1:
                src[i] = t * fac
            elif t1 < t <= t2:
                src[i] = fac - (t - t1) * fac

    elif pulse_type == 'ohnaka':
        al2 = alpha * alpha
        for i in range(npts):
            t = i * dt
            arg = alpha * t
            if arg <= 25.0:
                src[i] = al2 * t * np.exp(-arg)

    elif pulse_type == 'dirac':
        src[0] = 1.0 / dt

    return src


def _next_pow2(n):
    """Return smallest power of 2 >= n."""
    p = 1
    while p < n:
        p *= 2
    return p


def compute_greens(model, offsets, dt, npts, nmodes=100,
                   wave_types=('rayleigh',), pulse_type='parabolic',
                   ntau=1, idva=1):
    """
    Compute synthetic seismograms via modal summation.

    Generates vertical-component (ZVF) Green's functions for a vertical
    point force source at the surface.

    Parameters
    ----------
    model : LayeredModel
        1D layered earth model.
    offsets : array_like
        Source-receiver distances in km.
    dt : float
        Sample interval in seconds.
    npts : int
        Number of time samples.
    nmodes : int
        Maximum number of modes.
    wave_types : tuple of str
        Which wave types to include: 'rayleigh', 'love', or both.
    pulse_type : str
        Source pulse type.
    ntau : int
        Pulse duration parameter.
    idva : int
        Output type: 0=displacement, 1=velocity, 2=acceleration.

    Returns
    -------
    data : np.ndarray, shape (npts, noffsets)
        Synthetic seismogram matrix (time × offset).
    """
    offsets = np.asarray(offsets, dtype=np.float64)
    noffsets = len(offsets)
    twopi = 2.0 * np.pi

    # Pad to next power of 2
    nfft = _next_pow2(npts)

    df = 1.0 / (nfft * dt)
    np2 = nfft // 2 + 1

    # Source spectrum
    src_time = source_pulse(pulse_type, dt, nfft, ntau=ntau)
    src_spec = np.fft.rfft(src_time)

    # Frequencies
    freqs = np.arange(np2) * df

    # Initialize spectral Green's functions for each offset
    # Shape: (np2, noffsets)
    gz_spec = np.zeros((np2, noffsets), dtype=np.complex128)

    print(f"PySeisSynth: Computing dispersion curves...")

    # CPS uses xmom = moment / sqrt(2π). For unit force, moment=1.
    xmom = 1.0 / np.sqrt(twopi)

    for wt in wave_types:
        wt = wt.lower()
        # Compute dispersion curves (positive frequencies only)
        freq_pos = freqs[1:]  # skip DC
        c_disp = find_dispersion(model, freq_pos, wave_type=wt, nmodes=nmodes)

        print(f"  {wt.capitalize()}: found modes at "
              f"{np.sum(~np.isnan(c_disp))} (freq, mode) points")

        # For each frequency, compute eigenfunction and modal sum
        for ifreq in range(len(freq_pos)):
            freq = freq_pos[ifreq]
            omega = twopi * freq
            ispec = ifreq + 1  # index into spectrum (offset by 1 for DC)

            for imode in range(nmodes):
                c_val = c_disp[ifreq, imode]
                if np.isnan(c_val) or c_val <= 0:
                    continue

                wvno = omega / c_val

                # Compute eigenfunctions
                try:
                    if wt == 'love':
                        # Love waves not excited by vertical force
                        continue
                    else:
                        eig = rayleigh_eigen(model, omega, c_val)
                except Exception as e:
                    import traceback
                    print(f"Exception at freq={freq:.2f}, mode={imode}: {e}")
                    # traceback.print_exc()
                    continue

                are = eig['are']
                ugr = eig['ugr']
                gamma = eig['gamma']
                I0 = eig['I0']
                wvno_c = eig.get('wvno_corrected', wvno)  # corrected for dispersion

                if I0 < 1.0e-30 or are < 1.0e-30:
                    continue
                    
                if gamma < 0:
                    print(f"WARNING: negative gamma at freq={freq:.3f}, mode={imode}, gamma={gamma:.3e}, c_val={c_val:.3f}")

                # Source excitation at surface (layer 0)
                uz_src = eig['uz'][0]
                # Receiver at surface
                uz_rec = eig['uz'][0]
                ur_rec = eig['ur'][0]

                # Source spectrum at this frequency
                S = src_spec[ispec]

                # For each offset
                for ioff in range(noffsets):
                    r = offsets[ioff]
                    if r <= 0:
                        continue

                    # CPS amplitude factor:
                    #   v = sqrt(are_src * are_rec) / sqrt(wvno_src * r)
                    # Since source and receiver are both at surface:
                    #   v = are / sqrt(wvno * r)
                    v_amp = np.sqrt(are * are) / np.sqrt(wvno * r)

                    # Source excitation for VF (vertical force):
                    #   d = uz_src
                    # Receiver response for ZVF (Z component):
                    #   w = d * v * uz_rec
                    w_zn = uz_src * v_amp * uz_rec

                    # Attenuation
                    atn = gamma * r
                    if atn < 80.0:
                        att = np.exp(-atn)
                    else:
                        att = 0.0

                    # Phase (CPS convention for ZVF):
                    #   t1 = omega*tshift - wvno*r - pi/4
                    t1 = -wvno * r - np.pi / 4.0
                    phase = np.complex128(np.cos(t1) + 1j * np.sin(t1))

                    # ZVF modal contribution (CPS sign: negative)
                    gz_spec[ispec, ioff] += (
                        -w_zn * xmom * att * phase * S
                    )

        print(f"  {wt.capitalize()} modal summation done.")

    # Apply velocity/acceleration conversion (multiply by (iω)^idva)
    if idva > 0:
        for i in range(np2):
            omega = twopi * i * df
            factor = (1j * omega) ** idva
            gz_spec[i, :] *= factor

    # IFFT to time domain
    print(f"PySeisSynth: Inverse FFT...")
    data = np.zeros((npts, noffsets))
    for ioff in range(noffsets):
        full_spec = gz_spec[:, ioff]
        trace = np.fft.irfft(full_spec, n=nfft)
        data[:, ioff] = np.real(trace[:npts])

    print(f"PySeisSynth: Done. Output shape = {data.shape}")
    return data
