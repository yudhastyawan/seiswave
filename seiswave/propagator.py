"""
Propagator Matrix Module — Surface Wave Dispersion

Implements the period equation for Love and Rayleigh waves using
Thomson-Haskell (SH/Love) and Dunkin compound matrix (P-SV/Rayleigh)
formalisms.  Root-finding via systematic bisection over wavenumber
at each frequency to obtain phase velocity dispersion curves.

References:
    - Haskell (1964), BSSA 54, 377-393
    - Dunkin (1965), BSSA 55, 335-358
    - Herrmann (1979), BSSA 69, 1-16
    - Aki & Richards (1980), Chapter 7
"""
import numpy as np

def apply_causal_q(v_ref, q, freq, f_ref=1.0):
    """
    Apply causal Q dispersion to the phase velocity.
    CPS formula: V(f) = V(f_ref) * [1 + (1 / (pi*Q)) * ln(f / f_ref)]
    """
    if q >= 10000.0 or freq == f_ref:
        return v_ref
    return v_ref * (1.0 + (1.0 / (np.pi * q)) * np.log(freq / f_ref))

try:
    from numba import njit
except ImportError:
    # Fallback: decorator that does nothing
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def wrapper(fn):
            return fn
        return wrapper


# ===================================================================
#  Love Wave Secular Function (SH — Thomson-Haskell 2×2)
# ===================================================================

@njit(cache=True)
def _dltar_love(wvno, omega, d, b, rho, nlayers):
    """
    Evaluate Love wave period equation.

    Propagates the (displacement, stress) vector from the halfspace
    upward to the free surface using 2×2 Haskell matrices.
    The secular function = stress at the free surface,
    which should be zero at a root.

    Parameters
    ----------
    wvno : float
        Wavenumber (rad/km).
    omega : float
        Angular frequency (rad/s).
    d, b, rho : 1-D arrays
        Thickness, S-velocity, density per layer.
    nlayers : int
        Number of layers.

    Returns
    -------
    float
        Value of the secular function (zero at eigenvalues).
    """
    wvno2 = wvno * wvno
    omega2 = omega * omega

    # ---- halfspace initial conditions ----
    xkb_hs = omega / b[nlayers - 1]
    rsh2 = wvno2 - xkb_hs * xkb_hs
    if rsh2 < 0.0:
        rsh = np.sqrt(-rsh2)
        # imaginary nu_b → start vector is complex, but for the
        # secular function we keep real part only
        mu_hs = rho[nlayers - 1] * b[nlayers - 1] * b[nlayers - 1]
        # E^{-1} half-space row 1  (only real parts matter for Love)
        e1 = mu_hs * rsh   # coefficient of σ contribution (imaginary branch)
        e2 = 1.0
        lshimag = True
    else:
        rsh = np.sqrt(rsh2)
        mu_hs = rho[nlayers - 1] * b[nlayers - 1] * b[nlayers - 1]
        e1 = mu_hs * rsh
        e2 = 1.0
        lshimag = False

    # ---- propagate from layer (mmax-1) down to 1 ----
    for m in range(nlayers - 2, -1, -1):
        if b[m] <= 0.0:
            continue  # skip fluid layers for Love

        xkb = omega / b[m]
        rb2 = wvno2 - xkb * xkb
        mu = rho[m] * b[m] * b[m]

        if rb2 < 0.0:
            # imaginary vertical wave number (propagating regime)
            rb = np.sqrt(-rb2)
            q = rb * d[m]
            cossh = np.cos(q)
            if rb > 1.0e-30:
                sinq = np.sin(q)
                sinshr = sinq / rb
                rsinsh = -rb * sinq
            else:
                cossh = 1.0
                sinshr = d[m]
                rsinsh = 0.0
        else:
            rb = np.sqrt(rb2)
            q = rb * d[m]
            if q < 16.0:
                fac = np.exp(-2.0 * q)
            else:
                fac = 0.0
            cossh = 0.5 * (1.0 + fac)
            sinq = 0.5 * (1.0 - fac)
            if rb > 1.0e-30:
                sinshr = sinq / rb
                rsinsh = sinq * rb
            else:
                sinshr = d[m]
                rsinsh = 0.0

        # Haskell 2×2 matrix H for SH
        # H = [[cosh, sinh/mu_rb], [mu*rb*sinh, cosh]]
        h11 = cossh
        h12 = sinshr / mu
        h21 = mu * rsinsh
        h22 = cossh

        # multiply: [e1, e2] = [e1, e2] * H
        e10 = e1 * h11 + e2 * h21
        e20 = e1 * h12 + e2 * h22

        # normalize to avoid overflow ONLY if huge
        xnor = max(abs(e10), abs(e20))
        if xnor > 1.0e250:
            e1 = e10 / xnor
            e2 = e20 / xnor
        else:
            e1 = e10
            e2 = e20

    return e1


# ===================================================================
#  Rayleigh Wave Secular Function (P-SV — Dunkin compound 5×5)
# ===================================================================

@njit(cache=True)
def _varsv(p_arg, q_arg, rp, rsv, wvno, d_m, omega, a_m, b_m):
    """
    Compute propagator variables for P-SV in a single layer.

    Returns (cosp, cossv, rsinp, rsinsv, sinpr, sinsvr, pex, svex).
    """
    # ---- P part ----
    pex = 0.0
    xka = omega / a_m
    rp2 = wvno * wvno - xka * xka
    if rp2 < 0.0:
        rp_val = np.sqrt(-rp2)
        pr = rp_val * d_m
        cosp = np.cos(pr)
        sinp = np.sin(pr)
        if rp_val > 1.0e-30:
            rsinp = -rp_val * sinp
            sinpr = sinp / rp_val
        else:
            rsinp = 0.0
            sinpr = d_m
    else:
        rp_val = np.sqrt(rp2)
        pr = rp_val * d_m
        if pr < 16.0:
            fac = np.exp(-2.0 * pr)
        else:
            fac = 0.0
        cosp = 0.5 * (1.0 + fac)
        sinq = 0.5 * (1.0 - fac)
        pex = pr
        if rp_val > 1.0e-30:
            rsinp = rp_val * sinq
            sinpr = sinq / rp_val
        else:
            rsinp = 0.0
            sinpr = d_m

    # ---- SV part ----
    svex = 0.0
    if b_m > 0.0:
        xkb = omega / b_m
        rsv2 = wvno * wvno - xkb * xkb
        if rsv2 < 0.0:
            rsv_val = np.sqrt(-rsv2)
            qr = rsv_val * d_m
            cossv = np.cos(qr)
            sinq2 = np.sin(qr)
            if rsv_val > 1.0e-30:
                rsinsv = -rsv_val * sinq2
                sinsvr = sinq2 / rsv_val
            else:
                rsinsv = 0.0
                sinsvr = d_m
        else:
            rsv_val = np.sqrt(rsv2)
            qr = rsv_val * d_m
            if qr < 16.0:
                fac2 = np.exp(-2.0 * qr)
            else:
                fac2 = 0.0
            cossv = 0.5 * (1.0 + fac2)
            sinq2 = 0.5 * (1.0 - fac2)
            svex = qr
            if rsv_val > 1.0e-30:
                rsinsv = rsv_val * sinq2
                sinsvr = sinq2 / rsv_val
            else:
                rsinsv = 0.0
                sinsvr = d_m
    else:
        # fluid
        cossv = 1.0
        rsinsv = 0.0
        sinsvr = 0.0

    return cosp, cossv, rsinp, rsinsv, sinpr, sinsvr, pex, svex


@njit(cache=True)
def _dnka(cosp, rsinp, sinpr, cossv, rsinsv, sinsvr,
          rho_m, b_m, wvno, wvno2, om2, pex, exa):
    """
    Build the 5×5 Dunkin compound layer matrix.
    Corresponds to CPS subroutine dnka for elastic layer.
    """
    ca = np.zeros((5, 5))

    if b_m <= 0.0:
        # fluid layer
        if pex > 35.0:
            dfac = 0.0
        else:
            dfac = np.exp(-pex)
        ca[2, 2] = dfac
        ca[0, 0] = cosp
        ca[4, 4] = cosp
        ca[0, 1] = -rsinp / (rho_m * om2)
        ca[1, 0] = -rho_m * sinpr * om2
        ca[1, 1] = cosp
        ca[3, 3] = cosp
        ca[3, 4] = ca[0, 1]
        ca[4, 3] = ca[1, 0]
        return ca

    # elastic layer
    if exa < 60.0:
        a0 = np.exp(-exa)
    else:
        a0 = 0.0

    cpcq = cosp * cossv
    cpy = cosp * sinsvr
    cpz = cosp * rsinsv
    cqw = cossv * sinpr
    cqx = cossv * rsinp
    xy = rsinp * sinsvr
    xz = rsinp * rsinsv
    wy = sinpr * sinsvr
    wz = sinpr * rsinsv

    gam = 2.0 * b_m * b_m * wvno2 / om2
    gam2 = gam * gam
    gamm1 = gam - 1.0
    gamm2 = gamm1 * gamm1
    gg1 = gam * gamm1

    a0c = 2.0 * (a0 - cpcq)
    xz2 = xz / wvno2
    gxz2 = gam * xz2
    g2xz2 = gam2 * xz2
    a0cgg1 = a0c * (gam + gamm1)
    wy2 = wy * wvno2
    g2wy2 = gamm2 * wy2
    g1wy2 = gamm1 * wy2

    temp = a0c * gg1 + g2xz2 + g2wy2
    ca[2, 2] = a0 + temp + temp
    ca[0, 0] = cpcq - temp
    ca[0, 1] = (-cqx + wvno2 * cpy) / (rho_m * om2)
    temp2 = 0.5 * a0cgg1 + gxz2 + g1wy2
    ca[0, 2] = wvno * temp2 / (rho_m * om2)
    ca[0, 3] = (-cqw * wvno2 + cpz) / (rho_m * om2)
    temp3 = wvno2 * (a0c + wy2) + xz
    ca[0, 4] = -temp3 / (rho_m * rho_m * om2 * om2)

    ca[1, 0] = (-gamm2 * cqw + gam2 * cpz / wvno2) * rho_m * om2
    ca[1, 1] = cpcq
    ca[1, 2] = (gamm1 * cqw * wvno2 - gam * cpz) / wvno
    ca[1, 3] = -wz
    ca[1, 4] = ca[0, 3]

    temp4 = 0.5 * a0cgg1 * gg1 + gam2 * gxz2 + gamm2 * g1wy2
    ca[2, 0] = -2.0 * temp4 * rho_m * om2 / wvno
    ca[2, 1] = -wvno * (gam * cqx / wvno2 - gamm1 * cpy) * 2.0
    ca[2, 3] = -2.0 * ca[1, 2]
    ca[2, 4] = -2.0 * ca[0, 2]

    ca[3, 0] = (-gam2 * cqx / wvno2 + gamm2 * cpy) * rho_m * om2
    ca[3, 1] = -xy
    ca[3, 2] = -ca[2, 1] / 2.0
    ca[3, 3] = ca[1, 1]
    ca[3, 4] = ca[0, 1]

    temp5 = gamm2 * (a0c * gam2 + g2wy2) + gam2 * g2xz2
    ca[4, 0] = -rho_m * rho_m * om2 * om2 * temp5 / wvno2
    ca[4, 1] = ca[3, 0]
    ca[4, 2] = -ca[2, 0] / 2.0
    ca[4, 3] = ca[1, 0]
    ca[4, 4] = ca[0, 0]

    return ca


@njit(cache=True)
def _evalg_halfspace(wvno, omega, a_m, b_m, rho_m):
    """
    Set up halfspace boundary conditions for Rayleigh.
    Returns gbr(5) — the initial compound vector.

    Matches CPS sregn96.f evalg subroutine for jbdry=0.
    """
    wvno2 = wvno * wvno
    om2 = omega * omega
    xka = omega / a_m
    ra2 = wvno2 - xka * xka

    if b_m > 0.0:
        xkb = omega / b_m
        rb2 = wvno2 - xkb * xkb
    else:
        rb2 = wvno2

    # absolute values of roots for exponential evaluation
    ra = np.sqrt(ra2) if ra2 >= 0.0 else np.sqrt(-ra2)
    rb = np.sqrt(rb2) if rb2 >= 0.0 else np.sqrt(-rb2)

    rab = ra * rb if (ra2 >= 0.0 and rb2 >= 0.0) else -ra * rb if (ra2 < 0.0 and rb2 < 0.0) else 0.0

    gam = 0.0
    gamm1 = -1.0
    if b_m > 0.0:
        gam = 2.0 * b_m * b_m * wvno2 / om2
        gamm1 = gam - 1.0

    gbr = np.zeros(5, dtype=np.complex128)

    if b_m > 0.0:
        # elastic halfspace (from sregn96.f evalg)
        ra = np.sqrt(ra2 + 0j)
        rb = np.sqrt(rb2 + 0j)
        
        gbr[0] = rho_m * rho_m * om2 * om2 * (-gam * gam * ra * rb + wvno2 * gamm1 * gamm1)
        gbr[1] = -rho_m * wvno2 * ra * om2
        gbr[2] = -rho_m * (-gam * ra * rb + wvno2 * gamm1) * om2 * wvno
        gbr[3] = rho_m * wvno2 * rb * om2 
        gbr[4] = wvno2 * (wvno2 - ra * rb)
        
        # apply normalization as in CPS 
        # norm = 0.25 / (-zrho(m)*zrho(m)*om2*om2*wvno2*ra*rb)
        norm = 0.25 / (-rho_m * rho_m * om2 * om2 * wvno2 * ra * rb - 1e-30)
        for i in range(5):
            gbr[i] *= norm
    else:
        # fluid halfspace (from sregn96.f evalg)
        ra = np.sqrt(ra2 + 0j)
        if (ra2 + 0j).real >= 0:
             gbr[0] = 0.5 / ra 
        else:
             gbr[0] = 0.0
        gbr[1] = 0.5 / (-rho_m * om2)
        gbr[2] = 0.0
        gbr[3] = 0.0
        gbr[4] = 0.0

    return gbr.real  # CRITICAL: sregn96 casts gbr to dreal BEFORE propagation!


@njit(cache=True)
def _dltar_rayleigh(wvno, omega, d, a, b, rho, nlayers):
    """
    Evaluate Rayleigh wave period equation using Dunkin compound matrix.

    Propagates the 5-element compound vector from halfspace upward
    through each layer.  Returns the first element (secular function).

    Parameters
    ----------
    wvno : float
        Wavenumber.
    omega : float
        Angular frequency.
    d, a, b, rho : arrays
        Model parameters.
    nlayers : int
        Number of layers.

    Returns
    -------
    float
        Secular function value (zero at eigenvalues).
    """
    wvno2 = wvno * wvno
    om2 = omega * omega

    # halfspace BCs
    e = _evalg_halfspace(wvno, omega,
                         a[nlayers - 1], b[nlayers - 1], rho[nlayers - 1])

    # propagate upward
    for m in range(nlayers - 2, -1, -1):
        cosp, cossv, rsinp, rsinsv, sinpr, sinsvr, pex, svex = \
            _varsv(0.0, 0.0, 0.0, 0.0, wvno, d[m], omega, a[m], b[m])

        ca = _dnka(cosp, rsinp, sinpr, cossv, rsinsv, sinsvr,
                   rho[m], b[m], wvno, wvno2, om2, pex, pex + svex)

        # e * ca
        ee = np.zeros(5)
        for i in range(5):
            s = 0.0
            for j in range(5):
                s += e[j] * ca[j, i]
            ee[i] = s

        # normalize to avoid overflow ONLY if huge
        xnor = 0.0
        for i in range(5):
            if abs(ee[i]) > xnor:
                xnor = abs(ee[i])

        if xnor > 1.0e250:
            for i in range(5):
                e[i] = ee[i] / xnor
        else:
            for i in range(5):
                e[i] = ee[i]

    return e[0]


# ===================================================================
#  Dispersion Root Finder
# ===================================================================

def find_dispersion(model, freqs, wave_type='rayleigh', nmodes=10):
    """
    Compute phase velocity dispersion curves for the given model.

    Parameters
    ----------
    model : LayeredModel
        The 1D earth model.
    freqs : array_like
        Frequencies in Hz (must be > 0).
    wave_type : str
        'love' or 'rayleigh'.
    nmodes : int
        Maximum number of modes to search.

    Returns
    -------
    c_all : np.ndarray, shape (nfreqs, nmodes)
        Phase velocities in km/s.  NaN where mode doesn't exist.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    nfreq = len(freqs)
    c_all = np.full((nfreq, nmodes), np.nan, dtype=np.float64)

    d = model.h.copy()
    a = model.vp.copy()
    b = model.vs.copy()
    rho = model.rho.copy()
    nl = model.nlayers
    
    # Store Q per layer
    qp = model.qp
    qs = model.qs

    twopi = 2.0 * np.pi

    for ifreq in range(nfreq):
        freq = freqs[ifreq]
        if freq <= 0:
            continue
        omega = twopi * freq

        # Apply causal Q dispersion to velocities for this frequency
        a_f = np.copy(model.vp)
        b_f = np.copy(model.vs)
        cmin_f = 1e9
        cmax_f = 0.0
        for i in range(nl):
            v_low = b_f[i] if b_f[i] > 0 else a_f[i]
            if v_low < cmin_f:
                cmin_f = v_low
            if v_low > cmax_f:
                cmax_f = v_low

        wvmx = omega / cmin_f
        wvmn = omega / (cmax_f + 0.1)

        # estimate number of search intervals
        sum_val = 0.0
        for i in range(nl):
            vel = b_f[i] if b_f[i] > 0 else a_f[i]
            fac = 1.0 / (vel * vel) - 1.0 / (cmax_f * cmax_f)
            if fac < 0:
                fac = 0
            sum_val += d[i] * np.sqrt(fac)
        sum_val = 2.0 * freq * sum_val
        nsearch = max(500, int(200 + 10 * sum_val))

        dk = (wvmx - wvmn) / nsearch

        nroot = 0
        is_love = wave_type.lower() == 'love'

        # scan for sign changes
        c2 = wvmx
        if is_love:
            del2 = _dltar_love(c2, omega, d, b_f, rho, nl)
        else:
            del2 = _dltar_rayleigh(c2, omega, d, a_f, b_f, rho, nl)

        for iscan in range(1, nsearch + 1):
            c1 = wvmx - iscan * dk
            if c1 < wvmn:
                c1 = wvmn + 0.01 * dk

            if is_love:
                del1 = _dltar_love(c1, omega, d, b_f, rho, nl)
            else:
                del1 = _dltar_rayleigh(c1, omega, d, a_f, b_f, rho, nl)

            if np.sign(del1) * np.sign(del2) < 0:
                # sign change → root between c1 and c2
                # bisect to refine
                ca = c1
                cb = c2
                da = del1
                db = del2
                for _ in range(50):
                    cc = 0.5 * (ca + cb)
                    if is_love:
                        dc = _dltar_love(cc, omega, d, b, rho, nl)
                    else:
                        dc = _dltar_rayleigh(cc, omega, d, a, b, rho, nl)

                    if np.sign(da) * np.sign(dc) >= 0:
                        ca = cc
                        da = dc
                    else:
                        cb = cc
                        db = dc
                    if abs(cb - ca) < 1.0e-12 * ca:
                        break

                wvno_root = 0.5 * (ca + cb)
                c_root = omega / wvno_root
                if nroot < nmodes:
                    c_all[ifreq, nroot] = c_root
                    nroot += 1

                if nroot >= nmodes:
                    break

            c2 = c1
            del2 = del1

    return c_all
