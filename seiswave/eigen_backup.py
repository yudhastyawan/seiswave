"""
Eigenfunction Module — Displacement & Stress vs Depth

Computes eigenfunctions (displacement, stress) for each surface wave
mode, and the energy integrals needed for excitation and group velocity.

Equivalent to CPS slegn96 (Love) and sregn96 (Rayleigh).

The Rayleigh eigenfunction uses the two-pass method from CPS:
  - up():     Dunkin 5×5 compound matrix, bottom → top
  - down():   Haskell 4×4 propagator, top → bottom
  - svfunc(): combine both to form eigenfunctions at each layer

References:
    - Aki & Richards (1980), Chapter 7
    - Herrmann (1979), BSSA 69, 1-16
    - Dunkin (1965), BSSA 55, 335-358
"""

import numpy as np


# ===================================================================
#  Love Eigenfunction (unchanged from V1)
# ===================================================================

def love_eigen(model, omega, c):
    """
    Compute Love wave eigenfunctions for a given mode.
    Uses upward propagation from the halfspace through all layers
    with Haskell 2×2 SH matrices.
    """
    nl = model.nlayers
    d = model.h
    b = model.vs
    rho = model.rho
    wvno = omega / c
    wvno2 = wvno * wvno

    ut = np.zeros(nl)
    tt = np.zeros(nl)
    exl = np.zeros(nl)

    # halfspace BC
    mu_hs = rho[nl - 1] * b[nl - 1] ** 2
    xkb = omega / b[nl - 1]
    rb2 = wvno2 - xkb ** 2

    if rb2 >= 0:
        rb_hs = np.sqrt(rb2)
    else:
        rb_hs = np.sqrt(-rb2)

    ut[nl - 1] = 1.0
    if rb2 >= 0:
        tt[nl - 1] = -mu_hs * rb_hs
    else:
        tt[nl - 1] = 0.0
    exl[nl - 1] = 0.0

    for k in range(nl - 2, -1, -1):
        if b[k] <= 0:
            ut[k] = ut[k + 1]
            tt[k] = 0.0
            exl[k] = 0.0
            continue

        mu = rho[k] * b[k] ** 2
        xkb = omega / b[k]
        rb2 = wvno2 - xkb ** 2
        dpth = d[k]

        if rb2 < 0:
            rb = np.sqrt(-rb2)
            q = rb * dpth
            cossh = np.cos(q)
            if rb > 1.0e-30:
                sinq = np.sin(q)
                sinshr = sinq / rb
                rsinsh = -rb * sinq
            else:
                cossh = 1.0
                sinshr = dpth
                rsinsh = 0.0
            eexl = 0.0
        else:
            rb = np.sqrt(rb2)
            q = rb * dpth
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
                sinshr = dpth
                rsinsh = 0.0
            eexl = q

        a11 = cossh
        a12 = -sinshr / mu
        a21 = -mu * rsinsh
        a22 = cossh

        amp0 = a11 * ut[k + 1] + a12 * tt[k + 1]
        str0 = a21 * ut[k + 1] + a22 * tt[k + 1]

        rr = max(abs(amp0), abs(str0))
        if rr < 1.0e-80:
            rr = 1.0
        exl[k] = np.log(rr) + eexl
        ut[k] = amp0 / rr
        tt[k] = str0 / rr

    # normalize
    ext = 0.0
    tt[0] = 0.0
    for k in range(1, nl):
        if b[k] > 0:
            ext += exl[k - 1]
            fact = 0.0
            if ext < 80.0:
                fact = 1.0 / np.exp(ext)
            ut[k] = ut[k] * fact
            tt[k] = tt[k] * fact

    umax = ut[0] if abs(ut[0]) > 0 else 1.0
    for k in range(nl):
        ut[k] /= umax
        tt[k] /= umax

    # energy integrals
    I0 = 0.0
    I1 = 0.0
    I2 = 0.0

    for k in range(nl):
        if b[k] <= 0:
            continue
        mu = rho[k] * b[k] ** 2
        xkb = omega / b[k]
        rb2 = wvno2 - xkb ** 2

        if rb2 >= 0:
            rb = np.sqrt(rb2)
        else:
            rb = np.sqrt(-rb2)
        if rb < 1.0e-80:
            rb = 1.0e-80

        if k == nl - 1:
            upup = 0.5 * ut[k] ** 2 / rb
            dupdup = 0.5 * rb * ut[k] ** 2
        else:
            ut_avg2 = 0.5 * (ut[k] ** 2 + ut[min(k + 1, nl - 1)] ** 2)
            upup = ut_avg2 * d[k]
            if d[k] > 0:
                dut = (ut[min(k + 1, nl - 1)] - ut[k]) / d[k]
            else:
                dut = 0.0
            dupdup = dut ** 2 * d[k]

        I0 += rho[k] * upup
        I1 += mu * upup
        I2 += mu * dupdup

    # ================================================================
    # Compute ugr using numerical differentiation of the dispersion relation
    # U = -dF/dk / dF/domega
    # ================================================================
    eps = 1.0e-4 * omega
    from .propagator import _dltar_love
    
    df_domega = (_dltar_love(wvno, omega + eps, d, b, rho, nl) - 
                 _dltar_love(wvno, omega - eps, d, b, rho, nl)) / (2 * eps)
                 
    deps_k = 1.0e-4 * wvno
    df_dk = (_dltar_love(wvno + deps_k, omega, d, b, rho, nl) - 
             _dltar_love(wvno - deps_k, omega, d, b, rho, nl)) / (2 * deps_k)
             
    if abs(df_domega) > 1e-30:
        ugr_implicit = -df_dk / df_domega
    else:
        ugr_implicit = c
        
    ugr = ugr_implicit

    gamma = 0.0
    for k in range(nl):
        if b[k] > 0 and model.qs[k] > 0:
            mu = rho[k] * b[k] ** 2
            if k == nl - 1:
                upup = 0.5 * ut[k] ** 2 / rb
            else:
                upup = 0.5 * (ut[k] ** 2 + ut[min(k + 1, nl - 1)] ** 2) * d[k]
            gamma += omega * mu * upup / (2.0 * model.qs[k] * ugr * I0)

    return {
        'ut': ut, 'tt': tt,
        'I0': I0, 'I1': I1, 'I2': I2,
        'ugr': ugr, 'gamma': gamma,
    }


# ===================================================================
#  Rayleigh Eigenfunction — Two-pass CPS method
# ===================================================================

def _varsv_layer(omega, wvno, d_m, a_m, b_m):
    """
    Compute trig/hyperbolic variables for a single layer.
    Port of CPS varsv subroutine.

    Returns: (cosp, cossv, rsinp, rsinsv, sinpr, sinsvr, pex, svex)
    All values are SCALED by exp(-pex) and exp(-svex) respectively.
    """
    wvno2 = wvno * wvno

    # P-wave vertical wavenumber
    xka = omega / a_m
    ra2 = wvno2 - xka * xka

    if ra2 >= 0:
        # evanescent P
        ra_val = np.sqrt(ra2)
        pr = ra_val * d_m
        pex = pr
        if pr < 30.0:
            pfac = np.exp(-2.0 * pr)
        else:
            pfac = 0.0
        cosp = 0.5 * (1.0 + pfac)   # scaled by exp(-pex)
        sinp_half = 0.5 * (1.0 - pfac)
        if ra_val > 1.0e-80:
            rsinp = ra_val * sinp_half
            sinpr = sinp_half / ra_val
        else:
            rsinp = 0.0
            sinpr = d_m
    else:
        # propagating P
        ra_val = np.sqrt(-ra2)
        pr = ra_val * d_m
        pex = 0.0
        cosp = np.cos(pr)
        sinp_val = np.sin(pr)
        if ra_val > 1.0e-80:
            rsinp = -ra_val * sinp_val  # note sign for imaginary ra
            sinpr = sinp_val / ra_val
        else:
            rsinp = 0.0
            sinpr = d_m

    # SV-wave vertical wavenumber
    if b_m > 0:
        xkb = omega / b_m
        rb2 = wvno2 - xkb * xkb

        if rb2 >= 0:
            # evanescent SV
            rb_val = np.sqrt(rb2)
            qr = rb_val * d_m
            svex = qr
            if qr < 30.0:
                svfac = np.exp(-2.0 * qr)
            else:
                svfac = 0.0
            cossv = 0.5 * (1.0 + svfac)
            sinq_half = 0.5 * (1.0 - svfac)
            if rb_val > 1.0e-80:
                rsinsv = rb_val * sinq_half
                sinsvr = sinq_half / rb_val
            else:
                rsinsv = 0.0
                sinsvr = d_m
        else:
            # propagating SV
            rb_val = np.sqrt(-rb2)
            qr = rb_val * d_m
            svex = 0.0
            cossv = np.cos(qr)
            sinq_val = np.sin(qr)
            if rb_val > 1.0e-80:
                rsinsv = -rb_val * sinq_val
                sinsvr = sinq_val / rb_val
            else:
                rsinsv = 0.0
                sinsvr = d_m
    else:
        svex = 0.0
        cossv = 1.0
        rsinsv = 0.0
        sinsvr = 0.0

    return cosp, cossv, rsinp, rsinsv, sinpr, sinsvr, pex, svex


def _hska(cosp, rsinp, sinpr, tcossv, trsinsv, tsinsvr,
          rho_m, b_m, wvno, wvno2, om2, pex, svex):
    """
    Build Haskell 4×4 P-SV layer matrix.
    Port of CPS hska subroutine.

    Note: tcossv, trsinsv, tsinsvr are scaled by exp(-svex).
    We need to rescale them relative to pex.
    """
    AA = np.zeros((4, 4))

    if b_m <= 0:
        # fluid layer
        if pex > 35.0:
            dfac = 0.0
        else:
            dfac = np.exp(-pex)
        AA[0, 0] = dfac
        AA[3, 3] = dfac
        AA[1, 1] = cosp
        AA[2, 2] = cosp
        AA[1, 2] = -rsinp / (rho_m * om2)
        AA[2, 1] = -rho_m * om2 * sinpr
        return AA

    # elastic layer — adjust SV trig for relative scaling
    if (pex - svex) > 70.0:
        dfac = 0.0
    else:
        dfac = np.exp(svex - pex)
    cossv = dfac * tcossv
    rsinsv = dfac * trsinsv
    sinsvr = dfac * tsinsvr

    gam = 2.0 * b_m * b_m * wvno2 / om2
    gamm1 = gam - 1.0

    AA[0, 0] = cossv + gam * (cosp - cossv)
    AA[0, 1] = -wvno * gamm1 * sinpr + gam * rsinsv / wvno
    AA[0, 2] = -wvno * (cosp - cossv) / (rho_m * om2)
    AA[0, 3] = (wvno2 * sinpr - rsinsv) / (rho_m * om2)
    AA[1, 0] = gam * rsinp / wvno - wvno * gamm1 * sinsvr
    AA[1, 1] = cosp - gam * (cosp - cossv)
    AA[1, 2] = (-rsinp + wvno2 * sinsvr) / (rho_m * om2)
    AA[1, 3] = -AA[0, 2]
    AA[2, 0] = rho_m * om2 * gam * gamm1 * (cosp - cossv) / wvno
    AA[2, 1] = rho_m * om2 * (-gamm1 * gamm1 * sinpr +
                                gam * gam * rsinsv / wvno2)
    AA[2, 2] = AA[1, 1]
    AA[2, 3] = -AA[0, 1]
    AA[3, 0] = rho_m * om2 * (gam * gam * rsinp / wvno2 -
                                gamm1 * gamm1 * sinsvr)
    AA[3, 1] = -AA[2, 0]
    AA[3, 2] = -AA[1, 0]
    AA[3, 3] = AA[0, 0]

    return AA


def _normc(ee):
    """Normalize vector and return exponent."""
    xnor = np.max(np.abs(ee))
    if xnor < 1.0e-80:
        xnor = 1.0
    exn = np.log(xnor)
    ee[:] = ee / xnor
    return exn


def _dnka_eigen(cosp, rsinp, sinpr, cossv, rsinsv, sinsvr,
                rho_m, b_m, wvno, wvno2, om2, pex, exa):
    """Build Dunkin 5×5 compound matrix (same as propagator.py _dnka)."""
    ca = np.zeros((5, 5))

    if b_m <= 0:
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


def _evalg_hs(wvno, omega, a_m, b_m, rho_m):
    """
    Halfspace boundary conditions for Rayleigh (Dunkin compound vector).
    Exact port of CPS evalg for elastic halfspace (jbdry=0).
    
    Uses complex square root to handle both evanescent and propagating cases.
    """
    wvno2 = wvno * wvno
    om2 = omega * omega

    xka = omega / a_m
    ra2 = wvno2 - xka * xka

    if b_m > 0:
        xkb = omega / b_m
        rb2 = wvno2 - xkb * xkb
    else:
        rb2 = wvno2

    # Complex square roots (CPS uses CDSQRT)
    ra = np.sqrt(complex(ra2, 0.0))
    rb = np.sqrt(complex(rb2, 0.0))

    gam = 2.0 * b_m * b_m * wvno2 / om2 if b_m > 0 else 0.0
    gamm1 = gam - 1.0

    gbr = np.zeros(5)

    if b_m > 0:
        # Elastic halfspace — exact CPS formula from evalg lines 1972-1988
        # First compute unscaled values:
        g1 = rho_m * rho_m * om2 * om2 * (-gam * gam * ra * rb + wvno2 * gamm1 * gamm1)
        g2 = -rho_m * wvno2 * ra * om2
        g3 = -rho_m * (-gam * ra * rb + wvno2 * gamm1) * om2 * wvno
        g4 = rho_m * wvno2 * rb * om2
        g5 = wvno2 * (wvno2 - ra * rb)

        # Divide by normalization factor
        denom = -rho_m * rho_m * om2 * om2 * wvno2 * ra * rb
        if abs(denom) > 1e-80:
            fac = 0.25 / denom
        else:
            fac = 0.0

        gbr[0] = np.real(g1 * fac)
        gbr[1] = np.real(g2 * fac)
        gbr[2] = np.real(g3 * fac)
        gbr[3] = np.real(g4 * fac)
        gbr[4] = np.real(g5 * fac)
    else:
        gbr[0] = 0.0
        gbr[1] = np.real(-rho_m * ra) if ra2 >= 0 else np.real(rho_m * ra)
        gbr[2] = 0.0
        gbr[3] = 0.0
        gbr[4] = 0.0

    return gbr


def rayleigh_eigen(model, omega, c):
    """
    Compute Rayleigh wave eigenfunctions using two-pass CPS method.

    Pass 1 (up):   Dunkin 5×5 compound matrix propagated bottom → top
    Pass 2 (down): Haskell 4×4 propagator propagated top → bottom
    Combine:       B_i = (cd × vv) / (-R12_13) at each layer interface

    Parameters
    ----------
    model : LayeredModel
    omega : float   Angular frequency (rad/s)
    c : float       Phase velocity (km/s)

    Returns
    -------
    dict with: ur, uz, tr, tz, I0, ugr, are, gamma
    """
    nl = model.nlayers
    d = model.h
    a = model.vp
    b = model.vs
    rho = model.rho
    wvno = omega / c
    wvno2 = wvno * wvno
    om2 = omega * omega

    # ================================================================
    #  Pass 1: UP — Dunkin compound matrix bottom → top
    # ================================================================
    cd = np.zeros((nl, 5))
    exe = np.zeros(nl)

    # Halfspace BC
    gbr = _evalg_hs(wvno, omega, a[nl - 1], b[nl - 1], rho[nl - 1])
    cd[nl - 1, :] = gbr
    exe[nl - 1] = 0.0

    exsum_up = 0.0
    for m in range(nl - 2, -1, -1):
        cosp, cossv, rsinp, rsinsv, sinpr, sinsvr, pex, svex = \
            _varsv_layer(omega, wvno, d[m], a[m], b[m])

        ca = _dnka_eigen(cosp, rsinp, sinpr, cossv, rsinsv, sinsvr,
                         rho[m], b[m], wvno, wvno2, om2, pex, pex + svex)

        # multiply: ee = cd[m+1] @ ca
        ee = np.zeros(5)
        for i in range(5):
            s = 0.0
            for j in range(5):
                s += cd[m + 1, j] * ca[j, i]
            ee[i] = s

        exn = _normc(ee)
        exsum_up += pex + svex + exn
        exe[m] = exsum_up
        cd[m, :] = ee

    # Period equation check: cd[0, 0] should be ≈ 0

    # ================================================================
    #  Pass 2: DOWN — Haskell 4×4 top → bottom
    # ================================================================
    vv = np.zeros((nl, 4))
    exa = np.zeros(nl)

    # Top surface BC: first column of propagator = [1, 0, 0, 0]
    vv[0, :] = [1.0, 0.0, 0.0, 0.0]
    exa[0] = 0.0

    exsum_dn = 0.0
    for m in range(nl - 1):
        cosp, cossv, rsinp, rsinsv, sinpr, sinsvr, pex, svex = \
            _varsv_layer(omega, wvno, d[m], a[m], b[m])

        AA = _hska(cosp, rsinp, sinpr, cossv, rsinsv, sinsvr,
                   rho[m], b[m], wvno, wvno2, om2, pex, svex)

        # aa0 = AA @ vv[m]
        aa0 = np.zeros(4)
        for i in range(4):
            s = 0.0
            for j in range(4):
                s += AA[i, j] * vv[m, j]
            aa0[i] = s

        ex2 = _normc(aa0)
        exsum_dn += pex + ex2
        exa[m + 1] = exsum_dn
        vv[m + 1, :] = aa0

    # ================================================================
    #  Combine: eigenfunctions at each layer boundary
    # ================================================================
    f1213 = -cd[0, 1]  # = -X|12/13 at surface

    ur_arr = np.zeros(nl)
    uz_arr = np.zeros(nl)
    tz_arr = np.zeros(nl)
    tr_arr = np.zeros(nl)

    # Surface values
    if abs(f1213) > 1e-80:
        ur_arr[0] = cd[0, 2] / cd[0, 1]  # = -X|12/14 / X|12/13
    else:
        ur_arr[0] = 0.0
    uz_arr[0] = 1.0
    tz_arr[0] = 0.0
    tr_arr[0] = 0.0

    uu0_1 = ur_arr[0]  # ur at surface

    for i in range(1, nl):
        cd1 = cd[i, 0]
        cd2 = cd[i, 1]
        cd3 = cd[i, 2]
        cd4 = -cd[i, 2]   # X|12/23 = -X|12/14
        cd5 = cd[i, 3]
        cd6 = cd[i, 4]

        # Haskell vector at layer i (first column)
        tz1 = -vv[i, 3]
        tz2 = -vv[i, 2]
        tz3 = vv[i, 1]
        tz4 = vv[i, 0]

        # Combine: B = (cd × vv) / f1213
        uu1 = tz2 * cd6 - tz3 * cd5 + tz4 * cd4
        uu2 = -tz1 * cd6 + tz3 * cd3 - tz4 * cd2
        uu3 = tz1 * cd5 - tz2 * cd3 + tz4 * cd1
        uu4 = -tz1 * cd4 + tz2 * cd2 - tz3 * cd1

        ext = exa[i] + exe[i] - exe[0]
        if -80.0 < ext < 80.0:
            fact = np.exp(ext)
            ur_arr[i] = uu1 * fact / f1213
            uz_arr[i] = uu2 * fact / f1213
            tz_arr[i] = uu3 * fact / f1213
            tr_arr[i] = uu4 * fact / f1213
        else:
            ur_arr[i] = 0.0
            uz_arr[i] = 0.0
            tz_arr[i] = 0.0
            tr_arr[i] = 0.0

    # ================================================================
    #  Energy integrals — simplified but correct formulation
    #  Following CPS energy() subroutine structure
    # ================================================================
    sumi0 = 0.0  # ω² I0 = ρ (ur² + uz²)
    sumi1 = 0.0  # k² I1
    sumi2 = 0.0  # k I2
    sumi3 = 0.0  # I3

    for m in range(nl):
        mu_m = rho[m] * b[m] ** 2 if b[m] > 0 else 0.0
        lam_m = rho[m] * a[m] ** 2 - 2.0 * mu_m

        TA = rho[m] * a[m] ** 2
        TC = TA
        TF = lam_m
        TL = mu_m
        TN = mu_m

        # ODE coefficients for isotropic medium
        xl2m = lam_m + 2.0 * mu_m
        if xl2m > 1e-30:
            a12 = -wvno * TF / xl2m
            a14 = 1.0 / xl2m
        else:
            a12 = 0.0
            a14 = 0.0

        if mu_m > 1e-30:
            a21 = wvno
            a23 = -1.0 / mu_m
        else:
            a21 = wvno
            a23 = 0.0

        if m == nl - 1:
            # halfspace: proper P/SV potential decomposition
            # CPS intijr for typelyr=+1 (lower halfspace)
            xka_m = omega / a[m]
            ra2_m = wvno2 - xka_m ** 2
            ra_m = np.sqrt(complex(ra2_m, 0.0))
            if abs(ra_m) < 1e-30:
                ra_m = complex(1e-30, 0.0)

            if b[m] > 0:
                xkb_m = omega / b[m]
                rb2_m = wvno2 - xkb_m ** 2
                rb_m = np.sqrt(complex(rb2_m, 0.0))
                if abs(rb_m) < 1e-30:
                    rb_m = complex(1e-30, 0.0)
            else:
                rb_m = np.sqrt(complex(wvno2, 0.0))

            # Build E matrix and Einv for halfspace (from CPS evalg)
            gam_hs = 2.0 * b[m] ** 2 * wvno2 / om2 if b[m] > 0 else 0.0
            gamm1_hs = gam_hs - 1.0

            # Einv matrix rows 3,4 (downgoing potentials) — from CPS evalg
            # Einv(3,1) = 0.5*gam/wvno
            # Einv(3,2) = 0.5*gamm1/ra
            # Einv(3,3) = -0.5/(rho*om2)
            # Einv(3,4) = -0.5*wvno/(rho*om2*ra)
            # Einv(4,1) = 0.5*gamm1/rb
            # Einv(4,2) = 0.5*gam/wvno
            # Einv(4,3) = -0.5*wvno/(rho*om2*rb)
            # Einv(4,4) = -0.5/(rho*om2)
            rho_om2 = rho[m] * om2
            einv31 = 0.5 * gam_hs / wvno
            einv32 = 0.5 * gamm1_hs / ra_m
            einv33 = -0.5 / rho_om2
            einv34 = -0.5 * wvno / (rho_om2 * ra_m)
            einv41 = 0.5 * gamm1_hs / rb_m
            einv42 = 0.5 * gam_hs / wvno
            einv43 = -0.5 * wvno / (rho_om2 * rb_m)
            einv44 = -0.5 / rho_om2

            # Downgoing potential coefficients at top of halfspace
            km1pd = (einv31 * ur_arr[m] + einv32 * uz_arr[m] +
                     einv33 * tz_arr[m] + einv34 * tr_arr[m])
            km1sd = (einv41 * ur_arr[m] + einv42 * uz_arr[m] +
                     einv43 * tz_arr[m] + einv44 * tr_arr[m])

            # E matrix columns 3,4 (downgoing waves)
            # E(1,3) = wvno,   E(1,4) = -rb
            # E(2,3) = -ra,    E(2,4) = wvno
            # E(3,3) = rho*om2*gamm1,  E(3,4) = -rho*om2*gam*rb/wvno
            # E(4,3) = -rho*om2*gam*ra/wvno,  E(4,4) = rho*om2*gamm1

            # For intijr with i,j indices:
            # Row 1 of E is [ur] component, Row 2 is [uz] component
            e13 = wvno
            e14 = -rb_m
            e23 = -ra_m
            e24 = wvno
            e33 = rho_om2 * gamm1_hs
            e34 = -rho_om2 * gam_hs * rb_m / wvno
            e43 = -rho_om2 * gam_hs * ra_m / wvno
            e44 = rho_om2 * gamm1_hs

            # INT(i,j) for halfspace (typelyr=+1):
            # cintijr = e(i,3)*e(j,3)*km1pd^2/(2*ra)
            #         + (e(i,3)*e(j,4)+e(i,4)*e(j,3))*km1pd*km1sd/(ra+rb)
            #         + e(i,4)*e(j,4)*km1sd^2/(2*rb)
            def int_ij_hs(ei3, ei4, ej3, ej4):
                return np.real(
                    ei3 * ej3 * km1pd * km1pd / (2.0 * ra_m) +
                    (ei3 * ej4 + ei4 * ej3) * km1pd * km1sd / (ra_m + rb_m) +
                    ei4 * ej4 * km1sd * km1sd / (2.0 * rb_m)
                )

            # INT11 = integral of ur*ur = E(1,:)*E(1,:) terms
            INT11 = int_ij_hs(e13, e14, e13, e14)
            INT22 = int_ij_hs(e23, e24, e23, e24)
            INT13 = int_ij_hs(e13, e14, e33, e34)
            INT24 = int_ij_hs(e23, e24, e43, e44)
            INT33 = int_ij_hs(e33, e34, e33, e34)
            INT44 = int_ij_hs(e43, e44, e43, e44)

            URUR = INT11
            UZUZ = INT22

            # Derivatives from ODE coefficients
            DURDUR = a12**2 * INT22 + 2*a12*a14*INT24 + a14**2 * INT44
            DUZDUZ = a21**2 * INT11 + 2*a21*a23*INT13 + a23**2 * INT33
            URDUZ_e = a21 * INT11 + a23 * INT13
            UZDUR_e = a12 * INT22 + a14 * INT24
        else:
            # Internal layer: use trapezoidal rule
            h_m = d[m]
            if h_m < 1e-30:
                continue
            URUR = 0.5 * (ur_arr[m] ** 2 + ur_arr[m + 1] ** 2) * h_m
            UZUZ = 0.5 * (uz_arr[m] ** 2 + uz_arr[m + 1] ** 2) * h_m
            URDUZ = 0.5 * (ur_arr[m] * uz_arr[m] + ur_arr[m + 1] * uz_arr[m + 1]) * h_m

            # Compute derivatives at top and bottom from stress
            duzdz_top = (tz_arr[m] + wvno * TF * ur_arr[m]) / xl2m if xl2m > 1e-30 else 0.0
            duzdz_bot = (tz_arr[m + 1] + wvno * TF * ur_arr[m + 1]) / xl2m if xl2m > 1e-30 else 0.0

            if mu_m > 1e-30:
                durdz_top = -wvno * uz_arr[m] + tr_arr[m] / mu_m
                durdz_bot = -wvno * uz_arr[m + 1] + tr_arr[m + 1] / mu_m
            else:
                durdz_top = wvno * uz_arr[m]
                durdz_bot = wvno * uz_arr[m + 1]

            DURDUR = 0.5 * (durdz_top ** 2 + durdz_bot ** 2) * h_m
            DUZDUZ = 0.5 * (duzdz_top ** 2 + duzdz_bot ** 2) * h_m

        sumi0 += rho[m] * (URUR + UZUZ)
        sumi1 += TL * UZUZ + TA * URUR

        if m == nl - 1:
            # halfspace: use exact URDUZ and UZDUR from intijr decomposition
            sumi2 += TL * UZDUR_e - TF * URDUZ_e
        else:
            # Internal layer: use simplified ODE coefficients
            URDUZ_val = a21 * URUR  # wvno * URUR
            UZDUR_val = a12 * UZUZ  # -wvno*F/(λ+2μ) * UZUZ
            sumi2 += TL * UZDUR_val - TF * URDUZ_val
        sumi3 += TL * DURDUR + TC * DUZDUZ

    # Lagrangian should be ≈ 0
    flagr = om2 * sumi0 - wvno2 * sumi1 - 2.0 * wvno * sumi2 - sumi3

    # ================================================================
    # Compute ugr using numerical differentiation of the dispersion relation
    # Because trapezoidal energy integrals in thin layers cause large
    # errors for spatial derivatives (sumi1, sumi2, sumi3)
    # U = c / (1 - (omega/c) * dc_domega)
    # ================================================================
    eps = 1.0e-4 * omega  # Very small frequency shift
    from .propagator import _dltar_rayleigh
    
    df_domega = (_dltar_rayleigh(wvno, omega + eps, d, a, b, rho, nl) - 
                 _dltar_rayleigh(wvno, omega - eps, d, a, b, rho, nl)) / (2 * eps)
                 
    deps_k = 1.0e-4 * wvno
    df_dk = (_dltar_rayleigh(wvno + deps_k, omega, d, a, b, rho, nl) - 
             _dltar_rayleigh(wvno - deps_k, omega, d, a, b, rho, nl)) / (2 * deps_k)
             
    if abs(df_domega) > 1e-30:
        ugr_implicit = -df_dk / df_domega
    else:
        ugr_implicit = c
        
    ugr = ugr_implicit

    if sumi0 > 1e-30:
        are = wvno / (2.0 * omega * ugr * sumi0)
    else:
        are = 0.0

    # ================================================================
    # Attenuation coefficient gamma (CPS gammap formula)
    #   gamma = (k / 2c) * sum_i [ dcdb_i * Vs_i / Qs_i + dcda_i * Vp_i / Qp_i ]
    # AND velocity dispersion correction (Kramers-Kronig):
    #   dc = sum_i [ ln(omega/omgref_s) * dcdb_i*Vs_i/Qs_i / pi
    #              + ln(omega/omgref_p) * dcda_i*Vp_i/Qp_i / pi ]
    # where dcda, dcdb = partial derivatives of phase velocity w.r.t.
    # layer velocities, computed via implicit differentiation of F(omega, k) = 0.
    #   dc/dv_i = (omega / k^2) * (dF/dv_i) / (dF/dk)
    # df_dk is already computed above for group velocity.
    # ================================================================
    gamma = 0.0
    dc_dispersion = 0.0
    pi_val = np.pi
    has_atten = False
    for i in range(nl):
        qai = 1.0 / model.qp[i] if model.qp[i] > 0 else 0.0
        qbi = 1.0 / model.qs[i] if model.qs[i] > 0 else 0.0
        if qai != 0 or qbi != 0:
            has_atten = True
            break

    if has_atten and abs(df_dk) > 1e-30:
        for i in range(nl):
            qai = 1.0 / model.qp[i] if model.qp[i] > 0 else 0.0
            qbi = 1.0 / model.qs[i] if model.qs[i] > 0 else 0.0
            if qai == 0.0 and qbi == 0.0:
                continue

            # Partial w.r.t. Vs (beta) for layer i
            if qbi != 0.0 and b[i] > 0:
                eps_b = 1.0e-4 * b[i]
                b_up = b.copy(); b_up[i] += eps_b
                b_dn = b.copy(); b_dn[i] -= eps_b
                dF_dbi = (_dltar_rayleigh(wvno, omega, d, a, b_up, rho, nl)
                        - _dltar_rayleigh(wvno, omega, d, a, b_dn, rho, nl)) / (2.0 * eps_b)
                # Take absolute value: the partial of phase velocity wrt layer velocity
                # is strictly positive physically, but the secular function F can flip sign
                dcdb_i = (omega / (wvno * wvno)) * abs(dF_dbi / df_dk)
                x = dcdb_i * b[i] * qbi
                gamma += x
                # Kramers-Kronig velocity dispersion (CPS uses FREFS=1.0 Hz → omgref=2π)
                omgref = 2.0 * pi_val * 1.0  # reference freq = 1 Hz
                dc_dispersion += np.log(omega / omgref) * x / pi_val

            # Partial w.r.t. Vp (alpha) for layer i
            if qai != 0.0 and a[i] > 0:
                eps_a = 1.0e-4 * a[i]
                a_up = a.copy(); a_up[i] += eps_a
                a_dn = a.copy(); a_dn[i] -= eps_a
                dF_dai = (_dltar_rayleigh(wvno, omega, d, a_up, b, rho, nl)
                        - _dltar_rayleigh(wvno, omega, d, a_dn, b, rho, nl)) / (2.0 * eps_a)
                dcda_i = (omega / (wvno * wvno)) * abs(dF_dai / df_dk)
                x = dcda_i * a[i] * qai
                gamma += x
                omgref = 2.0 * pi_val * 1.0
                dc_dispersion += np.log(omega / omgref) * x / pi_val

        gamma = 0.5 * wvno * gamma / c

    # Corrected phase velocity and wavenumber (CPS applies this in gammap)
    if has_atten:
        c_corrected = c + dc_dispersion
    else:
        c_corrected = c
    wvno_corrected = omega / c_corrected if c_corrected > 0 else wvno

    return {
        'ur': ur_arr,
        'uz': uz_arr,
        'tr': tr_arr,
        'tz': tz_arr,
        'I0': sumi0,
        'ugr': ugr,
        'are': are,
        'gamma': gamma,
        'wvno_corrected': wvno_corrected,
    }

