"""
Earth Model Module — 1D Layered Earth Model

Defines the LayeredModel class and empirical velocity / density relations
for use with surface wave dispersion and seismogram synthesis.

References:
    - Gardner et al. (1974), Geophysics 39, 770-780  (density relation)
    - Poisson ratio relation for Vp from Vs
"""

import numpy as np


def compute_vp(vs: float, nu: float) -> float:
    """
    Compute P-wave velocity from S-wave velocity and Poisson's ratio.

    Vp = Vs * sqrt(2*(1-nu) / (1-2*nu))

    Parameters
    ----------
    vs : float
        S-wave velocity (any consistent unit, e.g. km/s).
    nu : float
        Poisson's ratio (0 < nu < 0.5).

    Returns
    -------
    float
        P-wave velocity in same unit as vs.
    """
    return vs * np.sqrt(2.0 * (1.0 - nu) / (1.0 - 2.0 * nu))


def compute_rho(vp: float) -> float:
    """
    Compute density from P-wave velocity using the Gardner relation.

    rho = 1.74 * Vp^0.25   (Vp in km/s → rho in g/cc)

    Parameters
    ----------
    vp : float
        P-wave velocity in km/s.

    Returns
    -------
    float
        Density in g/cm³.
    """
    return 1.74 * (vp ** 0.25)


class LayeredModel:
    """
    1D isotropic layered earth model.

    Attributes
    ----------
    nlayers : int
        Number of layers (including halfspace).
    h : np.ndarray
        Layer thicknesses in km. Last layer = 0 (halfspace).
    vp : np.ndarray
        P-wave velocities in km/s.
    vs : np.ndarray
        S-wave velocities in km/s.
    rho : np.ndarray
        Densities in g/cm³.
    qp : np.ndarray
        P-wave quality factors.
    qs : np.ndarray
        S-wave quality factors.
    """

    def __init__(self, h, vp, vs, rho, qp=None, qs=None):
        """
        Construct a LayeredModel directly from arrays.

        Parameters
        ----------
        h : array_like
            Thicknesses (km). Length = nlayers.
        vp : array_like
            P-wave velocities (km/s).
        vs : array_like
            S-wave velocities (km/s).
        rho : array_like
            Densities (g/cm³).
        qp : array_like, optional
            P quality factors (default 20.0).
        qs : array_like, optional
            S quality factors (default 20.0).
        """
        self.h = np.asarray(h, dtype=np.float64)
        self.vp = np.asarray(vp, dtype=np.float64)
        self.vs = np.asarray(vs, dtype=np.float64)
        self.rho = np.asarray(rho, dtype=np.float64)
        self.nlayers = len(self.h)

        if qp is None:
            self.qp = np.full(self.nlayers, 20.0, dtype=np.float64)
        else:
            self.qp = np.asarray(qp, dtype=np.float64)

        if qs is None:
            self.qs = np.full(self.nlayers, 20.0, dtype=np.float64)
        else:
            self.qs = np.asarray(qs, dtype=np.float64)

    @classmethod
    def from_h_vs(cls, layers, nu=0.40, qp_default=20.0, qs_default=20.0):
        """
        Construct a model from (thickness, Vs) pairs and a Poisson's ratio.

        Vp is computed via Poisson relation, density via Gardner relation.

        Parameters
        ----------
        layers : list of (float, float)
            Each element is (thickness_km, vs_km_s).
        nu : float
            Poisson's ratio. Default 0.40.
        qp_default : float
            Default P quality factor.
        qs_default : float
            Default S quality factor.

        Returns
        -------
        LayeredModel
        """
        h_arr = []
        vp_arr = []
        vs_arr = []
        rho_arr = []

        for thickness, vs in layers:
            vp = compute_vp(vs, nu)
            rho = compute_rho(vp)
            h_arr.append(thickness)
            vp_arr.append(vp)
            vs_arr.append(vs)
            rho_arr.append(rho)

        n = len(layers)
        return cls(
            h=h_arr,
            vp=vp_arr,
            vs=vs_arr,
            rho=rho_arr,
            qp=np.full(n, qp_default),
            qs=np.full(n, qs_default),
        )

    @property
    def depth_profile(self):
        """
        Return depth to top and bottom of each layer (in km).

        Returns
        -------
        depth_top : np.ndarray
        depth_bot : np.ndarray
        """
        depth_top = np.concatenate(([0.0], np.cumsum(self.h[:-1])))
        depth_bot = np.cumsum(self.h)
        return depth_top, depth_bot

    @property
    def cmin(self):
        """
        Estimate minimum phase velocity (Rayleigh wave in halfspace).

        Uses Newton iteration on the Rayleigh equation,
        mirroring CPS's `gtsolh` routine.
        """
        vs_min = np.min(self.vs[self.vs > 0])
        idx = np.argmin(self.vs[self.vs > 0])
        # find the layer whose vs == vs_min
        for i in range(self.nlayers):
            if self.vs[i] == vs_min:
                idx = i
                break
        return _gtsolh(self.vp[idx], self.vs[idx]) * 0.95

    @property
    def cmax(self):
        """Maximum phase velocity = max(Vs) across all layers."""
        return np.max(self.vs)

    def __repr__(self):
        return (
            f"LayeredModel(nlayers={self.nlayers}, "
            f"Vs_range=[{self.vs.min():.4f}, {self.vs.max():.4f}] km/s)"
        )


def _gtsolh(vp, vs):
    """
    Compute Rayleigh wave velocity in a halfspace by Newton iteration.

    Solves:  (2 - k²)² - 4 * sqrt(1 - γ²k²) * sqrt(1 - k²) = 0
    where k = c/Vs, γ = Vs/Vp.

    Equivalent to CPS subroutine `gtsolh`.

    Parameters
    ----------
    vp : float
        P-wave velocity.
    vs : float
        S-wave velocity.

    Returns
    -------
    float
        Rayleigh wave velocity.
    """
    c = 0.95 * vs
    for _ in range(5):
        gamma = vs / vp
        kappa = c / vs
        k2 = kappa ** 2
        gk2 = (gamma * kappa) ** 2
        fac1 = np.sqrt(abs(1.0 - gk2))
        fac2 = np.sqrt(abs(1.0 - k2))
        fr = (2.0 - k2) ** 2 - 4.0 * fac1 * fac2
        frp = (-4.0 * (2.0 - k2) * kappa
               + 4.0 * fac2 * gamma ** 2 * kappa / (fac1 + 1e-30)
               + 4.0 * fac1 * kappa / (fac2 + 1e-30))
        frp /= vs
        if abs(frp) > 1e-30:
            c = c - fr / frp
    return c
