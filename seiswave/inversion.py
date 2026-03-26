import numpy as np
from scipy.optimize import differential_evolution

def compute_dependent_params(Vs_array):
    """
    Menghitung parameter dependen (Vp, Densitas, Qs, Qp) berdasarkan profil Vs
    menggunakan relasi empiris Brocher (2005) dan pedoman standar atenuasi.
    
    Semua kalkulasi dilakukan dengan operasi matriks/vektorisasi numpy demi efisiensi.
    
    Parameters:
    -----------
    Vs_array : numpy.ndarray (1D)
        Array Kecepatan Gelombang Geser (Vs) dalam m/s.
        
    Returns:
    --------
    Vp : numpy.ndarray
        Kecepatan Gelombang Primer (m/s).
    rho : numpy.ndarray
        Densitas (kg/m^3).
    Qs : numpy.ndarray
        Faktor Kualitas/Atenuasi Gelombang S.
    Qp : numpy.ndarray
        Faktor Kualitas/Atenuasi Gelombang P.
    """
    # Pastikan input adalah numpy array untuk operasi vektorisasi
    Vs = np.asarray(Vs_array, dtype=np.float64)
    
    # 1. Konversi Vs dari m/s ke km/s untuk input polynomial Brocher (2005)
    Vs_km = Vs / 1000.0
    
    # 2. Hitung Vp dalam km/s (Relasi Brocher 2005)
    Vp_km = (0.9409 + 
             2.0947 * Vs_km - 
             0.8206 * (Vs_km**2) + 
             0.2683 * (Vs_km**3) - 
             0.0251 * (Vs_km**4))
    
    # Konversi kembali Vp ke m/s
    Vp = Vp_km * 1000.0
    
    # 3. Hitung Densitas (rho) dalam g/cm^3 lalu konversi ke kg/m^3 (Relasi Brocher 2005)
    rho_gcm3 = (1.6612 * Vp_km - 
                0.4721 * (Vp_km**2) + 
                0.0671 * (Vp_km**3) - 
                0.0043 * (Vp_km**4) + 
                0.000106 * (Vp_km**5))
    rho = rho_gcm3 * 1000.0
    
    # 4. Perkiraan Faktor Kualitas/Atenuasi empiris (secara umum Qs ~ Vs/10)
    Qs = Vs / 10.0
    Qp = 2.0 * Qs
    
    return Vp, rho, Qs, Qp


from seiswave import LayeredModel, compute_greens, calculate_dispersion_image

def generate_synthetic_spectrum(H, Vp, Vs, rho, Qp, Qs, forward_params):
    """
    Menghasilkan matriks spektrum f-c sintetik menggunakan seiswave.
    1. Membuat LayeredModel.
    2. Menghitung profil seismogram sintetik (Green's functions).
    3. Mentransformasikan ruang wavefield t-x ke domain dispersi f-c dengan Phase-Shift.
    
    Parameters:
    -----------
    forward_params : dict
        Berisi parameter akuisisi dan kalkulasi:
        - 'offsets': array jarak offset (km)
        - 'dt': sampling rate waktu (s)
        - 'npts': jumlah titik waktu
        - 'c_min', 'c_max', 'dc': rentang resolusi kecepatan fasa (m/s)
        - 'f_min', 'f_max': rentang resolusi frekuensi (Hz)
        
    Output: 
    -------
    E_syn : numpy.ndarray (2D)
        Matriks spektrum energi f-c evaluasi.
    """
    # Cek pilihan backend engine
    engine = forward_params.get('engine', 'pyseissynth').lower()
    from seiswave.dispersion import calculate_dispersion_image
    
    if engine == 'cps':
        from seiswave.cps_runner import run_cps_forward
        
        # 1. Jalankan sintesis menggunakan shell Fortran CPS
        data = run_cps_forward(H, Vp, Vs, rho, Qp, Qs, forward_params)
        
        # 2. Transformasi f-c (Sama untuk kedua engine)
        x_m = forward_params['offsets'] * 1000.0
        fs = 1.0 / forward_params['dt']
        
        _, _, E_syn = calculate_dispersion_image(
            data, x_m, fs,
            c_min=forward_params['c_min'], 
            c_max=forward_params['c_max'], 
            dc=forward_params.get('dc', 1),
            f_min=forward_params['f_min'], 
            f_max=forward_params['f_max']
        )
        
        if forward_params.get('return_seismogram', False):
            return E_syn, data
        return E_syn
        
    else:
        # Default: seiswave (Native Python)
        
        # 1. Bangun seiswave Layered Model
        # H (ketebalan) kurang 1 dari parameter lain karena lapisan terbawah adalah halfspace (H=0.0)
        layers = []
        for i in range(len(Vs)):
            h_val = H[i] if i < len(H) else 0.0
            layers.append((h_val, Vs[i]/1000.0)) # LayeredModel.from_h_vs butuh Vs dalam km/s
            
        model = LayeredModel(
            h=np.append(H / 1000.0, 0.0), # Tambah 0.0 untuk halfspace, konversi H ke km
            vp=Vp/1000.0, 
            vs=Vs/1000.0, 
            rho=rho/1000.0, # g/cm3
            qp=Qp, 
            qs=Qs
        )
        
        # 2. Forward Modeling (Sintetik Seismogram)
        # Param defaults
        nmodes = forward_params.get('nmodes', 100)
        
        data = compute_greens(
            model, 
            forward_params['offsets'], 
            dt=forward_params['dt'], 
            npts=forward_params['npts'], 
            nmodes=nmodes,
            wave_types=('rayleigh',), 
            pulse_type='parabolic', 
            ntau=1, 
            idva=0
        )
        
        # 3. Transformasi ke spektrum f-c (Phase Shift Method)
        x_m = forward_params['offsets'] * 1000.0
        fs = 1.0 / forward_params['dt']
        
        _, _, E_syn = calculate_dispersion_image(
            data, 
            x_m, 
            fs,
            c_min=forward_params['c_min'], 
            c_max=forward_params['c_max'], 
            dc=forward_params.get('dc', 1),
            f_min=forward_params['f_min'], 
            f_max=forward_params['f_max']
        )
        
        if forward_params.get('return_seismogram', False):
            return E_syn, data
        return E_syn


def misfit_function(model_params, *args):
    """
    Fungsi Objektif L2-Norm untuk FWI spektrum f-c terskala per-frekuensi.
    
    Parameters:
    -----------
    model_params : numpy.ndarray (1D)
    args : tuple
        args[0] = E_obs (2D obs spectrum normalized)
        args[1] = forward_params (dict dictating synthetics resolution)
    """
    E_obs = args[0]
    forward_params = args[1]
    
    num_layers = (len(model_params) + 1) // 2
    H = model_params[:num_layers - 1]
    Vs = model_params[num_layers - 1:]
    
    Vp, rho, Qs, Qp = compute_dependent_params(Vs)
    
    # Sintetik Wavefield + Transformasi
    E_syn = generate_synthetic_spectrum(H, Vp, Vs, rho, Qp, Qs, forward_params)
    
    # Normalisasi f-c (dimensi f, c biasanya keluaran col, row. Tetapi calculate_dispersion_image 
    # mengembalikan letak frekuensi di axis tertentu. 
    # Diketahui di calculate_dispersion_image: E_norm dimensinya (len(c_arr), len(freqs)): 
    #   sumbu 0 = c (baris), sumbu 1 = f (kolom).
    # Agar matriks dinormalisasi untuk setiap lajur/kolom frekuensi f,
    # kita membagi dengan array max sepanjang axis kecepatan c (axis 0).
    max_c = np.max(E_syn, axis=0, keepdims=True)
    E_syn_norm = E_syn / (max_c + 1e-12)
    
    # Hitung Residual (Least Squares)
    misfit = np.sum((E_obs - E_syn_norm)**2)
    
    return misfit


def run_inversion(E_obs, num_layers, bounds_H, bounds_Vs, forward_params, **de_kwargs):
    """
    Menjalankan Differential Evolution optimisasi inversi penuh f-c.
    
    Parameters:
    E_obs : ndarray
    num_layers : int
    bounds_H, bounds_Vs : list of tuples
    forward_params : dict (parameter parameter pemodelan maju)
    de_kwargs : dict (argumen tambahan opsional untuk differential_evolution, mis: popsize, maxiter, callback)
    """
    assert len(bounds_H) == num_layers - 1
    assert len(bounds_Vs) == num_layers
    
    bounds = bounds_H + bounds_Vs
    
    popsize = de_kwargs.pop('popsize', 15)
    maxiter = de_kwargs.pop('maxiter', 1000)
    
    result = differential_evolution(
        misfit_function,
        bounds=bounds,
        args=(E_obs, forward_params),
        workers=-1,
        updating='deferred',
        strategy='best1bin',
        popsize=popsize,
        maxiter=maxiter,
        tol=0.01,
        disp=True,
        **de_kwargs
    )
    
    best_model = result.x
    best_H = best_model[:num_layers - 1]
    best_Vs = best_model[num_layers - 1:]
    
    return best_H, best_Vs, result.fun


# =============================================================================
# MCMC (Markov Chain Monte Carlo) Bayesian Inversion
# =============================================================================

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt


def log_likelihood(model_params, E_obs, forward_params, sigma=0.1):
    """
    Menghitung Gaussian log-likelihood antara spektrum observasi dan sintetik.
    
    ln L = -1/(2*sigma^2) * sum((E_obs - E_syn_norm)^2)
    
    Parameters
    ----------
    model_params : np.ndarray
        Vektor parameter model [H1, ..., Hn-1, Vs1, ..., Vsn].
    E_obs : np.ndarray
        Spektrum f-c observasi (ternormalisasi).
    forward_params : dict
        Parameter pemodelan maju.
    sigma : float
        Standar deviasi noise data.
        
    Returns
    -------
    float
        Nilai log-likelihood.
    """
    try:
        num_layers = (len(model_params) + 1) // 2
        H = model_params[:num_layers - 1]
        Vs = model_params[num_layers - 1:]
        
        Vp, rho, Qs, Qp = compute_dependent_params(Vs)
        E_syn = generate_synthetic_spectrum(H, Vp, Vs, rho, Qp, Qs, forward_params)
        
        max_c = np.max(E_syn, axis=0, keepdims=True)
        E_syn_norm = E_syn / (max_c + 1e-12)
        
        residual = np.sum((E_obs - E_syn_norm) ** 2)
        return -0.5 * residual / (sigma ** 2)
    except Exception:
        return -np.inf


def log_prior(model_params, bounds):
    """
    Uniform log-prior: 0 jika semua parameter dalam batas, -inf jika di luar.
    
    Parameters
    ----------
    model_params : np.ndarray
        Vektor parameter model.
    bounds : list of tuple
        Batas (min, max) untuk setiap parameter.
        
    Returns
    -------
    float
        0.0 atau -np.inf.
    """
    for val, (lo, hi) in zip(model_params, bounds):
        if val < lo or val > hi:
            return -np.inf
    return 0.0


def log_posterior(model_params, E_obs, forward_params, sigma, bounds):
    """
    Menghitung log-posterior = log-likelihood + log-prior.
    
    Parameters
    ----------
    model_params : np.ndarray
        Vektor parameter model.
    E_obs, forward_params, sigma, bounds : (lihat fungsi di atas)
    
    Returns
    -------
    float
        Nilai log-posterior.
    """
    lp = log_prior(model_params, bounds)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(model_params, E_obs, forward_params, sigma)
    return lp + ll


def proposal_function(current, step_sizes, rng):
    """
    Gaussian random walk proposal.
    
    m' = m + N(0, step_sizes)
    
    Parameters
    ----------
    current : np.ndarray
        Posisi parameter saat ini.
    step_sizes : np.ndarray
        Standar deviasi langkah per parameter.
    rng : np.random.Generator
        Generator random number.
        
    Returns
    -------
    np.ndarray
        Model kandidat baru.
    """
    return current + rng.normal(0, step_sizes)


def metropolis_hastings(E_obs, forward_params, bounds, sigma=0.1,
                        n_samples=5000, step_sizes=None, adaptive=True,
                        initial_model=None, rng=None, verbose=True,
                        chain_id=0, callback=None):
    """
    Single-chain Metropolis-Hastings MCMC sampler dengan adaptive step sizing.
    
    Parameters
    ----------
    E_obs : np.ndarray
        Spektrum f-c observasi (ternormalisasi).
    forward_params : dict
        Parameter pemodelan maju.
    bounds : list of tuple
        Batas (min, max) untuk setiap parameter.
    sigma : float
        Standar deviasi noise data.
    n_samples : int
        Jumlah total sampel yang dihasilkan.
    step_sizes : np.ndarray or None
        Ukuran langkah awal. Jika None, dihitung otomatis (5% dari rentang).
    adaptive : bool
        Aktifkan adaptasi ukuran langkah selama burn-in.
    initial_model : np.ndarray or None
        Model awal. Jika None, diambil secara random dari prior.
    rng : np.random.Generator or None
        Random number generator.
    verbose : bool
        Cetak progress.
    chain_id : int
        Identifikasi chain (untuk logging).
        
    Returns
    -------
    samples : np.ndarray (n_samples, n_params)
        Seluruh rantai sampel.
    log_posteriors : np.ndarray (n_samples,)
        Log-posterior untuk setiap sampel.
    acceptance_rate : float
        Tingkat penerimaan keseluruhan.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n_params = len(bounds)
    bounds_arr = np.array(bounds)
    
    # Inisialisasi step sizes (5% dari rentang jika tidak diberikan)
    if step_sizes is None:
        ranges = bounds_arr[:, 1] - bounds_arr[:, 0]
        step_sizes = ranges * 0.05
    else:
        step_sizes = np.array(step_sizes, dtype=np.float64)
    
    # Model awal: random dari prior jika tidak diberikan
    if initial_model is None:
        current = bounds_arr[:, 0] + rng.random(n_params) * (bounds_arr[:, 1] - bounds_arr[:, 0])
    else:
        current = np.array(initial_model, dtype=np.float64)
    
    current_lp = log_posterior(current, E_obs, forward_params, sigma, bounds)
    
    # Alokasi penyimpanan
    samples = np.zeros((n_samples, n_params))
    log_posts = np.zeros(n_samples)
    n_accepted = 0
    
    # Adaptive tracking
    adapt_window = 100
    window_accepted = 0
    
    for i in range(n_samples):
        # Proposal
        candidate = proposal_function(current, step_sizes, rng)
        candidate_lp = log_posterior(candidate, E_obs, forward_params, sigma, bounds)
        
        # Metropolis acceptance ratio
        log_alpha = candidate_lp - current_lp
        
        if np.log(rng.random()) < log_alpha:
            current = candidate
            current_lp = candidate_lp
            n_accepted += 1
            window_accepted += 1
        
        samples[i] = current
        log_posts[i] = current_lp
        
        # Adaptive step sizing setiap adapt_window sampel
        if adaptive and (i + 1) % adapt_window == 0:
            window_rate = window_accepted / adapt_window
            if window_rate > 0.44:
                step_sizes *= 1.1
            elif window_rate < 0.23:
                step_sizes *= 0.9
            window_accepted = 0
            
            if verbose and (i + 1) % (adapt_window * 5) == 0:
                total_rate = n_accepted / (i + 1)
                print(f"  Chain {chain_id}: sample {i+1}/{n_samples}, "
                      f"acceptance={total_rate:.2%}, "
                      f"log_post={current_lp:.4f}")
        
        if callback is not None:
            callback(chain_id, i, n_samples, current, current_lp)
    
    acceptance_rate = n_accepted / n_samples
    return samples, log_posts, acceptance_rate


@dataclass
class MCMCResult:
    """
    Kontainer hasil inversi MCMC yang menyediakan akses ke rantai sampel,
    statistik konvergensi, dan metode ringkasan.
    
    Attributes
    ----------
    chains : np.ndarray
        Array 3D (n_chains, n_kept, n_params) berisi sampel pasca burn-in & thinning.
    acceptance_rates : list
        Tingkat penerimaan per chain.
    log_posteriors : np.ndarray
        Array 3D (n_chains, n_kept) log-posterior per sampel.
    param_names : list
        Nama parameter ['H1', 'Vs1', 'Vs2', ...].
    bounds : list
        Batas pencarian parameter.
    best_model : np.ndarray
        Estimasi MAP (Maximum A Posteriori).
    best_H : np.ndarray
        Ketebalan lapisan dari model MAP.
    best_Vs : np.ndarray
        Kecepatan geser dari model MAP.
    gelman_rubin_R : np.ndarray
        Statistik R-hat konvergensi per parameter.
    all_samples : np.ndarray
        Gabungan semua chain (n_total, n_params) untuk analisis.
    """
    chains: np.ndarray
    acceptance_rates: list
    log_posteriors: np.ndarray
    param_names: list
    bounds: list
    best_model: np.ndarray
    best_H: np.ndarray
    best_Vs: np.ndarray
    gelman_rubin_R: np.ndarray
    all_samples: np.ndarray = field(init=False)
    
    def __post_init__(self):
        # Gabungkan semua chain menjadi satu array untuk analisis
        self.all_samples = self.chains.reshape(-1, self.chains.shape[-1])
    
    def summary(self):
        """Cetak tabel ringkasan statistik MCMC."""
        header = f"{'Parameter':<10} {'Mean':>10} {'Std':>10} {'2.5%':>10} {'50%':>10} {'97.5%':>10} {'R-hat':>8}"
        print("=" * len(header))
        print("MCMC Inversion Summary")
        print("=" * len(header))
        print(header)
        print("-" * len(header))
        
        for i, name in enumerate(self.param_names):
            vals = self.all_samples[:, i]
            mean = np.mean(vals)
            std = np.std(vals)
            q025 = np.percentile(vals, 2.5)
            q50 = np.percentile(vals, 50.0)
            q975 = np.percentile(vals, 97.5)
            rhat = self.gelman_rubin_R[i] if i < len(self.gelman_rubin_R) else np.nan
            print(f"{name:<10} {mean:>10.2f} {std:>10.2f} {q025:>10.2f} {q50:>10.2f} {q975:>10.2f} {rhat:>8.4f}")
        
        print("-" * len(header))
        for j, rate in enumerate(self.acceptance_rates):
            print(f"Chain {j} acceptance rate: {rate:.2%}")
        print("=" * len(header))
    
    def percentiles(self, q):
        """
        Hitung persentil untuk setiap parameter.
        
        Parameters
        ----------
        q : float or array-like
            Persentil yang diinginkan (0-100).
        
        Returns
        -------
        np.ndarray
            Nilai persentil per parameter.
        """
        return np.percentile(self.all_samples, q, axis=0)
    
    def credible_interval(self, alpha=0.05):
        """
        Hitung interval kredibel simetris.
        
        Parameters
        ----------
        alpha : float
            Level signifikansi (default 0.05 untuk 95% CI).
        
        Returns
        -------
        lower : np.ndarray
        upper : np.ndarray
        """
        lower = np.percentile(self.all_samples, 100 * alpha / 2, axis=0)
        upper = np.percentile(self.all_samples, 100 * (1 - alpha / 2), axis=0)
        return lower, upper


def gelman_rubin(chains):
    """
    Menghitung statistik konvergensi Gelman-Rubin R-hat.
    
    R-hat mendekati 1.0 menunjukkan konvergensi yang baik.
    Secara umum R-hat < 1.1 dianggap konvergen.
    
    Parameters
    ----------
    chains : np.ndarray
        Array 3D (n_chains, n_samples, n_params).
    
    Returns
    -------
    R_hat : np.ndarray
        Statistik R-hat per parameter.
    """
    n_chains, n_samples, n_params = chains.shape
    R_hat = np.zeros(n_params)
    
    for p in range(n_params):
        # Mean per chain
        chain_means = np.mean(chains[:, :, p], axis=1)  # (n_chains,)
        # Variance per chain
        chain_vars = np.var(chains[:, :, p], axis=1, ddof=1)  # (n_chains,)
        
        # Between-chain variance
        overall_mean = np.mean(chain_means)
        B = n_samples * np.var(chain_means, ddof=1)
        
        # Within-chain variance
        W = np.mean(chain_vars)
        
        # Pooled variance estimate
        var_hat = ((n_samples - 1) / n_samples) * W + (1 / n_samples) * B
        
        # R-hat
        R_hat[p] = np.sqrt(var_hat / W) if W > 0 else np.inf
    
    return R_hat


def run_mcmc_inversion(E_obs, num_layers, bounds_H, bounds_Vs, forward_params,
                       sigma=0.1, n_chains=4, n_samples=5000, burn_in=1000,
                       thin=5, step_sizes=None, adaptive=True, seed=None,
                       verbose=True, initial_models=None, callback=None):
    """
    Menjalankan inversi MCMC Bayesian multi-chain dengan Metropolis-Hastings.
    
    Fitur komprehensif:
    - Multiple independent chains untuk diagnosis konvergensi
    - Adaptive step sizing untuk acceptance rate optimal (23-44%)
    - Burn-in period removal
    - Thinning untuk mengurangi autokorelasi
    - Gelman-Rubin R-hat konvergensi diagnostik
    - MAP (Maximum A Posteriori) estimation
    
    Parameters
    ----------
    E_obs : np.ndarray
        Spektrum f-c observasi (ternormalisasi per frekuensi).
    num_layers : int
        Jumlah lapisan (termasuk halfspace).
    bounds_H : list of tuple
        Batas (min, max) untuk ketebalan setiap lapisan.
    bounds_Vs : list of tuple
        Batas (min, max) untuk Vs setiap lapisan.
    forward_params : dict
        Parameter pemodelan maju (offsets, dt, npts, c_min, c_max, dc, f_min, f_max, nmodes).
    sigma : float
        Standar deviasi noise data (mengontrol lebar likelihood).
    n_chains : int
        Jumlah rantai independen (minimum 2 untuk Gelman-Rubin).
    n_samples : int
        Total sampel per chain (termasuk burn-in).
    burn_in : int
        Jumlah sampel awal yang dibuang.
    thin : int
        Interval thinning (simpan setiap thin-th sampel).
    step_sizes : np.ndarray or None
        Ukuran langkah proposal awal. None = otomatis (5% dari rentang).
    adaptive : bool
        Aktifkan adaptasi ukuran langkah.
    seed : int or None
        Random seed untuk reproduksibilitas.
    verbose : bool
        Cetak progress per chain.
    initial_models : list of np.ndarray or None
        Model awal untuk setiap chain. None = random dari prior.
        
    Returns
    -------
    MCMCResult
        Objek hasil berisi chains, statistik, dan metode analisis.
    """
    assert len(bounds_H) == num_layers - 1, "bounds_H harus berjumlah num_layers - 1"
    assert len(bounds_Vs) == num_layers, "bounds_Vs harus berjumlah num_layers"
    assert n_samples > burn_in, "n_samples harus lebih besar dari burn_in"
    assert n_chains >= 2, "Minimum 2 chains untuk diagnostik Gelman-Rubin"
    
    bounds = bounds_H + bounds_Vs
    
    # Nama parameter
    param_names = []
    for i in range(num_layers - 1):
        param_names.append(f"H{i+1}")
    for i in range(num_layers):
        param_names.append(f"Vs{i+1}")
    
    # Seed management
    base_rng = np.random.default_rng(seed)
    chain_seeds = [base_rng.integers(0, 2**31) for _ in range(n_chains)]
    
    if verbose:
        print(f"MCMC Inversion: {n_chains} chains × {n_samples} samples")
        print(f"  Burn-in: {burn_in}, Thinning: {thin}")
        print(f"  Parameters: {param_names}")
        print(f"  Sigma: {sigma}")
        print()
    
    all_chains_raw = []
    all_logpost_raw = []
    all_acceptance = []
    
    for c in range(n_chains):
        if verbose:
            print(f"--- Running Chain {c} ---")
        
        rng_c = np.random.default_rng(chain_seeds[c])
        init_model = initial_models[c] if initial_models is not None else None
        
        samples, log_posts, acc_rate = metropolis_hastings(
            E_obs, forward_params, bounds, sigma=sigma,
            n_samples=n_samples, step_sizes=step_sizes,
            adaptive=adaptive, initial_model=init_model,
            rng=rng_c, verbose=verbose, chain_id=c, callback=callback
        )
        
        all_chains_raw.append(samples)
        all_logpost_raw.append(log_posts)
        all_acceptance.append(acc_rate)
        
        if verbose:
            print(f"  Chain {c} complete. Acceptance rate: {acc_rate:.2%}\n")
    
    # Burn-in removal + thinning
    kept_chains = []
    kept_logposts = []
    for c in range(n_chains):
        post_burnin = all_chains_raw[c][burn_in:]
        post_burnin_lp = all_logpost_raw[c][burn_in:]
        thinned = post_burnin[::thin]
        thinned_lp = post_burnin_lp[::thin]
        kept_chains.append(thinned)
        kept_logposts.append(thinned_lp)
    
    chains_arr = np.array(kept_chains)       # (n_chains, n_kept, n_params)
    logpost_arr = np.array(kept_logposts)     # (n_chains, n_kept)
    
    # Gelman-Rubin R-hat
    R_hat = gelman_rubin(chains_arr)
    
    if verbose:
        print("Gelman-Rubin R-hat:")
        for i, name in enumerate(param_names):
            status = "✓" if R_hat[i] < 1.1 else "✗ (not converged)"
            print(f"  {name}: {R_hat[i]:.4f} {status}")
        print()
    
    # MAP estimate (model dengan log-posterior tertinggi dari semua chain)
    all_samples_flat = chains_arr.reshape(-1, chains_arr.shape[-1])
    all_logpost_flat = logpost_arr.reshape(-1)
    best_idx = np.argmax(all_logpost_flat)
    best_model = all_samples_flat[best_idx]
    
    best_H = best_model[:num_layers - 1]
    best_Vs = best_model[num_layers - 1:]
    
    result = MCMCResult(
        chains=chains_arr,
        acceptance_rates=all_acceptance,
        log_posteriors=logpost_arr,
        param_names=param_names,
        bounds=bounds,
        best_model=best_model,
        best_H=best_H,
        best_Vs=best_Vs,
        gelman_rubin_R=R_hat
    )
    
    if verbose:
        result.summary()
    
    return result


def plot_mcmc_results(result, true_model=None, save_path=None):
    """
    Visualisasi komprehensif 4-panel hasil MCMC.
    
    Panel 1: Trace plots — mixing dan konvergensi per parameter
    Panel 2: Posterior histograms — distribusi marginal + credible intervals
    Panel 3: Vs profile ensemble — profil 1D dengan median + 95% CI shading
    Panel 4: Corner plot — korelasi 2D antar parameter
    
    Parameters
    ----------
    result : MCMCResult
        Objek hasil dari run_mcmc_inversion.
    true_model : dict or None
        Dict dengan kunci 'H' dan 'Vs' (np.ndarray) untuk overlay model sebenarnya.
    save_path : str or None
        Path untuk menyimpan gambar. None = tampilkan saja.
    """
    n_params = len(result.param_names)
    n_chains = result.chains.shape[0]
    
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    
    # ========== Panel 1: Trace Plots ==========
    ax_traces = fig.add_subplot(gs[0, 0])
    colors_chain = plt.cm.Set2(np.linspace(0, 1, n_chains))
    
    for p in range(n_params):
        for c in range(n_chains):
            chain_data = result.chains[c, :, p]
            # Normalisasi ke [0,1] untuk overlay
            lo, hi = result.bounds[p]
            normalized = (chain_data - lo) / (hi - lo + 1e-12)
            
            # Gunakan tab10 untuk kontras tinggi, linewidth lebih tebal agar terlihat
            color_idx = c % 10
            lbl = f"Chain {c+1}" if p == 0 else "" # Hindari duplikasi di legenda
            ax_traces.plot(normalized + p, alpha=0.8, color=plt.cm.tab10(color_idx), 
                          linewidth=1.0, label=lbl)
        ax_traces.axhline(y=p, color='gray', linewidth=1.0, linestyle='--')
        
    ax_traces.legend(loc='upper right', fontsize=9, title="MCMC Chains")
    
    ax_traces.set_yticks(range(n_params))
    ax_traces.set_yticklabels(result.param_names)
    ax_traces.set_xlabel('Sample (post burn-in & thinned)')
    ax_traces.set_title('Trace Plots (normalized)')
    ax_traces.set_xlim(0, result.chains.shape[1])
    
    # ========== Panel 2: Posterior Histograms ==========
    n_hist_rows = int(np.ceil(n_params / 2))
    gs_hist = gs[0, 1].subgridspec(n_hist_rows, 2, hspace=0.5, wspace=0.3)
    
    lower_ci, upper_ci = result.credible_interval(alpha=0.05)
    
    for p in range(n_params):
        row_p = p // 2
        col_p = p % 2
        ax_h = fig.add_subplot(gs_hist[row_p, col_p])
        
        vals = result.all_samples[:, p]
        ax_h.hist(vals, bins=40, density=True, alpha=0.7, color='steelblue', 
                  edgecolor='white', linewidth=0.5)
        
        # Credible interval lines
        ax_h.axvline(lower_ci[p], color='red', linestyle='--', linewidth=1.5, label='95% CI')
        ax_h.axvline(upper_ci[p], color='red', linestyle='--', linewidth=1.5)
        
        # Median
        median = np.median(vals)
        ax_h.axvline(median, color='black', linewidth=1.5, label=f'Median={median:.1f}')
        
        # True value overlay
        if true_model is not None:
            n_h = len(true_model['H'])
            if p < n_h:
                true_val = true_model['H'][p]
            else:
                true_val = true_model['Vs'][p - n_h]
            ax_h.axvline(true_val, color='green', linewidth=2, linestyle='-', 
                        label=f'True={true_val:.1f}')
        
        ax_h.set_title(result.param_names[p], fontsize=10)
        ax_h.legend(fontsize=6, loc='upper right')
        ax_h.tick_params(labelsize=7)
    
    # ========== Panel 3: Vs Profile Ensemble ==========
    ax_vs = fig.add_subplot(gs[1, 0])
    
    n_h_params = sum(1 for n in result.param_names if n.startswith('H'))
    n_vs_params = sum(1 for n in result.param_names if n.startswith('Vs'))
    
    # Plot individual model realizations (thin random subset)
    n_plot = min(200, len(result.all_samples))
    indices = np.random.choice(len(result.all_samples), n_plot, replace=False)
    
    max_depth = 0
    for idx in indices:
        sample = result.all_samples[idx]
        H_s = sample[:n_h_params]
        Vs_s = sample[n_h_params:]
        
        z = [0]
        v = [Vs_s[0]]
        for i, h in enumerate(H_s):
            z.append(z[-1] + h)
            v.append(Vs_s[i])
            z.append(z[-1])
            v.append(Vs_s[i + 1])
        z.append(z[-1] + 10.0)
        v.append(Vs_s[-1])
        max_depth = max(max_depth, z[-1])
        
        ax_vs.plot(v, z, color='steelblue', alpha=0.03, linewidth=0.5)
    
    # Median model
    median_params = np.median(result.all_samples, axis=0)
    H_med = median_params[:n_h_params]
    Vs_med = median_params[n_h_params:]
    z_med = [0]
    v_med = [Vs_med[0]]
    for i, h in enumerate(H_med):
        z_med.append(z_med[-1] + h)
        v_med.append(Vs_med[i])
        z_med.append(z_med[-1])
        v_med.append(Vs_med[i + 1])
    z_med.append(z_med[-1] + 10.0)
    v_med.append(Vs_med[-1])
    ax_vs.plot(v_med, z_med, color='darkblue', linewidth=2.5, label='Median Model')
    
    # MAP model
    z_map = [0]
    v_map = [result.best_Vs[0]]
    for i, h in enumerate(result.best_H):
        z_map.append(z_map[-1] + h)
        v_map.append(result.best_Vs[i])
        z_map.append(z_map[-1])
        v_map.append(result.best_Vs[i + 1])
    z_map.append(z_map[-1] + 10.0)
    v_map.append(result.best_Vs[-1])
    ax_vs.plot(v_map, z_map, color='red', linewidth=2, linestyle='--', label='MAP Model')
    
    # True model
    if true_model is not None:
        z_true = [0]
        v_true = [true_model['Vs'][0]]
        for i, h in enumerate(true_model['H']):
            z_true.append(z_true[-1] + h)
            v_true.append(true_model['Vs'][i])
            z_true.append(z_true[-1])
            v_true.append(true_model['Vs'][i + 1])
        z_true.append(z_true[-1] + 10.0)
        v_true.append(true_model['Vs'][-1])
        ax_vs.plot(v_true, z_true, color='lime', linewidth=2.5, linestyle='-', label='True Model')
    
    ax_vs.invert_yaxis()
    ax_vs.set_xlabel('Vs (m/s)')
    ax_vs.set_ylabel('Depth (m)')
    ax_vs.set_title('Vs Profile Ensemble (200 realizations)')
    ax_vs.legend(loc='lower left', fontsize=8)
    ax_vs.grid(True, alpha=0.3)
    
    # ========== Panel 4: Corner Plot (2D Correlations) ==========
    if n_params <= 6:
        gs_corner = gs[1, 1].subgridspec(n_params, n_params, hspace=0.05, wspace=0.05)
        
        for i in range(n_params):
            for j in range(n_params):
                ax_c = fig.add_subplot(gs_corner[i, j])
                
                if i == j:
                    # Diagonal: 1D histogram
                    ax_c.hist(result.all_samples[:, i], bins=30, density=True,
                             color='steelblue', alpha=0.7, edgecolor='white', linewidth=0.3)
                    if true_model is not None:
                        if i < n_h_params:
                            ax_c.axvline(true_model['H'][i], color='lime', linewidth=1.5)
                        else:
                            ax_c.axvline(true_model['Vs'][i - n_h_params], color='lime', linewidth=1.5)
                elif i > j:
                    # Lower triangle: 2D scatter
                    ax_c.scatter(result.all_samples[:, j], result.all_samples[:, i],
                                s=0.5, alpha=0.1, color='steelblue')
                    if true_model is not None:
                        tv_j = true_model['H'][j] if j < n_h_params else true_model['Vs'][j - n_h_params]
                        tv_i = true_model['H'][i] if i < n_h_params else true_model['Vs'][i - n_h_params]
                        ax_c.scatter([tv_j], [tv_i], color='lime', s=30, zorder=5, marker='+')
                else:
                    ax_c.axis('off')
                
                # Labels
                if j == 0 and i > 0:
                    ax_c.set_ylabel(result.param_names[i], fontsize=7)
                else:
                    ax_c.set_yticklabels([])
                
                if i == n_params - 1:
                    ax_c.set_xlabel(result.param_names[j], fontsize=7)
                else:
                    ax_c.set_xticklabels([])
                
                ax_c.tick_params(labelsize=5)
    else:
        # Too many parameters for corner plot, show correlation matrix instead
        ax_corr = fig.add_subplot(gs[1, 1])
        corr = np.corrcoef(result.all_samples.T)
        im = ax_corr.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax_corr.set_xticks(range(n_params))
        ax_corr.set_xticklabels(result.param_names, rotation=45, fontsize=8)
        ax_corr.set_yticks(range(n_params))
        ax_corr.set_yticklabels(result.param_names, fontsize=8)
        ax_corr.set_title('Parameter Correlation Matrix')
        plt.colorbar(im, ax=ax_corr, shrink=0.8)
    
    fig.suptitle('MCMC Bayesian Inversion Results', fontsize=14, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"MCMC results saved to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)
