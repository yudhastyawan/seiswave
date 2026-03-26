import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import time
import sys
import os
import json
import tempfile

try:
    import obspy
    OBSPY_AVAILABLE = True
except ImportError:
    OBSPY_AVAILABLE = False

# Menambahkan parent directory ke system path untuk akses modul seiswave
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from seiswave.inversion import (
    compute_dependent_params,
    generate_synthetic_spectrum,
    run_inversion,
    run_mcmc_inversion,
    plot_mcmc_results
)
from seiswave.cps_runner import check_cps_installed
from seiswave import LayeredModel, compute_greens

# =============================================================================
# Streamlit App Configuration
# =============================================================================
st.set_page_config(
    page_title="seiswave Web UI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Session State Initialization
# =============================================================================
if 'model_df' not in st.session_state:
    st.session_state.model_df = pd.DataFrame({
        'Layer': [1, 2],
        'Thickness (m)': [5.0, np.nan],  # NaN berarti halfspace
        'Vs (m/s)': [150.0, 350.0]
    })
if 'E_syn' not in st.session_state:
    st.session_state.E_syn = None
if 'forward_params' not in st.session_state:
    st.session_state.forward_params = {
        'offsets': np.arange(10, 41, 5) / 1000.0,
        'dt': 0.002,
        'npts': 256,
        'c_min': 100,
        'c_max': 500,
        'dc': 10,
        'f_min': 5.0,
        'f_max': 40.0,
        'nmodes': 2
    }
if 'E_obs' not in st.session_state:
    st.session_state.E_obs = None

# =============================================================================
# Sidebar Navigation
# =============================================================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png", width=60)
st.sidebar.title("seiswave")
st.sidebar.markdown("**Surface Wave Inversion Engine**")

page = st.sidebar.radio(
    "Navigasi Modul",
    ["Geological Model Builder", "Forward Modeling (f-c)", "Dispersion Inversion", "Real Field Data Processing", "CPS vs seiswave Benchmark"]
)

st.sidebar.markdown("---")
st.sidebar.info("Modul interaktif ini menggunakan **CPS (Computer Programs in Seismology)** sebagai engine default. Engine Native Python (seiswave) tersedia sebagai opsi *experimental*.")

# =============================================================================
# Modul 1: Geological Model Builder
# =============================================================================
if page == "Geological Model Builder":
    st.markdown('<div class="main-header">1D Earth Model Builder</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Bangun model bawah permukaan 1D dengan relasi empiris Brocher (2005)</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Parameter Input (H & Vs)")
        st.write("Edit ketebalan (Thickness) dan kecepatan geser (Vs) untuk setiap lapisan. Lapisan terbawah secara otomatis menjadi Halfspace (Thickness kosong).")
        
        # Num layers control
        num_layers = st.number_input("Jumlah Lapisan (termasuk halfspace)", min_value=2, max_value=10, value=len(st.session_state.model_df))
        
        # Adjust dataframe if num layers changes
        current_len = len(st.session_state.model_df)
        if num_layers > current_len:
            new_rows = pd.DataFrame({
                'Layer': range(current_len + 1, num_layers + 1),
                'Thickness (m)': [5.0] * (num_layers - current_len),
                'Vs (m/s)': [350.0] * (num_layers - current_len)
            })
            st.session_state.model_df = pd.concat([st.session_state.model_df, new_rows], ignore_index=True)
            # Pastikan hanya baris terakhir yang NaN
            st.session_state.model_df['Thickness (m)'] = st.session_state.model_df['Thickness (m)'].fillna(5.0)
            st.session_state.model_df.loc[num_layers - 1, 'Thickness (m)'] = np.nan
        elif num_layers < current_len:
            st.session_state.model_df = st.session_state.model_df.iloc[:num_layers]
            st.session_state.model_df.loc[num_layers - 1, 'Thickness (m)'] = np.nan
            
        edited_df = st.data_editor(
            st.session_state.model_df,
            num_rows="fixed",
            width="stretch",
            hide_index=True
        )
        st.session_state.model_df = edited_df
        
        # Compute dependents button
        if st.button("Hitung Parameter Dependen (Vp, Rho, Q)", type="primary"):
            Vs_array = edited_df['Vs (m/s)'].values
            Vp, rho, Qs, Qp = compute_dependent_params(Vs_array)
            
            result_df = edited_df.copy()
            result_df['Vp (m/s)'] = np.round(Vp, 2)
            result_df['Density (kg/m³)'] = np.round(rho, 2)
            result_df['Qs'] = np.round(Qs, 2)
            result_df['Qp'] = np.round(Qp, 2)
            
            st.session_state.full_model_df = result_df
            st.success("Parameter berhasil dihitung dengan persamaan Brocher (2005).")
            st.dataframe(result_df, hide_index=True, width="stretch")
            
            csv_model = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Unduh Profil Bumi 1D (CSV)",
                data=csv_model,
                file_name='earth_model_1d.csv',
                mime='text/csv',
            )
            
    with col2:
        st.subheader("Visualisasi Profil 1D")
        if 'full_model_df' in st.session_state:
            df = st.session_state.full_model_df
            H = df['Thickness (m)'].values[:-1]
            Vs = df['Vs (m/s)'].values
            Vp = df['Vp (m/s)'].values
            
            z = [0]
            v_s = [Vs[0]]
            v_p = [Vp[0]]
            
            for i, h in enumerate(H):
                z.append(z[-1] + h)
                v_s.append(Vs[i])
                v_p.append(Vp[i])
                
                z.append(z[-1])
                v_s.append(Vs[i+1])
                v_p.append(Vp[i+1])
                
            # Add extension for halfspace
            z.append(z[-1] + 15.0)
            v_s.append(Vs[-1])
            v_p.append(Vp[-1])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=v_s, y=z, mode='lines', name='Vs (m/s)', line=dict(color='blue', width=3)))
            fig.add_trace(go.Scatter(x=v_p, y=z, mode='lines', name='Vp (m/s)', line=dict(color='red', width=3)))
            fig.update_layout(
                title="Velocity Profile",
                xaxis_title="Velocity (m/s)",
                yaxis_title="Depth (m)",
                yaxis_autorange="reversed",
                template="plotly_white",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Klik tombol di sebelah kiri untuk menghasilkan plot.")

# =============================================================================
# Modul 2: Forward Modeling (f-c)
# =============================================================================
elif page == "Forward Modeling (f-c)":
    st.markdown('<div class="main-header">Forward Modeling (Seismogram & Dispersi)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Hasilkan seismogram sintetik dan citra dispersi f-c dari model 1D yang telah dibangun</div>', unsafe_allow_html=True)
    
    if 'full_model_df' not in st.session_state:
        st.warning("⚠️ Silakan bangun dan hitung model 1D di Tab 'Geological Model Builder' terlebih dahulu.")
    else:
        with st.expander("Parameter Akuisisi & Pemrosesan", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Parameter Geometri Offset**")
                min_off = st.number_input("Min Offset (m)", value=10.0, step=1.0)
                max_off = st.number_input("Max Offset (m)", value=40.0, step=1.0)
                dx = st.number_input("Spasi Geophone (m)", value=5.0, step=1.0)
            with col2:
                st.markdown("**Parameter Rekaman Waktu**")
                dt = st.number_input("Sampling Rate dt (s)", value=0.002, format="%.4f")
                npts = st.number_input("Jumlah Sampel (npts)", value=256, step=64)
                nmodes = st.number_input("Jumlah Mode (nmodes)", value=2, min_value=1, max_value=100)
            with col3:
                st.markdown("**Rentang Dispersi f-c**")
                c_min = st.number_input("Min V_phase (m/s)", value=100.0, step=10.0)
                c_max = st.number_input("Max V_phase (m/s)", value=500.0, step=10.0)
                dc = st.number_input("Resolusi dc (m/s)", value=10.0, step=1.0)
                f_min = st.number_input("Min Freq (Hz)", value=5.0, step=1.0)
                f_max = st.number_input("Max Freq (Hz)", value=40.0, step=1.0)
                
                has_cps = check_cps_installed()
                if has_cps:
                    engines = ["CPS (Computer Programs in Seismology)", "seiswave (Native Python) ⚠️ Experimental"]
                else:
                    engines = ["seiswave (Native Python) ⚠️ Experimental"]
                engine_choice = st.selectbox("Engine Forward", engines)
                engine_code = 'cps' if 'CPS' in engine_choice else 'pyseissynth'
                
        if st.button("Generate Synthetic Seismogram & f-c Spectrum", type="primary"):
            offsets = np.arange(min_off, max_off + dx, dx) / 1000.0 # to km
            st.session_state.forward_params = {
                'offsets': offsets,
                'dt': dt,
                'npts': int(npts),
                'c_min': c_min,
                'c_max': c_max,
                'dc': dc,
                'f_min': f_min,
                'f_max': f_max,
                'nmodes': int(nmodes),
                'engine': engine_code
            }
            
            df = st.session_state.full_model_df
            H = df['Thickness (m)'].values[:-1]
            Vs = df['Vs (m/s)'].values
            Vp = df['Vp (m/s)'].values
            rho = df['Density (kg/m³)'].values
            Qs = df['Qs'].values
            Qp = df['Qp'].values
            st.session_state.forward_params['return_seismogram'] = True
            with st.spinner('Menghitung Seismogram Sintetik dan Dispersi...'):
                E_syn, data = generate_synthetic_spectrum(
                    H, Vp, Vs, rho, Qp, Qs, st.session_state.forward_params
                )
                
                # Normalisasi
                max_c = np.max(E_syn, axis=0, keepdims=True)
                E_syn_norm = E_syn / (max_c + 1e-12)
                st.session_state.E_syn = E_syn_norm
                st.session_state.E_obs = E_syn_norm # Assign as observation directly for simplicity
                st.session_state.seismo_data = data
                
                st.success("Komputasi Selesai!")
                
        if st.session_state.E_syn is not None:
            # 1. Plot Seismogram
            st.markdown("### Synthetic Seismogram (Shot Gather)")
            import plotly.graph_objects as go
            fig_seis = go.Figure()
            
            data = st.session_state.seismo_data
            npts, n_offsets = data.shape
            dt_val = st.session_state.forward_params['dt']
            t_arr = np.arange(npts) * dt_val
            offsets = st.session_state.forward_params['offsets'] * 1000.0
            
            # Ensure data is real-valued (if IFFT returned complex type)
            data_real = np.real(data)
            
            # Trace Equalization: normalisasi setiap trace agar selalu terlihat
            dx = (offsets[1] - offsets[0]) if len(offsets) > 1 else 10.0
            
            for i in range(n_offsets):
                tr = data_real[:, i]
                tr_max = np.max(np.abs(tr))
                if tr_max > 0:
                    tr = tr / tr_max  # Normalize ke range [-1, 1]
                
                # Skala amplitudo agar maksimum 40% dari jarak antar geophone
                trace_display = (tr * dx * 0.4) + offsets[i]
                
                fig_seis.add_trace(go.Scatter(
                    x=trace_display, y=t_arr, mode='lines', 
                    line=dict(color='black', width=1),
                    name=f"Offset {offsets[i]:.0f}m"
                ))
            
            fig_seis.update_layout(
                xaxis_title="Offset (m) + Amplitude",
                yaxis_title="Time (s)",
                yaxis_autorange="reversed",
                showlegend=False,
                height=500,
                template="plotly_white"
            )
            st.plotly_chart(fig_seis, use_container_width=True)
            
            # 2. Plot Dispersion Image
            st.markdown("### Spektrum Dispersi f-c")
            c_arr = np.arange(c_min, c_max + dc, dc)
            f_arr = np.linspace(f_min, f_max, st.session_state.E_syn.shape[1])
            
            fig = px.imshow(
                st.session_state.E_syn,
                x=f_arr, 
                y=c_arr,
                color_continuous_scale='jet',
                origin='lower',
                labels=dict(x="Frequency (Hz)", y="Phase Velocity (m/s)", color="Amplitude"),
                aspect='auto'
            )
            fig.update_layout(title="Synthetic f-c Dispersion Spectrum")
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("Spektrum f-c sintetik ini telah disimpan sebagai Data Observasi (`E_obs`) dan siap diinversi di tab selanjutnya.")
            
            # Download forward params & data
            st.markdown("---")
            col_dl1, col_dl2, col_dl3 = st.columns(3)
            with col_dl1:
                fp_info = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in st.session_state.forward_params.items()}
                import json
                st.download_button("📥 Unduh Parameter Forward (JSON)", data=json.dumps(fp_info, indent=2), file_name='forward_params.json', mime='application/json', key='dl_fwd_params')
            with col_dl2:
                csv_seis = pd.DataFrame(np.real(st.session_state.seismo_data)).to_csv(index=False).encode('utf-8')
                st.download_button("📥 Unduh Seismogram (CSV)", data=csv_seis, file_name='synthetic_seismogram.csv', mime='text/csv', key='dl_fwd_seis')
            with col_dl3:
                csv_fc = pd.DataFrame(st.session_state.E_syn).to_csv(index=False).encode('utf-8')
                st.download_button("📥 Unduh Spektrum f-c (CSV)", data=csv_fc, file_name='synthetic_fc_spectrum.csv', mime='text/csv', key='dl_fwd_fc')

# =============================================================================
# Modul 3: Dispersion Inversion
# =============================================================================
elif page == "Dispersion Inversion":
    st.markdown('<div class="main-header">Dispersion Inversion Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Inversi Full Wavefield domain f-c menggunakan algoritma Global Optimization (DE / MCMC)</div>', unsafe_allow_html=True)
    
    if st.session_state.E_obs is None:
        st.warning("⚠️ Tidak ada data spektrum observasi yang dimuat. Silakan buat sintetik di Tab 'Forward Modeling' atau unggah matriks f-c (belum tersedia).")
    else:
        st.success("✅ Data spektrum observasi (f-c) termuat di memori.")
        
        # --- Parameter Sintetik untuk Inversi ---
        st.markdown("### Parameter Sintetik Forward Modeling")
        if 'forward_params' in st.session_state:
            fp = st.session_state.forward_params
            is_real_data = 'real_offsets' in st.session_state
            
            with st.expander("Parameter dt, npts, nmodes (untuk forward sintetik saat inversi)", expanded=True):
                col_p1, col_p2, col_p3 = st.columns(3)
                with col_p1:
                    if is_real_data:
                        st.info(f"🔒 **dt = {fp['dt']:.4f} s** (dari data riil)")
                        st.info(f"🔒 **npts = {fp['npts']}** (dari data riil)")
                    else:
                        st.info(f"dt = {fp['dt']:.4f} s | npts = {fp['npts']}")
                with col_p2:
                    nmodes_inv = st.number_input(
                        "Jumlah Mode (nmodes)",
                        value=int(fp.get('nmodes', 2)),
                        min_value=1, max_value=200,
                        help="Jumlah mode gelombang permukaan yang akan dihitung oleh engine forward"
                    )
                    fp['nmodes'] = int(nmodes_inv)
                with col_p3:
                    st.caption(f"Offsets: {len(fp['offsets'])} receiver")
                    st.caption(f"f: {fp['f_min']}-{fp['f_max']} Hz")
                    st.caption(f"c: {fp['c_min']}-{fp['c_max']} m/s")
        else:
            st.warning("⚠️ Forward params belum tersedia. Silakan jalankan Forward Modeling atau Real Field Data Processing terlebih dahulu.")
        
        st.markdown("---")
        algoritma = st.radio("Pilih Algoritma Inversi:", ["MCMC Bayesian - Komprehensif", "Differential Evolution (DE) - Cepat"])
        
        has_cps = check_cps_installed()
        if has_cps:
            engine_inv = st.radio("Pilih Engine Forward Inversi:", ["CPS (Computer Programs in Seismology)", "seiswave (Native Python) ⚠️ Experimental"], horizontal=True)
            engine_code = 'cps' if 'CPS' in engine_inv else 'pyseissynth'
        else:
            engine_code = 'pyseissynth'
            
        if 'forward_params' in st.session_state:
            st.session_state.forward_params['engine'] = engine_code
            
        st.markdown("### Konfigurasi Model & Batas Pencarian Parameter")
        
        num_layers = st.number_input("Jumlah Lapisan (termasuk Halfspace)", min_value=2, max_value=12, value=3)
        
        has_true_model = 'model_df' in st.session_state
        use_true_model = False
        if has_true_model:
            use_true_model = st.checkbox("Gunakan Model Pemandu dari Tab 'Model Builder' sebagai referensi", value=False)
        
        bounds_H = []
        bounds_Vs = []
        
        cols = st.columns(min(num_layers, 5))
        for i in range(num_layers):
            with cols[i % 5]:
                st.markdown(f"**Layer {i+1}**")
                if i < num_layers - 1:
                    if use_true_model and i < len(st.session_state.model_df) - 1:
                        true_h = float(st.session_state.model_df.loc[i, 'Thickness (m)'])
                        default_h = (max(1.0, true_h-5.0), true_h+5.0)
                    else:
                        default_h = (2.0, 30.0)
                    h_bound = st.slider(f"H{i+1} Bound (m)", 1.0, 100.0, default_h, key=f"h_{i}")
                    bounds_H.append(h_bound)
                else:
                    st.write("Halfspace (H = ∞)")
                    
                if use_true_model and i < len(st.session_state.model_df):
                    true_v = float(st.session_state.model_df.loc[i, 'Vs (m/s)'])
                    default_v = (max(100.0, true_v-100.0), true_v+150.0)
                else:
                    default_v = (100.0, 800.0)
                v_bound = st.slider(f"Vs{i+1} Bound (m/s)", 50.0, 2000.0, default_v, key=f"v_{i}")
                bounds_Vs.append(v_bound)
                
        st.markdown("---")
        show_live_plot = st.checkbox("Tampilkan Plot Intermediate (Dinamic Plotting - memperlambat proses)", value=False)
        if show_live_plot:
            if "MCMC" in algoritma:
                plot_interval = st.number_input("Update plot tiap N iterasi", min_value=1, max_value=500, value=20, step=10)
            else:
                plot_interval = st.number_input("Update plot tiap N iterasi", min_value=1, max_value=50, value=1, step=1)
        else:
            plot_interval = 20 if "MCMC" in algoritma else 1
        st.markdown("---")
        
        # Konfigurasi spesifik algoritma
        if "MCMC" in algoritma:
            st.markdown("### Parameter MCMC")
            col1, col2, col3, col4 = st.columns(4)
            n_chains = col1.number_input("N Chains", 2, 8, 3)
            n_samples = col2.number_input("N Samples", 100, 10000, 1000, step=100)
            burn_in = col3.number_input("Burn In", 0, 5000, 200, step=100)
            thin = col4.number_input("Thinning", 1, 20, 2)
            
            if st.button("🚀 Jalankan MCMC Inversion", type="primary"):
                st.markdown("💡 **Penjelasan Singkat Log-Posterior:**\n* **Log-Posterior (MCMC)** merupakan ukuran probabilitas kebenaran parameter. Nilainya didapatkan dari formulasi **Log-Likelihood** (selisih kuadrat *f-c* sintetik vs observasi berdasar distribusi *Gaussian Error*) ditambah bobot **Log-Prior**. Semakin kurvanya naik lalu mendatar (stabil) di nilai tertingginya, berarti inversi MCMC telah menemukan model yang **paling cocok (konvergen)** dengan observasi lapangan.*")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                plot_placeholder = st.empty()
                misfit_placeholder = st.empty()
                
                total_iters = int(n_chains * n_samples)
                lp_history = []
                mcmc_iter_models = []  # Simpan model tiap iterasi
                
                def mcmc_callback(chain_id, i, n_samples_chain, current, current_lp):
                    # Progress update
                    current_idx = chain_id * n_samples_chain + i
                    progress_pct = min(1.0, current_idx / total_iters)
                    progress_bar.progress(progress_pct)
                    status_text.text(f"MCMC berjalan: Chain {chain_id+1}/{int(n_chains)} | Sampel {i+1}/{n_samples_chain} | Log-Posterior: {current_lp:.4f}")
                    
                    # Update Log-Posterior curve (Misfit-equivalent)
                    lp_history.append(current_lp)
                    # Simpan model tiap iterasi
                    num_l = (len(current) + 1) // 2
                    cur_H_log = current[:num_l - 1].tolist()
                    cur_Vs_log = current[num_l - 1:].tolist()
                    mcmc_iter_models.append({'chain': chain_id, 'sample': i, 'lp': current_lp, 'misfit_l2': -2.0 * 0.01 * current_lp, 'H': cur_H_log, 'Vs': cur_Vs_log})
                    if current_idx % 10 == 0 or current_idx == total_iters - 1:
                        fig_lp = px.line(y=lp_history, x=range(1, len(lp_history)+1), labels={'x': 'Jumlah Sampel Evaluasi', 'y': 'Log-Posterior'})
                        fig_lp.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20))
                        misfit_placeholder.plotly_chart(fig_lp, use_container_width=True, key=f"lp_mcmc_{chain_id}_{i}")
                    
                    if show_live_plot and i > 0 and i % plot_interval == 0:
                        # Temporary plot for MAP mapping
                        fp = st.session_state.forward_params
                        fp['return_seismogram'] = False
                        num_l = (len(current) + 1) // 2
                        cur_H = current[:num_l - 1]
                        cur_Vs = current[num_l - 1:]
                        cur_Vp, cur_rho, cur_Qs, cur_Qp = compute_dependent_params(cur_Vs)
                        E_syn_cur = generate_synthetic_spectrum(cur_H, cur_Vp, cur_Vs, cur_rho, cur_Qp, cur_Qs, fp)
                        cur_norm = E_syn_cur / (np.max(E_syn_cur, axis=0, keepdims=True) + 1e-12)
                        
                        f_min, f_max = fp['f_min'], fp['f_max']
                        c_min, c_max = fp['c_min'], fp['c_max']
                        ext = [f_min, f_max, c_min, c_max]
                        
                        # Generate 1D Model Coordinates
                        d_plot = [0]
                        v_plot = [cur_Vs[0]]
                        cur_d = 0
                        for idx in range(len(cur_H)):
                            cur_d += cur_H[idx]
                            d_plot.extend([cur_d, cur_d])
                            val_next = cur_Vs[idx+1] if idx+1 < len(cur_Vs) else cur_Vs[-1]
                            v_plot.extend([cur_Vs[idx], val_next])
                        d_plot.append(cur_d + 10.0) # Halfspace
                        v_plot.append(cur_Vs[-1])
                        
                        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
                        ax[0].imshow(st.session_state.E_obs, aspect='auto', origin='lower', cmap='jet', extent=ext)
                        ax[0].set_title("Observasi f-c")
                        ax[0].set_xlabel("Frequency (Hz)")
                        ax[0].set_ylabel("Phase V (m/s)")
                        
                        ax[1].imshow(cur_norm, aspect='auto', origin='lower', cmap='jet', extent=ext)
                        ax[1].set_title(f"Sintetik (Chain {chain_id}, Iter {i})")
                        ax[1].set_xlabel("Frequency (Hz)")
                        
                        ax[2].plot(v_plot, d_plot, 'r-', linewidth=2)
                        ax[2].set_ylim(max(d_plot), 0)
                        ax[2].set_xlabel("Vs (m/s)")
                        ax[2].set_ylabel("Depth (m)")
                        ax[2].set_title("1D Vs Model Sementara")
                        
                        plot_placeholder.pyplot(fig)
                        plt.close(fig)

                with st.spinner("Menjalankan MCMC Chains secara Random Walk (ini butuh waktu)..."):
                    fp = st.session_state.forward_params
                    fp['return_seismogram'] = False # Ensure not expecting tuple in MCMC log_likelihood
                    # Run MCMC
                    result = run_mcmc_inversion(
                        E_obs=st.session_state.E_obs,
                        num_layers=num_layers,
                        bounds_H=bounds_H,
                        bounds_Vs=bounds_Vs,
                        forward_params=fp,
                        sigma=0.1,
                        n_chains=int(n_chains),
                        n_samples=int(n_samples),
                        burn_in=int(burn_in),
                        thin=int(thin),
                        adaptive=True,
                        seed=42,
                        verbose=False,
                        callback=mcmc_callback
                    )
                    
                    # Simpan ke session state
                    if use_true_model and 'model_df' in st.session_state:
                        df_true = st.session_state.model_df
                        true_model_plot = {
                            'H': df_true['Thickness (m)'].values[:-1],
                            'Vs': df_true['Vs (m/s)'].values
                        }
                    else:
                        true_model_plot = None
                    plot_mcmc_results(result, true_model=true_model_plot, save_path="temp_mcmc.png")
                    st.session_state.mcmc_result = result
                    st.session_state.mcmc_plot_path = "temp_mcmc.png"
                    st.session_state.mcmc_true_model = true_model_plot
                    st.session_state.mcmc_lp_history = lp_history
                    st.session_state.mcmc_iter_models = mcmc_iter_models
                    st.success("MCMC Inversi Selesai!")
        
        # Tampilkan hasil MCMC dari session_state (persist)
        if 'mcmc_result' in st.session_state and "MCMC" in algoritma:
            result = st.session_state.mcmc_result
            
            # Kurva Log-Posterior (persist)
            if st.session_state.get('mcmc_lp_history'):
                st.markdown("#### Kurva Log-Posterior (MCMC)")
                lp_data = st.session_state.mcmc_lp_history
                fig_lp = px.line(
                    y=lp_data,
                    x=list(range(1, len(lp_data) + 1)),
                    labels={'x': 'Jumlah Sampel Evaluasi', 'y': 'Log-Posterior'}
                )
                fig_lp.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_lp, use_container_width=True)
                csv_lp = pd.DataFrame({'Sample': range(1, len(lp_data)+1), 'LogPosterior': lp_data}).to_csv(index=False).encode('utf-8')
                st.download_button("📥 Unduh Kurva Log-Posterior (CSV)", data=csv_lp, file_name='mcmc_log_posterior.csv', mime='text/csv', key='dl_mcmc_lp')
            
            # Kurva Misfit L2 (derived from LP: misfit = -2 * sigma^2 * LP)
            if st.session_state.get('mcmc_lp_history'):
                st.markdown("#### Kurva Misfit L2 (MCMC)")
                lp_data = st.session_state.mcmc_lp_history
                sigma_mcmc = 0.1
                misfit_data = [-2.0 * sigma_mcmc**2 * lp for lp in lp_data]
                fig_misfit = px.line(
                    y=misfit_data,
                    x=list(range(1, len(misfit_data) + 1)),
                    labels={'x': 'Jumlah Sampel Evaluasi', 'y': 'Misfit L2'}
                )
                fig_misfit.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_misfit, use_container_width=True)
                csv_misfit_mcmc = pd.DataFrame({'Sample': range(1, len(misfit_data)+1), 'Misfit_L2': misfit_data}).to_csv(index=False).encode('utf-8')
                st.download_button("📥 Unduh Kurva Misfit L2 (CSV)", data=csv_misfit_mcmc, file_name='mcmc_misfit_l2.csv', mime='text/csv', key='dl_mcmc_misfit')
            
            st.markdown("#### Hasil Posterior dan Diagnostik MCMC")
            if os.path.exists(st.session_state.get('mcmc_plot_path', '')):
                st.image(st.session_state.mcmc_plot_path, width="stretch")
            
            col_d1, col_d2, col_d3 = st.columns(3)
            with col_d1:
                # Chain traces (per chain, per sample, all params + log-posterior)
                rows_chain = []
                for c in range(result.chains.shape[0]):
                    for s in range(result.chains.shape[1]):
                        row = {'Chain': c+1, 'Sample': s+1, 'LogPosterior': result.log_posteriors[c, s]}
                        for p, pname in enumerate(result.param_names):
                            row[pname] = result.chains[c, s, p]
                        rows_chain.append(row)
                csv_chains = pd.DataFrame(rows_chain).to_csv(index=False).encode('utf-8')
                st.download_button("📥 Chain Traces (CSV)", data=csv_chains, file_name='mcmc_chain_traces.csv', mime='text/csv', key='dl_mcmc_chains')
            with col_d2:
                # Convergence diagnostics: acceptance rates + R-hat
                diag_data = {'Parameter': result.param_names, 'R_hat': result.gelman_rubin_R.tolist()}
                diag_df = pd.DataFrame(diag_data)
                acc_row = pd.DataFrame([{'Parameter': f'AcceptRate_Chain{i+1}', 'R_hat': rate} for i, rate in enumerate(result.acceptance_rates)])
                diag_df = pd.concat([diag_df, acc_row], ignore_index=True)
                csv_diag = diag_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Konvergensi R-hat (CSV)", data=csv_diag, file_name='mcmc_convergence.csv', mime='text/csv', key='dl_mcmc_conv')
            with col_d3:
                if os.path.exists(st.session_state.get('mcmc_plot_path', '')):
                    with open(st.session_state.mcmc_plot_path, 'rb') as img_f:
                        st.download_button("📥 Gambar Diagnostik (PNG)", data=img_f.read(), file_name='mcmc_diagnostics.png', mime='image/png', key='dl_mcmc_diag_img')
            
            # f-c Spectrum Comparison: Observasi vs Median vs MAP
            st.markdown("#### Perbandingan Citra f-c: Observasi vs Median vs MAP")
            with st.spinner("Menghitung spektrum sintetik untuk Median dan MAP model..."):
                fp = st.session_state.forward_params.copy()
                fp['return_seismogram'] = False
                
                E_obs = st.session_state.E_obs
                f_min, f_max = fp['f_min'], fp['f_max']
                c_min, c_max = fp['c_min'], fp['c_max']
                
                # MAP model
                num_l = (len(result.best_model) + 1) // 2
                map_H = result.best_model[:num_l - 1]
                map_Vs = result.best_model[num_l - 1:]
                map_Vp, map_rho, map_Qs, map_Qp = compute_dependent_params(map_Vs)
                E_map = generate_synthetic_spectrum(map_H, map_Vp, map_Vs, map_rho, map_Qp, map_Qs, fp)
                E_map_norm = E_map / (np.max(E_map, axis=0, keepdims=True) + 1e-12)
                misfit_map = np.sum((E_obs - E_map_norm) ** 2)
                
                # Median model
                median_params = np.median(result.all_samples, axis=0)
                med_H = median_params[:num_l - 1]
                med_Vs = median_params[num_l - 1:]
                med_Vp, med_rho, med_Qs, med_Qp = compute_dependent_params(med_Vs)
                E_med = generate_synthetic_spectrum(med_H, med_Vp, med_Vs, med_rho, med_Qp, med_Qs, fp)
                E_med_norm = E_med / (np.max(E_med, axis=0, keepdims=True) + 1e-12)
                misfit_med = np.sum((E_obs - E_med_norm) ** 2)
                
                f_arr = np.linspace(f_min, f_max, E_obs.shape[1])
                c_arr = np.linspace(c_min, c_max, E_obs.shape[0])
                
                from plotly.subplots import make_subplots
                fig_fc = make_subplots(rows=1, cols=3, subplot_titles=[
                    "Observasi", 
                    f"Median Model (Misfit: {misfit_med:.4f})", 
                    f"MAP Model (Misfit: {misfit_map:.4f})"
                ])
                
                for idx, (E_data, name) in enumerate([(E_obs, "Obs"), (E_med_norm, "Median"), (E_map_norm, "MAP")]):
                    fig_fc.add_trace(go.Heatmap(
                        z=E_data, x=f_arr, y=c_arr,
                        colorscale='Jet', showscale=(idx == 2),
                        name=name
                    ), row=1, col=idx+1)
                
                fig_fc.update_layout(height=400, template="plotly_white")
                for i in range(1, 4):
                    fig_fc.update_xaxes(title_text="Frequency (Hz)", row=1, col=i)
                    fig_fc.update_yaxes(title_text="Phase Velocity (m/s)", row=1, col=i)
                st.plotly_chart(fig_fc, use_container_width=True)
                
                st.info(f"**Misfit L2** — MAP Model: `{misfit_map:.6f}` | Median Model: `{misfit_med:.6f}`")
            
            st.markdown("#### Statistik Interval Kredibel 95%")
            lower, upper = result.credible_interval()
            
            stats_data = []
            for k, param_name in enumerate(result.param_names):
                stats_data.append({
                    "Parameter": param_name,
                    "MAP Estimate": result.best_model[k],
                    "Mean": np.mean(result.all_samples[:, k]),
                    "95% CI Lower": lower[k],
                    "95% CI Upper": upper[k],
                    "R-hat (<1.1 = Conv)": result.gelman_rubin_R[k]
                })
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, width="stretch")
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                csv_mcmc = stats_df.to_csv(index=False).encode('utf-8')
                st.download_button(label="📥 Statistik (CSV)", data=csv_mcmc, file_name='inversion_mcmc_stats.csv', mime='text/csv', key='dl_mcmc_stats')
            with col_m2:
                samples_csv = pd.DataFrame(result.all_samples, columns=result.param_names).to_csv(index=False).encode('utf-8')
                st.download_button(label="📥 Semua Sampel (CSV)", data=samples_csv, file_name='mcmc_all_samples.csv', mime='text/csv', key='dl_mcmc_samples')
            with col_m3:
                # Unduh iterasi model + LP + Misfit L2
                if st.session_state.get('mcmc_iter_models'):
                    rows = []
                    for m in st.session_state.mcmc_iter_models:
                        row = {'Chain': m['chain'], 'Sample': m['sample'], 'LogPosterior': m['lp'], 'Misfit_L2': m.get('misfit_l2', -2.0 * 0.01 * m['lp'])}
                        for j, h in enumerate(m['H']):
                            row[f'H{j+1}'] = h
                        for j, v in enumerate(m['Vs']):
                            row[f'Vs{j+1}'] = v
                        rows.append(row)
                    csv_iter = pd.DataFrame(rows).to_csv(index=False).encode('utf-8')
                    st.download_button(label="📥 Iterasi+Misfit (CSV)", data=csv_iter, file_name='mcmc_iteration_history.csv', mime='text/csv', key='dl_mcmc_iter')
            with col_m4:
                # Best model
                best = result.best_model
                num_l = (len(best) + 1) // 2
                best_H = best[:num_l - 1]
                best_Vs = best[num_l - 1:]
                df_best = pd.DataFrame({
                    'Layer': range(1, len(best_Vs) + 1),
                    'Thickness (m)': list(best_H) + [np.nan],
                    'Vs (m/s)': best_Vs
                })
                csv_best = df_best.to_csv(index=False).encode('utf-8')
                st.download_button(label="📥 Model Terbaik (CSV)", data=csv_best, file_name='mcmc_best_model.csv', mime='text/csv', key='dl_mcmc_best')
                    
        if "DE" in algoritma:
            st.markdown("### Parameter Differential Evolution")
            col1, col2 = st.columns(2)
            maxiter = col1.number_input("Max Iterations", 10, 500, 50, step=10)
            popsize = col2.number_input("Population Size", 5, 50, 10, step=5)
            
            if st.button("🚀 Jalankan DE Inversion", type="primary"):
                st.markdown("💡 **Penjelasan Singkat Misfit:**\n* **Misfit L2 (DE)** didapatkan dengan menghitung **Selisih Kuadrat (*Sum of Squared Errors / L2-Norm*)** antara nilai amplitudo citra spektrum *f-c* observasi dengan spektrum sintetik yang dinormalisasi. Semakin kecil angkanya (mendekati 0), tebakan profil Vs *(Shear Velocity)* dari algoritma Differential Evolution menjadi semakin presisi/identik dengan bumi aslinya.*")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                plot_placeholder = st.empty()
                misfit_placeholder = st.empty()
                
                misfit_history = []
                iter_count = [0]
                iter_models = []  # Simpan model tiap iterasi
                
                def de_callback(xk, convergence=0):
                    iter_count[0] += 1
                    i = iter_count[0]
                    progress_pct = min(1.0, i / int(maxiter))
                    progress_bar.progress(progress_pct)
                    
                    # Compute current misfit for logging
                    fp = st.session_state.forward_params
                    fp['return_seismogram'] = False
                    num_l = (len(xk) + 1) // 2
                    cur_H = xk[:num_l - 1]
                    cur_Vs = xk[num_l - 1:]
                    cur_Vp, cur_rho, cur_Qs, cur_Qp = compute_dependent_params(cur_Vs)
                    E_syn_cur = generate_synthetic_spectrum(cur_H, cur_Vp, cur_Vs, cur_rho, cur_Qp, cur_Qs, fp)
                    cur_norm = E_syn_cur / (np.max(E_syn_cur, axis=0, keepdims=True) + 1e-12)
                    cur_misfit = np.sum((st.session_state.E_obs - cur_norm)**2)
                    
                    misfit_history.append(cur_misfit)
                    iter_models.append({'iter': i, 'misfit': cur_misfit, 'H': cur_H.tolist(), 'Vs': cur_Vs.tolist()})
                    status_text.text(f"Iterasi DE ke-{i}/{int(maxiter)} | Misfit L2: {cur_misfit:.4f} | Convergence: {convergence:.4f}")
                    
                    # Update Misfit Curve dynamically
                    fig_msf = px.line(y=misfit_history, x=range(1, len(misfit_history)+1), labels={'x': 'Iterasi', 'y': 'Misfit L2'})
                    fig_msf.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20))
                    misfit_placeholder.plotly_chart(fig_msf, use_container_width=True, key=f"msf_{i}")
                    
                    if show_live_plot and i % plot_interval == 0:
                        f_min, f_max = fp['f_min'], fp['f_max']
                        c_min, c_max = fp['c_min'], fp['c_max']
                        ext = [f_min, f_max, c_min, c_max]
                        
                        # Generate 1D Model Coordinates
                        d_plot = [0]
                        v_plot = [cur_Vs[0]]
                        cur_d = 0
                        for idx in range(len(cur_H)):
                            cur_d += cur_H[idx]
                            d_plot.extend([cur_d, cur_d])
                            val_next = cur_Vs[idx+1] if idx+1 < len(cur_Vs) else cur_Vs[-1]
                            v_plot.extend([cur_Vs[idx], val_next])
                        d_plot.append(cur_d + 10.0) # Halfspace
                        v_plot.append(cur_Vs[-1])
                        
                        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
                        ax[0].imshow(st.session_state.E_obs, aspect='auto', origin='lower', cmap='jet', extent=ext)
                        ax[0].set_title("Observasi f-c")
                        ax[0].set_xlabel("Frequency (Hz)")
                        ax[0].set_ylabel("Phase V (m/s)")
                        
                        ax[1].imshow(cur_norm, aspect='auto', origin='lower', cmap='jet', extent=ext)
                        ax[1].set_title(f"Sintetik Iter {i}")
                        ax[1].set_xlabel("Frequency (Hz)")
                        
                        ax[2].plot(v_plot, d_plot, 'r-', linewidth=2)
                        ax[2].set_ylim(max(d_plot), 0)
                        ax[2].set_xlabel("Vs (m/s)")
                        ax[2].set_ylabel("Depth (m)")
                        ax[2].set_title("1D Vs Model Sementara")
                        
                        plot_placeholder.pyplot(fig)
                        plt.close(fig)
                    return False
                        
                with st.spinner("Mengoptimasi parameter menggunakan Differential Evolution..."):
                    fp = st.session_state.forward_params
                    fp['return_seismogram'] = False
                    best_H, best_Vs, best_misfit = run_inversion(
                        E_obs=st.session_state.E_obs,
                        num_layers=num_layers,
                        bounds_H=bounds_H,
                        bounds_Vs=bounds_Vs,
                        forward_params=fp,
                        maxiter=int(maxiter),
                        popsize=int(popsize),
                        disp=False,
                        callback=de_callback
                    )
                    
                    # Simpan hasil ke session_state agar persist
                    st.session_state.de_result = {
                        'best_H': best_H, 'best_Vs': best_Vs, 'best_misfit': best_misfit,
                        'misfit_history': misfit_history, 'iter_models': iter_models,
                        'use_true_model': use_true_model
                    }
                    st.success(f"Optimisasi DE Selesai! Misfit Value: {best_misfit:.4f}")
        
        # Tampilkan hasil DE dari session_state (persist)
        if 'de_result' in st.session_state and "DE" in algoritma:
            res = st.session_state.de_result
            best_H, best_Vs, best_misfit = res['best_H'], res['best_Vs'], res['best_misfit']
            
            # Kurva Misfit (persist)
            if res.get('misfit_history'):
                st.markdown("#### Kurva Misfit L2 (DE)")
                fig_misfit = px.line(
                    y=res['misfit_history'],
                    x=list(range(1, len(res['misfit_history']) + 1)),
                    labels={'x': 'Iterasi', 'y': 'Misfit L2'}
                )
                fig_misfit.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_misfit, use_container_width=True)
                csv_misfit = pd.DataFrame({'Iterasi': range(1, len(res['misfit_history'])+1), 'Misfit_L2': res['misfit_history']}).to_csv(index=False).encode('utf-8')
                st.download_button("📥 Unduh Kurva Misfit (CSV)", data=csv_misfit, file_name='de_misfit_curve.csv', mime='text/csv', key='dl_de_misfit')
            
            st.markdown("#### Hasil Model 1D (Inverted)")
            
            def build_step_profile(H, Vs, max_ext=15.0):
                z = [0]
                v = [Vs[0]]
                for i, h in enumerate(H):
                    z.append(z[-1] + h)
                    v.append(Vs[i])
                    z.append(z[-1])
                    v.append(Vs[i+1])
                z.append(z[-1] + max_ext)
                v.append(Vs[-1])
                return z, v
            
            z_inv, v_inv = build_step_profile(best_H, best_Vs)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=v_inv, y=z_inv, mode='lines', name='Inverted Model', line=dict(color='red', width=3, dash='dash')))
            
            if res.get('use_true_model') and 'model_df' in st.session_state:
                df_true = st.session_state.model_df
                true_H_plot = df_true['Thickness (m)'].values[:-1]
                true_Vs_plot = df_true['Vs (m/s)'].values
                z_true, v_true = build_step_profile(true_H_plot, true_Vs_plot)
                fig.add_trace(go.Scatter(x=v_true, y=z_true, mode='lines', name='True Model', line=dict(color='green', width=3)))
                fig.update_layout(title="1D Shear Velocity Profile Comparison (True vs Inverted)")
            else:
                fig.update_layout(title="1D Inverted Shear Velocity Profile")
                
            fig.update_layout(
                xaxis_title="Velocity (m/s)",
                yaxis_title="Depth (m)",
                yaxis_autorange="reversed",
                template="plotly_white",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            df_inv = pd.DataFrame({
                "Layer": range(1, len(best_H) + 2),
                "Thickness (m)": list(best_H) + [np.nan],
                "Vs (m/s)": best_Vs
            })
            st.markdown("#### Tabel Parameter Model 1D")
            st.dataframe(df_inv)
            
            # Downloads
            st.markdown("---")
            col_d1, col_d2, col_d3 = st.columns(3)
            with col_d1:
                csv_de = df_inv.to_csv(index=False).encode('utf-8')
                st.download_button(label="📥 Unduh Model Terbaik (CSV)", data=csv_de, file_name='inversion_de_model.csv', mime='text/csv', key='dl_de_best')
            with col_d2:
                # Unduh semua iterasi + misfit
                if res.get('iter_models'):
                    rows = []
                    for m in res['iter_models']:
                        row = {'Iterasi': m['iter'], 'Misfit': m['misfit']}
                        for j, h in enumerate(m['H']):
                            row[f'H{j+1}'] = h
                        for j, v in enumerate(m['Vs']):
                            row[f'Vs{j+1}'] = v
                        rows.append(row)
                    csv_iter = pd.DataFrame(rows).to_csv(index=False).encode('utf-8')
                    st.download_button(label="📥 Unduh Semua Iterasi + Misfit (CSV)", data=csv_iter, file_name='de_iteration_history.csv', mime='text/csv', key='dl_de_iter')
            with col_d3:
                import json
                inv_params = {
                    'algorithm': 'Differential Evolution',
                    'best_misfit': float(best_misfit),
                    'num_layers': len(best_Vs),
                    'bounds_H': [list(b) for b in bounds_H] if bounds_H else [],
                    'bounds_Vs': [list(b) for b in bounds_Vs] if bounds_Vs else [],
                }
                st.download_button(label="📥 Unduh Parameter Inversi (JSON)", data=json.dumps(inv_params, indent=2), file_name='de_inversion_params.json', mime='application/json', key='dl_de_params')
# =============================================================================
# Modul 4: Real Field Data Processing
# =============================================================================
elif page == "Real Field Data Processing":
    st.markdown('<div class="main-header">Real Field Data Processing</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Unggah, inspeksi, dan proses rekaman seismik lapangan asli *(SEG-Y / SEG-2)* ke spektrum f-c</div>', unsafe_allow_html=True)
    
    if not OBSPY_AVAILABLE:
        st.error("Pustaka `obspy` tidak ditemukan di Environment saat ini. Modul ini membutuhkan `obspy` untuk membaca format standar seismik (.sgy/.seg2). Silakan tutup server lokal (`Ctrl+C`), jalankan `pip install obspy`, dan jalankan kembali `app.py`.")
    else:
        st.markdown("### 1. Unggah File Rekaman Lapangan")
        uploaded_file = st.file_uploader("Pilih file seismik (SEG-Y / SEG-2)", type=['sgy', 'segy', 'sg2', 'seg2'])
        
        if uploaded_file is not None:
            # Gunakan tempfile dengan ekstensi aslinya agar obspy mengenali format SEG-Y atau SEG-2
            ext = os.path.splitext(uploaded_file.name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
                
            try:
                with st.spinner("Membaca data seismik..."):
                    if ext in ['.sgy', '.segy']:
                        stream = obspy.read(tmp_path, format="SEGY")
                    elif ext in ['.sg2', '.seg2']:
                        stream = obspy.read(tmp_path, format="SEG2")
                    else:
                        stream = obspy.read(tmp_path)
                
                n_traces = len(stream)
                npts = stream[0].stats.npts
                dt = stream[0].stats.delta
                
                data = np.zeros((npts, n_traces), dtype=np.float32)
                for i, tr in enumerate(stream):
                    data[:, i] = tr.data
                    
                st.success("File berhasil dibaca & di-decode oleh ObsPy Engine.")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Jumlah Trace (Geophone)", n_traces)
                with col2:
                    st.metric("Sampling Rate (dt)", f"{dt} s")
                with col3:
                    durasi = (npts - 1) * dt
                    st.metric("Durasi Rekaman / Npts", f"{durasi:.3f} s  ({npts})")
                
                # Simpan data mentah ke session state HANYA saat file baru di-upload
                current_filename = uploaded_file.name
                if st.session_state.get('_loaded_filename') != current_filename:
                    st.session_state.raw_data = data.copy()
                    st.session_state.preproc_data = data.copy()
                    st.session_state.preproc_dt = dt
                    st.session_state.preproc_npts = npts
                    st.session_state.preproc_n_traces = n_traces
                    st.session_state._loaded_filename = current_filename
                
                st.markdown("---")
                st.markdown("### 2. Pre-Processing & Quality Control")
                st.caption("Terapkan operasi QC secara sekuensial sebelum transformasi dispersi. Klik 'Apply' pada setiap operasi untuk menerapkan perubahan.")
                
                # ---- Remove Mean ----
                with st.expander("📊 Remove Mean (DC Removal)", expanded=False):
                    st.write("Menghapus nilai rata-rata (DC offset) dari setiap trace sehingga sinyal berosilasi di sekitar nol.")
                    if st.button("Apply Remove Mean", key="rmean_btn"):
                        d = st.session_state.preproc_data
                        st.session_state.preproc_data = (d - np.mean(d, axis=0, keepdims=True)).astype(np.float32)
                        st.success("Remove Mean berhasil diterapkan pada semua trace!")
                
                # ---- Remove Trend ----
                with st.expander("📈 Remove Trend (Linear Detrend)", expanded=False):
                    st.write("Menghapus tren linear dari setiap trace menggunakan regresi linear (least-squares fit).")
                    if st.button("Apply Remove Trend", key="rtrend_btn"):
                        d = st.session_state.preproc_data
                        n_samp = d.shape[0]
                        x = np.arange(n_samp, dtype=np.float64)
                        for col in range(d.shape[1]):
                            y = d[:, col].astype(np.float64)
                            coeffs = np.polyfit(x, y, 1)  # linear: slope, intercept
                            trend = np.polyval(coeffs, x)
                            d[:, col] = (y - trend).astype(np.float32)
                        st.session_state.preproc_data = d
                        st.success("Remove Trend berhasil diterapkan pada semua trace!")
                
                # ---- A. Flip Trace Order ----
                with st.expander("🔄 Flip Susunan Trace", expanded=False):
                    st.write("Membalik urutan trace seismik (misal: diurutkan dari jauh ke dekat → dekat ke jauh)")
                    if st.button("Apply Flip", key="flip_btn"):
                        st.session_state.preproc_data = st.session_state.preproc_data[:, ::-1].copy()
                        st.success("Trace berhasil di-flip!")
                
                # ---- B. Kill Trace (Zero Amplitude) ----
                with st.expander("❌ Kill Trace (Nolkan Amplitudo)", expanded=False):
                    cur_n = st.session_state.preproc_data.shape[1]
                    st.write(f"Jumlah trace saat ini: **{cur_n}**. Trace yang di-kill akan tetap ada (posisi dipertahankan) namun amplitudonya dinolkan.")
                    kill_indices = st.text_input(
                        "Indeks trace yang akan di-kill (pisahkan koma, 1-indexed)",
                        placeholder="contoh: 1, 5, 12",
                        key="kill_input"
                    )
                    if st.button("Apply Kill", key="kill_btn"):
                        if kill_indices.strip():
                            try:
                                indices = [int(x.strip()) - 1 for x in kill_indices.split(",")]
                                valid = [i for i in indices if 0 <= i < cur_n]
                                if valid:
                                    for vi in valid:
                                        st.session_state.preproc_data[:, vi] = 0.0
                                    st.success(f"Berhasil menolkan amplitudo {len(valid)} trace (indeks: {[v+1 for v in valid]}). Posisi geophone tetap dipertahankan.")
                                else:
                                    st.warning("Tidak ada indeks valid yang ditemukan.")
                            except ValueError:
                                st.error("Format indeks tidak valid. Gunakan angka dipisahkan koma.")
                
                # ---- B2. Erase Trace (Hapus Trace + Posisi) ----
                with st.expander("🗑️ Erase Trace (Hapus Trace & Posisi)", expanded=False):
                    cur_n2 = st.session_state.preproc_data.shape[1]
                    st.write(f"Jumlah trace saat ini: **{cur_n2}**. Trace yang di-erase akan benar-benar dihilangkan beserta posisinya.")
                    erase_indices = st.text_input(
                        "Indeks trace yang akan di-erase (pisahkan koma, 1-indexed)",
                        placeholder="contoh: 1, 5, 12",
                        key="erase_input"
                    )
                    if st.button("Apply Erase", key="erase_btn"):
                        if erase_indices.strip():
                            try:
                                indices = [int(x.strip()) - 1 for x in erase_indices.split(",")]
                                valid = [i for i in indices if 0 <= i < cur_n2]
                                if valid:
                                    mask = np.ones(cur_n2, dtype=bool)
                                    mask[valid] = False
                                    st.session_state.preproc_data = st.session_state.preproc_data[:, mask].copy()
                                    st.session_state.preproc_n_traces = st.session_state.preproc_data.shape[1]
                                    st.success(f"Berhasil menghapus {len(valid)} trace (indeks: {[v+1 for v in valid]}). Sisa: {st.session_state.preproc_data.shape[1]} trace.")
                                else:
                                    st.warning("Tidak ada indeks valid yang ditemukan.")
                            except ValueError:
                                st.error("Format indeks tidak valid. Gunakan angka dipisahkan koma.")
                
                # ---- C. Time Window Cut ----
                with st.expander("✂️ Potong Waktu (Time Window)", expanded=False):
                    cur_npts = st.session_state.preproc_data.shape[0]
                    cur_dt = st.session_state.preproc_dt
                    max_time = (cur_npts - 1) * cur_dt
                    
                    col_tc1, col_tc2 = st.columns(2)
                    t_start = col_tc1.number_input("Waktu Awal (s)", value=0.0, min_value=0.0, max_value=float(max_time), step=0.001, format="%.4f", key="tcut_start")
                    t_end = col_tc2.number_input("Waktu Akhir (s)", value=float(max_time), min_value=0.0, max_value=float(max_time), step=0.001, format="%.4f", key="tcut_end")
                    
                    if st.button("Apply Time Cut", key="tcut_btn"):
                        i_start = max(0, int(t_start / cur_dt))
                        i_end = min(cur_npts, int(t_end / cur_dt) + 1)
                        if i_end > i_start:
                            st.session_state.preproc_data = st.session_state.preproc_data[i_start:i_end, :].copy()
                            st.session_state.preproc_npts = st.session_state.preproc_data.shape[0]
                            st.success(f"Data dipotong: {st.session_state.preproc_npts} sampel ({t_start:.4f}s - {t_end:.4f}s)")
                        else:
                            st.error("Rentang waktu tidak valid.")
                
                # ---- D. Polygon Mute (Body Wave Removal) ----
                with st.expander("📐 Polygon Mute (Hapus Body Wave)", expanded=False):
                    st.write("Definisikan poligon mute di domain (Trace, Waktu). Semua sampel **di atas** garis poligon akan di-nol-kan (mute).")
                    st.caption("Masukkan titik-titik poligon sebagai pasangan (trace_idx, time_s), pisahkan tiap titik dengan titik koma.")
                    
                    poly_input = st.text_input(
                        "Titik Poligon Mute (trace,waktu; ...)",
                        placeholder="contoh: 1,0.05; 12,0.10; 24,0.15",
                        key="poly_input"
                    )
                    mute_mode = st.radio("Mode Mute:", ["Mute di atas garis (first break)", "Mute di bawah garis (late arrivals)"], key="mute_mode_radio", horizontal=True)
                    
                    if st.button("Apply Polygon Mute", key="poly_btn"):
                        if poly_input.strip():
                            try:
                                points = []
                                for pt in poly_input.split(";"):
                                    parts = pt.strip().split(",")
                                    tr_idx = int(parts[0].strip()) - 1  # 1-indexed → 0-indexed
                                    t_val = float(parts[1].strip())
                                    points.append((tr_idx, t_val))
                                
                                if len(points) >= 2:
                                    cur_data = st.session_state.preproc_data
                                    cur_dt_val = st.session_state.preproc_dt
                                    n_samp, n_tr = cur_data.shape
                                    
                                    # Interpolasi linier untuk setiap trace
                                    poly_traces = np.array([p[0] for p in points])
                                    poly_times = np.array([p[1] for p in points])
                                    
                                    for tr_i in range(n_tr):
                                        # Interpolasi waktu mute di trace ini
                                        t_mute = np.interp(tr_i, poly_traces, poly_times)
                                        i_mute = int(t_mute / cur_dt_val)
                                        
                                        if "di atas" in mute_mode:
                                            cur_data[:max(0, i_mute), tr_i] = 0.0
                                        else:
                                            cur_data[min(n_samp, i_mute):, tr_i] = 0.0
                                    
                                    st.session_state.preproc_data = cur_data
                                    st.success(f"Polygon mute berhasil diterapkan pada {n_tr} trace.")
                                else:
                                    st.warning("Minimal 2 titik poligon diperlukan.")
                            except (ValueError, IndexError) as e:
                                st.error(f"Format poligon tidak valid: {e}")
                
                # ---- E. Reset to Original ----
                if st.button("🔁 Reset ke Data Original", key="reset_preproc"):
                    st.session_state.preproc_data = st.session_state.raw_data.copy()
                    st.session_state.preproc_npts = st.session_state.raw_data.shape[0]
                    st.session_state.preproc_n_traces = st.session_state.raw_data.shape[1]
                    st.success("Data dikembalikan ke kondisi original.")
                
                # ---- Preview Plot ----
                st.markdown("#### Preview Data Setelah Pre-Processing")
                preview_data = st.session_state.preproc_data
                p_npts = preview_data.shape[0]
                p_ntr = preview_data.shape[1]
                p_dt = st.session_state.preproc_dt
                t_arr_p = np.arange(p_npts) * p_dt
                
                fig_preview = go.Figure()
                dx_preview = 1.0  # normalized spacing for preview
                for i in range(p_ntr):
                    tr = preview_data[:, i]
                    tr_max = np.max(np.abs(tr))
                    if tr_max > 0:
                        tr = tr / tr_max
                    trace_disp = (tr * 0.4) + i + 1
                    fig_preview.add_trace(go.Scatter(
                        x=trace_disp, y=t_arr_p, mode='lines',
                        line=dict(color='#1e3a5f', width=0.8),
                        name=f"Tr-{i+1}", showlegend=False
                    ))
                fig_preview.update_layout(
                    xaxis_title="Trace #",
                    yaxis_title="Waktu (s)",
                    yaxis_autorange="reversed",
                    height=450,
                    template="plotly_white",
                    title=f"Preview: {p_ntr} trace × {p_npts} sampel (dt={p_dt:.4f}s)"
                )
                st.plotly_chart(fig_preview, use_container_width=True)
                
                # Update data & npts untuk downstream
                data = st.session_state.preproc_data
                npts = st.session_state.preproc_npts
                n_traces = st.session_state.preproc_n_traces
                
                st.markdown("---")
                st.markdown("### 3. Parameter Geometri dan Transformasi Dispersi")
                
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    st.markdown("**Akusisi Geometri Lapangan**")
                    min_off = st.number_input("Minimum Offset (Jarak sumber ke G1, m)", value=5.0, step=1.0)
                    dx = st.number_input("Spasi antar Geophone (m)", value=2.0, step=1.0)
                    
                with col_g2:
                    st.markdown("**Batas Pencarian *Phase-Shift***")
                    col_sp1, col_sp2 = st.columns(2)
                    c_min = col_sp1.number_input("Min Phase V (m/s)", value=100.0, step=10.0)
                    c_max = col_sp1.number_input("Max Phase V (m/s)", value=500.0, step=10.0)
                    dc = col_sp1.number_input("Resolusi Velocity (m/s)", value=10.0, step=1.0)
                    
                    f_min = col_sp2.number_input("Min Frequency (Hz)", value=5.0, step=1.0)
                    f_max = col_sp2.number_input("Max Frequency (Hz)", value=40.0, step=1.0)
                    
                if st.button("🚀 Transformasi Data Riil ke Spektrum f-c", type="primary"):
                    offsets = np.arange(n_traces) * dx + min_off
                    
                    with st.spinner("Mengaplikasikan Metode *Phase-Shift* (Park et al., 1999) pada data riil..."):
                        from seiswave.dispersion import calculate_dispersion_image
                        fs = 1.0 / dt
                        freqs, c_arr, E_raw = calculate_dispersion_image(
                            data, offsets, fs, c_min, c_max, dc, f_min, f_max
                        )
                        
                        # Simpan ke session state agar langsung bisa di-inversi
                        st.session_state.E_obs = E_raw 
                        st.session_state.forward_params = {
                            'offsets': offsets / 1000.0, # in km just in case used in seiswave LayeredModel checks
                            'dt': dt,
                            'npts': npts,
                            'c_min': c_min,
                            'c_max': c_max,
                            'dc': dc,
                            'f_min': f_min,
                            'f_max': f_max,
                            'nmodes': 2 # default, adjustable di Dispersion Inversion
                        }
                        
                        st.session_state.seismo_data = data
                        st.session_state.real_offsets = offsets
                        
                        st.success("Matriks Normalized f-c berhasil terekstrak! Data Anda kini otomatis terhubung ke Tab **Dispersion Inversion**.")
                        
                st.markdown("---")
                if 'real_offsets' in st.session_state and st.session_state.E_obs is not None:
                    
                    # 1. Plot Seismogram Ripple
                    st.markdown("#### A. Shot Gather (Raw Wiggle Trace)")
                    fig_seis = go.Figure()
                    t_arr = np.arange(npts) * dt
                    offsets = st.session_state.real_offsets
                    dx_real = (offsets[1] - offsets[0]) if len(offsets) > 1 else max(1.0, dx)
                    
                    data_real = st.session_state.seismo_data
                    for i in range(len(offsets)):
                        tr = data_real[:, i]
                        tr_max = np.max(np.abs(tr))
                        if tr_max > 0:
                            tr = tr / tr_max  # Normalize ke range [-1, 1]
                        
                        # Set amplitudo visual sekitar 40-50% spasi geophone
                        trace_display = (tr * dx_real * 0.4) + offsets[i]
                        
                        fig_seis.add_trace(go.Scatter(
                            x=trace_display, y=t_arr, mode='lines', 
                            line=dict(color='#0f172a', width=1),
                            name=f"Tr-{i+1} : {offsets[i]:.0f}m"
                        ))
                    
                    fig_seis.update_layout(
                        xaxis_title="Jarak Offset (m) + Magnitudo Amplitudo",
                        yaxis_title="Waktu (s)",
                        yaxis_autorange="reversed",
                        showlegend=False,
                        height=550,
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_seis, use_container_width=True)
                    
                    # 2. Plot f-c Dispersion
                    st.markdown("#### B. Spektrum Energi Dispersi (f-c)")
                    f_arr = np.linspace(st.session_state.forward_params['f_min'], 
                                        st.session_state.forward_params['f_max'], 
                                        st.session_state.E_obs.shape[1])
                    c_arr_plot = np.arange(st.session_state.forward_params['c_min'], 
                                        st.session_state.forward_params['c_max'] + st.session_state.forward_params['dc'], 
                                        st.session_state.forward_params['dc'])
                    
                    fig_fc = px.imshow(
                        st.session_state.E_obs,
                        x=f_arr, 
                        y=c_arr_plot,
                        color_continuous_scale='jet',
                        origin='lower',
                        labels=dict(x="Frequency (Hz)", y="Phase Velocity (m/s)", color="Normalized Energy"),
                        aspect='auto'
                    )
                    fig_fc.update_layout(height=500)
                    st.plotly_chart(fig_fc, use_container_width=True)
                    
                    st.markdown("---")
                    import json
                    col_dl1, col_dl2, col_dl3, col_dl4 = st.columns(4)
                    with col_dl1:
                        csv_real_seis = pd.DataFrame(st.session_state.seismo_data).to_csv(index=False).encode('utf-8')
                        st.download_button(label="📥 Seismogram (CSV)", data=csv_real_seis, file_name='real_shot_gather.csv', mime='text/csv', key='dl_real_seis')
                    with col_dl2:
                        csv_real_fc = pd.DataFrame(st.session_state.E_obs).to_csv(index=False).encode('utf-8')
                        st.download_button(label="📥 Spektrum f-c (CSV)", data=csv_real_fc, file_name='real_fc_spectrum.csv', mime='text/csv', key='dl_real_fc')
                    with col_dl3:
                        # Parameter pre-processing
                        preproc_info = {
                            'original_npts': int(st.session_state.raw_data.shape[0]) if 'raw_data' in st.session_state else None,
                            'original_n_traces': int(st.session_state.raw_data.shape[1]) if 'raw_data' in st.session_state else None,
                            'current_npts': int(st.session_state.preproc_npts) if 'preproc_npts' in st.session_state else None,
                            'current_n_traces': int(st.session_state.preproc_n_traces) if 'preproc_n_traces' in st.session_state else None,
                            'dt': float(st.session_state.preproc_dt) if 'preproc_dt' in st.session_state else None,
                        }
                        st.download_button(label="📥 Pre-Processing Info (JSON)", data=json.dumps(preproc_info, indent=2), file_name='preprocessing_params.json', mime='application/json', key='dl_preproc')
                    with col_dl4:
                        # Forward params
                        fp_info = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in st.session_state.forward_params.items()}
                        st.download_button(label="📥 Forward Params (JSON)", data=json.dumps(fp_info, indent=2), file_name='real_forward_params.json', mime='application/json', key='dl_real_fwd')
                    
            except Exception as e:
                st.error(f"Gagal memproses file seismik: {str(e)}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)


# =============================================================================
# Modul 5: CPS vs seiswave Benchmark
# =============================================================================
elif page == "CPS vs seiswave Benchmark":
    st.markdown('<div class="main-header">Benchmark Engine: CPS vs seiswave</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Bandingkan waveform seismogram dan spektrum f-c yang dihasilkan oleh murni Python dengan program Fortran Computer Programs in Seismology (CPS).</div>', unsafe_allow_html=True)
    
    if not check_cps_installed():
        st.error("Program inti CPS (seperti `sprep96`, `sdisp96`, `sregn96`) tidak ditemukan di environment command-line/PATH Anda. Modul komparasi ini tidak dapat digunakan. Pastikan binary CPS ada di direktori /bin atau sudah di-register ke shell PATH environment OSX Anda.")
    elif 'full_model_df' not in st.session_state:
        st.warning("⚠️ Silakan bangun terlebih dahulu sebuah model lapisan 1D Pemandu Bumi di Tab 'Geological Model Builder'. Kami butuh profil lapisannya sebagai bahan uji komputasi perbandingan model secara serentak.")
    else:
        st.success("✅ Engine CPS (.exe/bin Fortran) terhubung secara native dengan terminal komputer mesin Anda!")
        
        with st.expander("Parameter Konfigurasi Sintesis", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Akusisi Geometri Lapangan**")
                min_off = st.number_input("Min Offset (m)", value=10.0, step=1.0, key="cps_moff")
                max_off = st.number_input("Max Offset (m)", value=40.0, step=1.0, key="cps_xoff")
                dx = st.number_input("Spasi Geophone (m)", value=5.0, step=1.0, key="cps_dx")
            with col2:
                st.markdown("**Domain Waktu Sintetik**")
                dt = st.number_input("Sampling Rate dt (s)", value=0.002, format="%.4f", key="cps_dt")
                npts = st.number_input("Jumlah Sampel (npts)", value=256, step=64, key="cps_npts")
            with col3:
                st.markdown("**Transformasi Spektrum f-c**")
                c_min = st.number_input("Min V_phase (m/s)", value=100.0, step=10.0, key="cps_cmin")
                c_max = st.number_input("Max V_phase (m/s)", value=500.0, step=10.0, key="cps_cmax")
                dc = st.number_input("Resolusi Velocity (m/s)", value=10.0, step=1.0, key="cps_dc")
                f_min = st.number_input("Min Frequency (Hz)", value=5.0, step=1.0, key="cps_fmin")
                f_max = st.number_input("Max Frequency (Hz)", value=40.0, step=1.0, key="cps_fmax")
                nmodes_input = st.number_input("Jumlah Mode Dispersi (nmodes)", value=2, min_value=1, max_value=20, step=1, key="cps_nmodes")
                
        if st.button("Jalankan Uji Komparasi (Benchmark)", type="primary"):
            offsets = np.arange(min_off, max_off + dx, dx) / 1000.0 # to km
            fwd_params = {
                'offsets': offsets,
                'dt': dt,
                'npts': int(npts),
                'c_min': c_min,
                'c_max': c_max,
                'dc': dc,
                'f_min': f_min,
                'f_max': f_max,
                'nmodes': int(nmodes_input),
                'return_seismogram': True
            }
            
            df = st.session_state.full_model_df
            H = df['Thickness (m)'].values[:-1]
            Vs = df['Vs (m/s)'].values
            Vp = df['Vp (m/s)'].values
            rho = df['Density (kg/m³)'].values
            Qs = df['Qs'].values
            Qp = df['Qp'].values
            
            st.markdown("---")
            col_stat1, col_stat2 = st.columns(2)
            
            with col_stat1:
                with st.spinner("⏳ Menjalankan seiswave (Native Python Processing)..."):
                    import time
                    t0 = time.time()
                    E_py, data_py = generate_synthetic_spectrum(
                        H, Vp, Vs, rho, Qp, Qs, fwd_params
                    )
                    t_py = time.time() - t0
                st.success(f"seiswave Selesai: {t_py:.2f} detik")
                
            with col_stat2:
                with st.spinner("⏳ Menjalankan program Fortran CPS..."):
                    from seiswave.cps_runner import run_cps_forward
                    t1 = time.time()
                    data_cps = run_cps_forward(H, Vp, Vs, rho, Qp, Qs, fwd_params)
                    
                    # Convert cps seismo to f-c energy using dispersion phase shift
                    from seiswave.dispersion import calculate_dispersion_image
                    fs = 1.0 / dt
                    freqs, c_arr, E_raw_cps = calculate_dispersion_image(
                        data_cps, offsets * 1000.0, fs, c_min, c_max, dc, f_min, f_max
                    )
                    t_cps = time.time() - t1
                st.success(f"CPS Core Selesai: {t_cps:.2f} detik")
                
            st.info("💡 **Catatan Evaluasi Waktu:** seiswave melakukan kalkulasi murni (*Modal Summation -> Seismo -> Dispersi*) di dalam *Memory* interpreter Python tanpa pindah file eksternal, sedangkan CPS yang di-*wrap* ini harus melompati transfer/I-O antar command terminal disk dengan file Fortran mentah.")
            
            # Normalisasi F-C Keduanya
            max_c_py = np.max(E_py, axis=0, keepdims=True)
            E_py_norm = E_py / (max_c_py + 1e-12)
            
            max_c_cps = np.max(E_raw_cps, axis=0, keepdims=True)
            E_cps_norm = E_raw_cps / (max_c_cps + 1e-12)
            
            # Plot Seismo Comparison
            st.markdown("### 1. Komparasi Waveform (Seismogram Sintetik)")
            fig_seis = go.Figure()
            t_arr = np.arange(npts) * dt
            real_off = offsets * 1000.0
            dx_step = dx
            
            data_py_real = np.real(data_py)
            data_cps_real = data_cps
            
            for i in range(len(offsets)):
                tr_py = data_py_real[:, i]
                tr_cps = data_cps_real[:, i]
                
                max_tr = max(np.max(np.abs(tr_py)), np.max(np.abs(tr_cps))) + 1e-20
                tr_py = tr_py / max_tr
                tr_cps = tr_cps / max_tr
                
                trace_py_display = (tr_py * dx_step * 0.4) + real_off[i]
                trace_cps_display = (tr_cps * dx_step * 0.4) + real_off[i]
                
                show_l = True if i == len(offsets)-1 else False
                
                fig_seis.add_trace(go.Scatter(
                    x=trace_cps_display, y=t_arr, mode='lines', 
                    line=dict(color='black', width=3),
                    name="CPS (Fortran Solid)" if show_l else "",
                    showlegend=show_l
                ))
                fig_seis.add_trace(go.Scatter(
                    x=trace_py_display, y=t_arr, mode='lines', 
                    line=dict(color='red', width=1.5, dash='dash'),
                    name="seiswave (Dashed)" if show_l else "",
                    showlegend=show_l
                ))
                
            fig_seis.update_layout(
                xaxis_title="Jarak Offset (m) + Amplitudo Waveform Relatif",
                yaxis_title="Time (s)",
                yaxis_autorange="reversed",
                height=650,
                template="plotly_white"
            )
            st.plotly_chart(fig_seis, use_container_width=True)
            
            # Plot Dispersion Comparison
            st.markdown("### 2. Komparasi Spektrum Energi / Citra Dispersi (f-c)")
            from plotly.subplots import make_subplots
            fig_fc = make_subplots(rows=1, cols=2, subplot_titles=("seiswave (Native Python)", "CPS (Computer Programs in Seismology)"), horizontal_spacing=0.05)
            
            f_arr = np.linspace(f_min, f_max, E_py.shape[1])
            c_arr_plot = np.arange(c_min, c_max + dc, dc)
            
            # seiswave (Left)
            fig_fc.add_trace(
                go.Heatmap(z=E_py_norm, x=f_arr, y=c_arr_plot, colorscale='jet', colorbar=dict(x=0.45)),
                row=1, col=1
            )
            # CPS (Right)
            fig_fc.add_trace(
                go.Heatmap(z=E_cps_norm, x=f_arr, y=c_arr_plot, colorscale='jet', colorbar=dict(x=1.0)),
                row=1, col=2
            )
            
            fig_fc.update_xaxes(title_text="Frequency (Hz)", row=1, col=1)
            fig_fc.update_xaxes(title_text="Frequency (Hz)", row=1, col=2)
            fig_fc.update_yaxes(title_text="Phase Velocity (m/s)", row=1, col=1)
            
            fig_fc.update_layout(height=450)
            st.plotly_chart(fig_fc, use_container_width=True)
            # Correlation Metric Cross-check Analisis
            corr_scores = []
            for i in range(len(offsets)):
                py_t = data_py_real[:, i]
                cps_t = data_cps_real[:, i]
                cc = np.corrcoef(py_t, cps_t)[0, 1]
                corr_scores.append(cc)
            avg_corr = np.mean(corr_scores)
            
            fc_corr = np.corrcoef(E_py_norm.flatten(), E_cps_norm.flatten())[0, 1]
            
            st.markdown("### Kesimpulan Benchmark")
            
            # Tampilkan metrik seismogram
            if avg_corr > 0.95:
                st.success(f"**Tingkat Kesamaan Waveform Signal (*Zero-lag Cross-Correlation*): {avg_corr * 100:.3f}%** — seiswave mereplikasi performa waveform waktu CPS dengan sempurna!")
            else:
                st.warning(f"**Tingkat Kesamaan Waveform Signal (*Zero-lag Cross-Correlation*): {avg_corr * 100:.3f}%** — Terdapat perbedaan amplitudo (kemungkinan akibat handling perambatan/taper P-wave).")
                
            # Tampilkan metrik f-c image
            if fc_corr > 0.95:
                st.success(f"**Tingkat Kesamaan Citra Dispersi (*2D Matrix F-C Correlation*): {fc_corr * 100:.3f}%** — Distribusi energi frekuensi-kecepatan seiswave konsisten penuh melawan CPS!")
            else:
                st.warning(f"**Tingkat Kesamaan Citra Dispersi (*2D Matrix F-C Correlation*): {fc_corr * 100:.3f}%** — Terdapat deviasi pada rentang frekuensi tertentu.")

            st.markdown("---")
            col_bench1, col_bench2 = st.columns(2)
            with col_bench1:
                csv_diff_seis = pd.DataFrame(np.abs(data_py_real - data_cps_real)).to_csv(index=False).encode('utf-8')
                st.download_button("Unduh Selisih Absolut Seismogram (CSV)", data=csv_diff_seis, file_name="benchmark_diff_seis.csv", mime="text/csv")
            with col_bench2:
                csv_diff_fc = pd.DataFrame(np.abs(E_py_norm - E_cps_norm)).to_csv(index=False).encode('utf-8')
                st.download_button("Unduh Selisih Absolut f-c (CSV)", data=csv_diff_fc, file_name="benchmark_diff_fc.csv", mime="text/csv")
