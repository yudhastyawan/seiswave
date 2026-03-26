import os
import subprocess
import numpy as np
import tempfile
import glob
try:
    import obspy
except ImportError:
    pass

def check_cps_installed():
    """Periksa lokasi / instalasi CPS"""
    import shutil
    return shutil.which('sprep96') is not None

def write_cps_model(filename, H, Vp, Vs, rho, Qp, Qs, nmodes=100):
    with open(filename, 'w') as f:
        f.write("MODEL.01\n")
        f.write(f"sprep96 -M model.d -d dfile -NMOD {nmodes} -R -L -P\n")
        f.write("ISOTROPIC\n")
        f.write("KILOMETER\n")
        f.write("FLAT EARTH\n")
        f.write("1-D\n")
        f.write("CONSTANT VELOCITY\n")
        f.write("LINE08\n")
        f.write("LINE09\n")
        f.write("LINE10\n")
        f.write("LINE11\n")
        f.write(" H(KM)   VP(KM/S)  VS(KM/S)  RHO(GM/CC)  QP   QS  ETAP  ETAS  FREFP  FREFS\n")
        
        nlayers = len(Vs)
        for i in range(nlayers):
            # Input dari front-end UI adalah unit SI (m, m/s, kg/m3)
            # CPS meminta KM, KM/S, GM/CC
            h = H[i] / 1000.0 if i < len(H) else 0.0
            vp = Vp[i] / 1000.0
            vs = Vs[i] / 1000.0
            r = rho[i] / 1000.0
            qp = Qp[i]
            qs = Qs[i]
            f.write(f"  {h:15.8f}  {vp:15.8f}  {vs:15.8f}  {r:15.8f}  {qp:10.1f}  {qs:10.1f}  0.0  0.0  0.0  0.0\n")

def write_cps_dfile(filename, offsets, dt, npts):
    with open(filename, 'w') as f:
        for r in offsets:
            f.write(f"{r:.4f} {dt:.4f} {npts} 0.0 0.0\n")

def run_cps_forward(H, Vp, Vs, rho, Qp, Qs, forward_params):
    """
    Eksekusi otomatis seluruh alur CPS di folder temporary untuk menghasilkan
    matriks sintetik 2D seismogram.
    Menggunakan ekstensi Python Native f2py jika tersedia, fallback ke subprocess.
    """
    offsets = forward_params['offsets'] # Ini diasumsikan diubah ke km dari app.py
    dt = forward_params['dt']
    npts = forward_params['npts']
    nmodes = forward_params.get('nmodes', 100)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            model_path = "model.d"
            dfile_path = "dfile"
            
            write_cps_model(model_path, H, Vp, Vs, rho, Qp, Qs, nmodes=nmodes)
            write_cps_dfile(dfile_path, offsets, dt, npts)
            
            try:
                import sys
                import importlib.util
                import platform as _plat
                # Cari extension (.so di Unix, .pyd di Windows) secara dinamis
                base_dir = os.path.dirname(os.path.abspath(__file__))
                src_dir = os.path.join(base_dir, "src_fortran")
                mod_path = (
                    glob.glob(os.path.join(src_dir, "cps_core*.so")) +
                    glob.glob(os.path.join(src_dir, "cps_core*.pyd"))
                )
                if not mod_path:
                    _ext = '.pyd' if _plat.system() == 'Windows' else '.so'
                    raise ImportError(
                        f"cps_core{_ext} tidak ditemukan di {src_dir}. "
                        f"Jalankan: python seiswave/src_fortran/build.py"
                    )
                spec = importlib.util.spec_from_file_location("cps_core", mod_path[0])
                cps_core = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(cps_core)
                
                def run_native_stage(name, args):
                    # F2PY enforces dimension(50), pad
                    padded = args + [''] * (50 - len(args))
                    cps_core.pysetargs(len(args), padded)
                    getattr(cps_core, f"run_{name}")()

                run_native_stage("sprep96", ["sprep96", "-M", "model.d", "-d", "dfile", "-R", "-ALL"])
                run_native_stage("sdisp96", ["sdisp96"])
                run_native_stage("sregn96", ["sregn96", "-GAM"])
                
                # spulse96 redirect native stdout ke file96
                args = ["spulse96", "-d", "dfile", "-V", "-p", "-l", "1"]
                padded = args + [''] * (50 - len(args))
                cps_core.pysetargs(len(args), padded)
                
                with open("file96", "wb") as f96:
                    original_stdout_fd = os.dup(1)
                    os.dup2(f96.fileno(), 1)
                    try:
                        cps_core.run_spulse96()
                    finally:
                        os.dup2(original_stdout_fd, 1)
                        os.close(original_stdout_fd)

                # berhasil menggunakan f2py
                print("berhasil menggunakan f2py")
                        
            except ImportError as e:
                # Fallback ke Subprocess jika compiled error
                subprocess.run(["sprep96", "-M", "model.d", "-d", "dfile", "-R", "-ALL"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(["sdisp96"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(["sregn96", "-GAM"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                with open("file96", "w") as f96:
                    subprocess.run(["spulse96", "-d", "dfile", "-V", "-p", "-l", "1"], stdout=f96, check=True)

                print("berhasil menggunakan subprocess")
            
            subprocess.run(["f96tosac", "-FMT", "2", "file96"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            sac_files = sorted(glob.glob("*.ZVF"))
            if not sac_files:
                raise RuntimeError("CPS gagal men-_generate_ SAC files.")
                
            data_cps = np.zeros((npts, len(offsets)), dtype=np.float32)
            for i, sacf in enumerate(sac_files):
                if i >= len(offsets):
                    break
                st = obspy.read(sacf)
                tr = st[0]
                n_copy = min(npts, len(tr.data))
                data_cps[:n_copy, i] = tr.data[:n_copy]
                
            return data_cps
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Gagal menjalankan module F2PY CPS: {e}")
        finally:
            os.chdir(original_cwd)
