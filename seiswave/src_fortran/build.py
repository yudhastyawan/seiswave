#!/usr/bin/env python3
"""
Cross-platform build script for CPS Native Python Extension (cps_core).

Supports:
  - macOS (arm64 / x86_64)
  - Linux (x86_64 / aarch64)
  - Windows (x86_64)

Requirements:
  - Python 3.x with numpy (f2py)
  - gfortran (GNU Fortran compiler)

Usage:
  python build.py          # auto-detect platform and build
  python build.py --clean  # remove all built .so/.pyd files
"""

import os
import sys
import glob
import struct
import platform
import subprocess
import shutil


# ---- Configuration ----
MODULE_NAME = "cps_core"
PYF_FILE = "cps_core.pyf"
FORTRAN_SOURCES = [
    "sprep96.f", "sdisp96.f", "sregn96.f", "spulse96.f",
    "mgtarg_f2py.f", "igetmod.f", "lgstr.f", "mchdep.f",
    "f96subf.f", "f2csub.f", "sio.f"
]
F77_FLAGS = "-ffixed-line-length-132"


def get_platform_info():
    """Detect current OS, architecture, and Python details."""
    os_name = platform.system().lower()     # 'darwin', 'linux', 'windows'
    machine = platform.machine().lower()    # 'arm64', 'x86_64', 'aarch64', 'amd64'
    bits = struct.calcsize("P") * 8         # 32 or 64
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"

    # Normalize arch
    if machine in ('arm64', 'aarch64'):
        arch = 'arm64'
    elif machine in ('x86_64', 'amd64', 'x64'):
        arch = 'x86_64'
    else:
        arch = machine

    # Normalize OS
    if os_name == 'darwin':
        os_label = 'macos'
    elif os_name == 'windows':
        os_label = 'windows'
    else:
        os_label = 'linux'

    return {
        'os': os_label,
        'os_raw': os_name,
        'arch': arch,
        'machine_raw': machine,
        'bits': bits,
        'python_version': py_ver,
        'ext': '.pyd' if os_label == 'windows' else '.so'
    }


def find_gfortran(info):
    """Find the best gfortran compiler for the current platform."""
    candidates = []

    if info['os'] == 'macos':
        if info['arch'] == 'arm64':
            # Homebrew arm64 paths
            candidates += glob.glob("/opt/homebrew/bin/gfortran*")
        else:
            # Homebrew Intel paths
            candidates += glob.glob("/usr/local/bin/gfortran*")
        # Fallback: any gfortran on PATH
        candidates.append("gfortran")

    elif info['os'] == 'linux':
        candidates += glob.glob("/usr/bin/gfortran*")
        candidates.append("gfortran")

    elif info['os'] == 'windows':
        # MSYS2 / MinGW-w64 paths
        candidates += glob.glob("C:/msys64/mingw64/bin/gfortran.exe")
        candidates += glob.glob("C:/mingw64/bin/gfortran.exe")
        # Conda-forge gfortran installs as x86_64-w64-mingw32-gfortran.exe
        if 'CONDA_PREFIX' in os.environ:
            conda_bin = os.path.join(os.environ['CONDA_PREFIX'], 'Library', 'bin')
            candidates.append(os.path.join(conda_bin, 'gfortran.exe'))
            candidates.append(os.path.join(conda_bin, 'x86_64-w64-mingw32-gfortran.exe'))
        candidates.append("gfortran")
        candidates.append("x86_64-w64-mingw32-gfortran")

    # Test each candidate
    for fc in candidates:
        fc_path = shutil.which(fc) if not os.path.isabs(fc) else fc
        if fc_path and os.path.isfile(fc_path):
            try:
                result = subprocess.run([fc_path, "--version"],
                                       capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return fc_path
            except (subprocess.TimeoutExpired, OSError):
                continue

    return None


def clean(script_dir):
    """Remove all compiled extension files."""
    patterns = [f"{MODULE_NAME}*.so", f"{MODULE_NAME}*.pyd", f"{MODULE_NAME}*.dylib"]
    removed = 0
    for pat in patterns:
        for f in glob.glob(os.path.join(script_dir, pat)):
            os.remove(f)
            print(f"  Removed: {os.path.basename(f)}")
            removed += 1
    if removed == 0:
        print("  Nothing to clean.")


def build(script_dir):
    """Build the cps_core extension for the current platform."""
    info = get_platform_info()

    print("=" * 60)
    print("CPS Core Extension Builder")
    print("=" * 60)
    print(f"  OS:           {info['os']} ({info['os_raw']})")
    print(f"  Architecture: {info['arch']} ({info['machine_raw']})")
    print(f"  Python:       {info['python_version']} ({info['bits']}-bit)")
    print(f"  Extension:    {info['ext']}")
    print()

    # Find compiler
    gfortran = find_gfortran(info)
    if gfortran is None:
        print("ERROR: gfortran not found!")
        print()
        if info['os'] == 'macos':
            print("Install via Homebrew:")
            print("  brew install gcc")
        elif info['os'] == 'linux':
            print("Install via package manager:")
            print("  sudo apt install gfortran     # Debian/Ubuntu")
            print("  sudo dnf install gcc-gfortran  # Fedora/RHEL")
        elif info['os'] == 'windows':
            print("Install via MSYS2 (recommended):")
            print("  1. Install MSYS2: https://www.msys2.org/")
            print("  2. In MSYS2 terminal: pacman -S mingw-w64-x86_64-gcc-fortran")
            print("  3. Add C:\\msys64\\mingw64\\bin to PATH")
            print()
            print("Or install via Conda:")
            print("  conda install -c conda-forge gfortran")
        sys.exit(1)

    print(f"  Compiler:     {gfortran}")

    # Verify compiler architecture matches Python
    print()
    print("Verifying compiler <-> Python architecture match...")
    try:
        result = subprocess.run([gfortran, "-dumpmachine"], capture_output=True, text=True, timeout=5)
        gcc_target = result.stdout.strip()
        print(f"  Compiler target: {gcc_target}")

        # Check for mismatch
        if info['arch'] == 'arm64' and 'arm' not in gcc_target and 'aarch64' not in gcc_target:
            print(f"  WARNING: Compiler targets '{gcc_target}' but Python runs on arm64.")
            print("  The resulting extension may not load. Consider using a native arm64 gfortran.")
        elif info['arch'] == 'x86_64' and 'x86_64' not in gcc_target and 'amd64' not in gcc_target:
            print(f"  WARNING: Compiler targets '{gcc_target}' but Python runs on x86_64.")
    except Exception:
        print("  Could not verify compiler target (non-fatal).")

    # Clean old builds
    print()
    print("Cleaning old builds...")
    clean(script_dir)

    # Verify source files exist
    print()
    print("Checking source files...")
    for src in [PYF_FILE] + FORTRAN_SOURCES:
        full = os.path.join(script_dir, src)
        if not os.path.isfile(full):
            print(f"  ERROR: Missing source file: {src}")
            sys.exit(1)
    print(f"  All {len(FORTRAN_SOURCES) + 1} files present.")

    # Build command
    print()
    print("Building extension...")
    
    # Workaround for Python 3.11+ Windows where setuptools bundled distutils lacks msvccompiler,
    # and stdlib distutils fails to parse newer MinGW linker versions.
    if info['os'] == 'windows':
        import shutil
        py_ver = f"{sys.version_info.major}{sys.version_info.minor}"
        mingw_lib = os.path.join(sys.base_prefix, "libs", f"libpython{py_ver}.a")
        msvc_lib = os.path.join(sys.base_prefix, "libs", f"python{py_ver}.lib")
        if not os.path.exists(mingw_lib) and os.path.exists(msvc_lib):
            try:
                print(f"  MinGW lib missing. Copying {os.path.basename(msvc_lib)} -> {os.path.basename(mingw_lib)} to bypass gendef requirement.")
                shutil.copy(msvc_lib, mingw_lib)
            except Exception as e:
                print(f"  WARNING: Could not apply MinGW import library workaround: {e}")

        runner_path = os.path.join(script_dir, "run_f2py_patched.py")
        with open(runner_path, "w") as f:
            f.write('''import sys, os
os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

# 1. Bypass distutils.cygwinccompiler ld version parse bug
try:
    import distutils.cygwinccompiler
    orig_get_versions = distutils.cygwinccompiler.get_versions
    def patched_get_versions():
        res = orig_get_versions()
        res = tuple((x if x is not None else "2.30.0") for x in res)
        return res
    distutils.cygwinccompiler.get_versions = patched_get_versions
except Exception:
    pass

# 2. Bypass numpy.distutils.mingw32ccompiler hardcoding 'gcc' string
try:
    import numpy.distutils.mingw32ccompiler
    import distutils.ccompiler
    orig_set_executables = distutils.ccompiler.CCompiler.set_executables
    def patched_set_executables(self, **kwargs):
        cc = os.environ.get("CC", "gcc")
        cxx = os.environ.get("CXX", "g++")
        for k, v in kwargs.items():
            if type(v) is str:
                if v.startswith("gcc"):
                    kwargs[k] = v.replace("gcc", cc, 1)
                elif v.startswith("g++"):
                    kwargs[k] = v.replace("g++", cxx, 1)
        orig_set_executables(self, **kwargs)
    numpy.distutils.mingw32ccompiler.Mingw32CCompiler.set_executables = patched_set_executables
except Exception:
    pass

import numpy.f2py
sys.exit(numpy.f2py.main())
''')
        cmd = [sys.executable, runner_path, "-c", PYF_FILE]
    else:
        cmd = [sys.executable, "-m", "numpy.f2py", "-c", PYF_FILE]

    cmd += FORTRAN_SOURCES + [
        "-m", MODULE_NAME,
        f"--f77flags={F77_FLAGS}",
        "--fcompiler=gnu95",
        f"--f77exec={gfortran}",
        f"--f90exec={gfortran}",
    ]

    # Windows-specific: Use mingw32 for the C compiler wrappers
    if info['os'] == 'windows':
        cmd.append("--compiler=mingw32")

    print(f"  Command: {' '.join(cmd[:6])} ...")
    print()

    env = os.environ.copy()
    env["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

    # Windows-specific: Ensure gcc/g++ can be found if installed as x86_64-w64-mingw32-gcc
    if info['os'] == 'windows':
        import shutil
        if not shutil.which("gcc"):
            mingw_gcc = shutil.which("x86_64-w64-mingw32-gcc")
            if mingw_gcc:
                mingw_gcc = mingw_gcc.replace("\\", "/")
                env["CC"] = mingw_gcc
                print(f"  Set CC={mingw_gcc} to bypass distutils hardcodes")
                
                mingw_gpp = shutil.which("x86_64-w64-mingw32-g++")
                if mingw_gpp:
                    mingw_gpp = mingw_gpp.replace("\\", "/")
                    env["CXX"] = mingw_gpp
                    print(f"  Set CXX={mingw_gpp}")

    result = subprocess.run(cmd, cwd=script_dir, env=env)

    if result.returncode != 0:
        print()
        print("BUILD FAILED!")
        sys.exit(1)

    # Verify output
    print()
    print("Verifying build output...")
    built_files = glob.glob(os.path.join(script_dir, f"{MODULE_NAME}*{info['ext']}"))
    if not built_files:
        # f2py may produce .so even on some platforms
        built_files = glob.glob(os.path.join(script_dir, f"{MODULE_NAME}*.so"))
        built_files += glob.glob(os.path.join(script_dir, f"{MODULE_NAME}*.pyd"))

    if built_files:
        for bf in built_files:
            size_kb = os.path.getsize(bf) / 1024
            print(f"  ✅ {os.path.basename(bf)} ({size_kb:.0f} KB)")
        print()
        print("BUILD SUCCESSFUL!")
    else:
        print("  ERROR: No extension file was generated.")
        sys.exit(1)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if "--clean" in sys.argv:
        print("Cleaning built files...")
        clean(script_dir)
    else:
        build(script_dir)
