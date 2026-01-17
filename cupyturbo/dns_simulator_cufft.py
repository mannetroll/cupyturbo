"""
dns_simulator_numpy.py — 2D Homogeneous Turbulence DNS (NumPy / CuPy port)

This is the fast SciPy/CuPy implementation converted to NumPy/CuPy.

ONLY FFT selection:
  • CPU: numpy.fft
  • GPU: cupyx.scipy.fft (if available; enables get_fft_plan), else cupy.fft

Everything else is kept structurally the same.
"""
import time
import datetime as _dt
import math
import sys
from dataclasses import dataclass
from typing import Literal

import numpy as _np

try:
    print(" Checking CuPy...")
    import cupy as _cp
    _cp.show_config()
except Exception:  # CuPy is optional
    _cp = None
    print(" CuPy not installed")

# Cache for CUDA usability probing (None=unknown).
_CUPY_USABLE = None

import numpy as np  # keep both _np and np like your other files

# ===============================================================
# FFT selection (CPU: numpy.fft, GPU: cupyx.scipy.fft or cupy.fft)
# ===============================================================
_npfft = _np.fft

try:
    import cupyx.scipy.fft as _cpfft  # type: ignore
except Exception:
    _cpfft = None


def _fft_mod_for_state(S: "DnsState"):
    """
    FFT selection:
      - CPU: numpy.fft
      - GPU: cupyx.scipy.fft (fallback to cupy.fft if cupyx.scipy.fft is unavailable)
    """
    if S.backend == "gpu":
        if _cpfft is not None:
            return _cpfft
        return S.xp.fft
    return _npfft


# ===============================================================
# Fortran-style random generator used in PAO (port of frand)
# ===============================================================
def frand(seed_list):
    """
    Port of the Fortran LCG used in PAO:

      IMM = 420029
      IT  = 2017
      ID  = 5011

      seed = (seed*IMM + IT) mod ID
      r    = seed / ID

    `seed_list` is a 1-element list to mimic Fortran SAVE/INTENT(INOUT).
    """
    IMM = 420029
    IT = 2017
    ID = 5011

    seed_list[0] = (seed_list[0] * IMM + IT) % ID
    return np.float32(seed_list[0] / ID)

# ---------------------------------------------------------------------------
# Backend selection: xp = np (CPU) or cp (GPU, if available)
# ---------------------------------------------------------------------------

def _cupy_is_usable() -> bool:
    """
    Returns True if CuPy is importable *and* the CUDA runtime/driver setup looks usable.

    Conservative for frozen (PyInstaller) "lean" builds:
      - Missing CUDA DLLs → False
      - Driver missing/too old (cudaErrorInsufficientDriver) → False
      - No CUDA devices → False
    """
    global _CUPY_USABLE

    if _CUPY_USABLE is not None:
        return bool(_CUPY_USABLE)

    if _cp is None:
        _CUPY_USABLE = False
        return False

    try:
        ndev = int(_cp.cuda.runtime.getDeviceCount())
        _CUPY_USABLE = (ndev > 0)
    except Exception:
        _CUPY_USABLE = False

    return bool(_CUPY_USABLE)


def get_xp(backend: Literal["cpu", "gpu", "auto"] = "auto"):
    """
    backend = "gpu"  → force CuPy (error if not available/usable)
    backend = "cpu"  → force NumPy
    backend = "auto" → use CuPy only if it is usable, else NumPy
    """
    # Auto-select: try GPU first, but *only* if usable.
    if backend == "auto":
        if _cupy_is_usable():
            return _cp
        return _np

    if backend == "gpu":
        if _cp is None:
            raise RuntimeError("CuPy is not installed, but backend='gpu' was requested.")
        if not _cupy_is_usable():
            raise RuntimeError(
                "CuPy is installed, but CUDA is not usable on this machine "
                "(missing CUDA runtime DLLs, no GPU, or insufficient driver)."
            )
        return _cp

    # CPU forced
    return _np


# ---------------------------------------------------------------------------
# Fortran-style random generator used in PAO, port of frand(seed)
# ---------------------------------------------------------------------------
class Frand:
    IMM = 420029
    IT = 2017
    ID = 5011

    def __init__(self, seed: int = 1):
        self.seed = int(seed)

    def __call__(self) -> float:
        self.seed = (self.seed * self.IMM + self.IT) % self.ID
        return float(self.seed) / float(self.ID)


# ===============================================================
# Python equivalent of dnsCudaDumpUCFullCsv
# ===============================================================
def dump_uc_full_csv(S: "DnsState", UC_full, comp: int):
    N = S.Nbase
    NX_full = S.NX_full
    NZ_full = S.NZ_full
    NK_full = S.NK_full
    ND2 = N // 2

    if S.backend == "gpu":
        UC_local = _np.asarray(UC_full.get())
    else:
        UC_local = _np.asarray(UC_full)

    for i in range(NX_full):
        row_vals = []

        use_mode = (i < 2 * ND2)
        if use_mode:
            kx = i // 2
            imag_row = (i & 1) == 1
        else:
            kx = None
            imag_row = False

        for z in range(NZ_full):
            if use_mode and kx < NK_full:
                v = UC_local[comp, z, kx]
                val = float(v.imag if imag_row else v.real)
            else:
                val = 0.0
            row_vals.append(f"{val:10.5f}")

        print(",".join(row_vals))

    print(f"[CSV] Wrote UC_full, {NX_full}x{NZ_full}, comp={comp}")


# ---------------------------------------------------------------------------
# DNS state
# ---------------------------------------------------------------------------
@dataclass
class DnsState:
    xp: any                 # numpy or cupy module
    backend: str            # "cpu" or "gpu"

    Nbase: int
    NX: int
    NZ: int
    NK: int

    NX_full: int
    NZ_full: int
    NK_full: int

    Re: float
    K0: float
    visc: float
    cflnum: float
    seed_init: int = 1

    # Cached FFT module (numpy.fft or cupyx.scipy.fft or cupy.fft)
    fft: any = None

    # Reusable cuFFT plans (GPU only, when cupyx.scipy.fft is available)
    fft_plan_rfft2_ur_full: any = None
    fft_plan_irfft2_uc01: any = None

    # Precomputed grid constants for CFL computation (dx==dz==2*pi/N)
    inv_dx: float = 0.0

    # CFL scratch to avoid per-step allocations (full 3/2 grid)
    cfl_tmp: any = None
    cfl_absw: any = None

    # Time integration
    t: float = 0.0
    dt: float = 0.0
    cn: float = 1.0
    cnm1: float = 0.0
    it: int = 0

    # Spectral wavenumber vectors
    alfa: any = None
    gamma: any = None

    # Compact grid (AoS)
    ur: any = None
    uc: any = None

    # Full 3/2 grid (SoA)
    ur_full: any = None
    uc_full: any = None

    # Vorticity and non-linear history
    om2: any = None
    fnm1: any = None

    scratch1: any = None
    scratch2: any = None

    # Precomputed index grids for STEP3
    step3_z_indices: any = None
    step3_kx_indices: any = None
    step3_z_spec: any = None

    # STEP3 scratch buffers & constants
    step3_uc1_th: any = None
    step3_uc2_th: any = None
    step3_uc3_th: any = None

    step3_K2: any = None
    step3_GA: any = None
    step3_G2mA2: any = None
    step3_invK2_sub: any = None

    step3_ARG: any = None
    step3_DEN: any = None
    step3_NUM: any = None

    step3_mask_ix0: any = None
    step3_divxz: any = None

    def sync(self):
        if self.backend == "gpu":
            self.xp.cuda.Stream.null.synchronize()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Helper to create a DnsState
# ---------------------------------------------------------------------------
def create_dns_state(
    N: int = 8,
    Re: float = 1e5,
    K0: float = 100.0,
    CFL: float = 0.75,
    backend: Literal["cpu", "gpu", "auto"] = "auto",
    seed: int = 1,
) -> DnsState:
    xp = get_xp(backend)

    if backend == "auto":
        effective_backend = "gpu" if (_cp is not None and xp is _cp) else "cpu"
    else:
        effective_backend = backend

    print(f" backend:  {backend}")

    Nbase = N
    NX = N
    NZ = N

    NX_full = 3 * NX // 2
    NZ_full = 3 * NZ // 2
    NK_full = NX_full // 2 + 1

    NK = 3 * NX // 4 + 1
    NX_half = NX // 2

    visc = 1.0 / float(Re)

    state = DnsState(
        xp=xp,
        backend=effective_backend,
        Nbase=Nbase,
        NX=NX,
        NZ=NZ,
        NK=NK,
        NX_full=NX_full,
        NZ_full=NZ_full,
        NK_full=NK_full,
        Re=Re,
        K0=K0,
        visc=visc,
        cflnum=CFL,
        seed_init=int(seed),
    )

    # Cache FFT module for the chosen backend
    state.fft = _fft_mod_for_state(state)

    # Precompute inverse grid spacing (dx==dz==2*pi/N)
    state.inv_dx = float(state.Nbase) / (2.0 * math.pi)

    # Allocate arrays
    state.ur = xp.zeros((NZ, NX, 3), dtype=xp.float32)
    state.uc = xp.zeros((NZ, NK, 3), dtype=xp.complex64)

    state.ur_full = xp.zeros((3, NZ_full, NX_full), dtype=xp.float32)
    state.uc_full = xp.zeros((3, NZ_full, NK_full), dtype=xp.complex64)

    # CFL scratch buffers
    state.cfl_tmp = xp.empty((NZ_full, NX_full), dtype=xp.float32)
    state.cfl_absw = xp.empty((NZ_full, NX_full), dtype=xp.float32)

    state.om2 = xp.zeros((NZ, NX_half), dtype=xp.complex64)
    state.fnm1 = xp.zeros((NZ, NX_half), dtype=xp.complex64)

    state.alfa = xp.zeros((NX_half,), dtype=xp.float32)
    state.gamma = xp.zeros((NZ,), dtype=xp.float32)

    # Reusable cuFFT plans (GPU only)
    if state.backend == "gpu":
        plan_mod = None
        if _cpfft is not None and hasattr(_cpfft, "get_fft_plan"):
            plan_mod = _cpfft

        if plan_mod is not None:
            state.fft_plan_rfft2_ur_full = plan_mod.get_fft_plan(
                state.ur_full, axes=(1, 2), value_type="R2C"
            )
            state.fft_plan_irfft2_uc01 = plan_mod.get_fft_plan(
                state.uc_full[0:2],
                shape=(state.NZ_full, state.NX_full),
                axes=(1, 2),
                value_type="C2R",
            )
            print(f"FFT plan_mod: {plan_mod.__name__}")
        else:
            print("FFT plan_mod: None (using cupy.fft without cached plans)")

    # PAO-style initialization
    dns_pao_host_init(state)

    # DT and CN will be initialized in run_dns via CFL
    state.dt = 0.0
    state.cn = 1.0
    state.cnm1 = 0.0

    state.scratch1 = xp.zeros((NZ, NX_half), dtype=xp.complex64)
    state.scratch2 = xp.zeros((NZ, NX_half), dtype=xp.complex64)

    # Precompute index grids used in STEP3
    NZ = state.NZ
    NX_half = state.NX // 2
    state.step3_z_indices = xp.arange(NZ, dtype=xp.int32)
    state.step3_kx_indices = xp.arange(NX_half, dtype=xp.int32)
    NZ_half = NZ // 2
    zi = state.step3_z_indices
    state.step3_z_spec = xp.where(
        zi <= (NZ_half - 1),
        zi,
        zi + NZ_half,
    )

    # STEP3: preallocate gather buffers
    state.step3_uc1_th = xp.empty((NZ, NX_half), dtype=xp.complex64)
    state.step3_uc2_th = xp.empty((NZ, NX_half), dtype=xp.complex64)
    state.step3_uc3_th = xp.empty((NZ, NX_half), dtype=xp.complex64)

    # STEP3: precompute constant spectral grids
    ax = state.alfa[None, :]
    gz = state.gamma[:, None]
    ax2 = ax * ax
    gz2 = gz * gz

    state.step3_K2 = (ax2 + gz2).astype(xp.float32, copy=False)
    state.step3_GA = (gz * ax).astype(xp.float32, copy=False)
    state.step3_G2mA2 = (gz2 - ax2).astype(xp.float32, copy=False)

    if NX_half > 1:
        state.step3_invK2_sub = (
            xp.float32(1.0) / (state.step3_K2[:, 1:] + xp.float32(1.0e-30))
        ).astype(xp.float32, copy=False)
    else:
        state.step3_invK2_sub = xp.empty((NZ, 0), dtype=xp.float32)

    state.step3_ARG = xp.empty((NZ, NX_half), dtype=xp.float32)
    state.step3_DEN = xp.empty((NZ, NX_half), dtype=xp.float32)
    state.step3_NUM = xp.empty((NZ, NX_half), dtype=xp.complex64)

    state.step3_mask_ix0 = (state.step3_z_indices >= 1) & (xp.abs(state.gamma) > 0.0)

    NX32 = xp.float32(1.5) * xp.float32(state.Nbase)
    NZ32 = xp.float32(1.5) * xp.float32(state.Nbase)
    state.step3_divxz = xp.float32(1.0) / (NX32 * NZ32)

    return state


# ===============================================================
# PAO init
# ===============================================================
def dns_pao_host_init(S: DnsState):
    xp = S.xp
    N = S.NX
    NE = S.NZ
    ND2 = N // 2
    NED2 = NE // 2
    PI = np.float32(3.14159265358979)

    DXZ = np.float32(2.0) * PI / np.float32(N)
    K0 = np.float32(S.K0)
    NORM = PI * K0 * K0

    print("--- INITIALIZING NumPy/CuPy ---", _dt.datetime.now().strftime("%Y-%m-%d %H:%M"))
    print(f" N={N}, K0={int(K0)}, Re={S.Re}")

    alfa = np.zeros(ND2, dtype=np.float32)
    gamma = np.zeros(NE, dtype=np.float32)

    E1 = np.float32(1.0)
    E3 = np.float32(1.0) / E1

    DALFA = np.float32(1.0) / E1
    DGAMMA = np.float32(1.0) / E3

    for x in range(NED2):
        alfa[x] = np.float32(x) * DALFA

    gamma[0] = np.float32(0.0)
    for z in range(1, NED2 + 1):
        gamma[z] = np.float32(z) * DGAMMA
        gamma[NE - z] = -gamma[z]

    UR = np.zeros((ND2, NE, 2), dtype=np.complex64)

    seed = [int(S.seed_init)]
    RANVEC = np.zeros(97, dtype=np.float32)

    for _ in range(97):
        frand(seed)
    for i in range(97):
        RANVEC[i] = frand(seed)

    def random_from_vec(r: np.float32) -> np.float32:
        idx = int(float(r) * 97.0)
        if idx < 0:
            idx = 0
        if idx > 96:
            idx = 96
        v = RANVEC[idx]
        RANVEC[idx] = r
        return v

    for z in range(NE):
        gz = gamma[z]
        for x in range(NED2):
            r = frand(seed)
            th = np.float32(2.0) * PI * random_from_vec(r)

            ARG = np.complex64(np.cos(th) + 1j * np.sin(th))

            ax = alfa[x]
            K2 = np.float32(ax * ax + gz * gz)
            K = np.float32(np.sqrt(K2)) if K2 > 0.0 else np.float32(0.0)

            if ax == 0.0:
                UR[x, z, 1] = np.complex64(0.0 + 0.0j)
                ABSU2 = np.float32(np.exp(- (K / K0) * (K / K0)) / NORM)
                amp = np.float32(np.sqrt(ABSU2))
                UR[x, z, 0] = np.complex64(amp) * ARG
            else:
                denom = np.float32(1.0) + (gz * gz) / (ax * ax)
                ABSW2 = np.float32(np.exp(- (K / K0) * (K / K0)) / (denom * NORM))
                ampw = np.float32(np.sqrt(ABSW2))

                w = np.complex64(ampw) * ARG
                u = np.complex64(- (gz / ax)) * w

                UR[x, z, 1] = w
                UR[x, z, 0] = u

    UR[0, 0, 0] = np.complex64(0.0 + 0.0j)
    UR[0, 0, 1] = np.complex64(0.0 + 0.0j)

    for z in range(1, NED2):
        UR[0, NE - z, 0] = np.conj(UR[0, z, 0])
        UR[0, NE - z, 1] = np.conj(UR[0, z, 1])

    UR[:, NED2, 0] = 0.0 + 0.0j
    UR[:, NED2, 1] = 0.0 + 0.0j

    A1 = A2 = A3 = A4 = A5 = A6 = A7 = 0.0
    E110 = 0.0

    for x in range(ND2):
        x1 = (x == 0)
        ax2 = float(alfa[x]) * float(alfa[x])

        for z in range(NE):
            U1 = UR[x, z, 0]
            U3 = UR[x, z, 1]

            u1u1 = float(np.abs(U1) ** 2)
            u3u3 = float(np.abs(U3) ** 2)

            gz2 = float(gamma[z]) * float(gamma[z])
            K2 = ax2 + gz2
            m = 1.0 if x1 else 2.0

            A1 += m * u1u1
            A2 += m * u3u3
            A3 += m * u1u1 * ax2
            A4 += m * u1u1 * gz2
            A5 += m * u3u3 * ax2
            A6 += m * u3u3 * gz2
            A7 += m * (u1u1 + u3u3) * K2 * K2

            if x1:
                E110 += u1u1

    Q2 = A1 + A2
    W2 = A3 + A4 + A5 + A6
    visc = math.sqrt(Q2 * Q2 / (float(S.Re) * W2))
    S.visc = np.float32(visc)

    EP = visc * W2
    De = 2.0 * visc * visc * A7
    KOL = (visc * visc * visc / EP) ** 0.25
    NLAM = 0.0
    if E110 != 0.0:
        NLAM = 2.0 * A1 / E110

    a11 = 2.0 * A1 / Q2 - 1.0
    e11 = 2.0 * (A3 + A4) / W2 - 1.0
    tscale = 0.5 * Q2 / EP
    dxKol = float(DXZ) / KOL
    Lux = 2.0 * math.pi / math.sqrt(2.0 * A1 / A3)
    Luz = 2.0 * math.pi / math.sqrt(2.0 * A1 / A4)
    Lwx = 2.0 * math.pi / math.sqrt(2.0 * A2 / A5)
    Lwz = 2.0 * math.pi / math.sqrt(2.0 * A2 / A6)
    Ceps2 = 0.5 * Q2 * De / (EP * EP)

    print(f" N           ={N:12.0f}")
    print(f" Reynolds n. ={float(S.Re):12.1g}")
    print(f" K0          ={K0:12.0f}")
    print(f" Energy      ={Q2:12.4f}")
    print(f" WiWi        ={W2:12.4f}")
    print(f" Epsilon     ={EP:12.4f}")
    print(f" a11         ={a11:12.4f}")
    print(f" e11         ={e11:12.4f}")
    print(f" Time scale  ={tscale:12.4g}")
    print(f" Kolmogorov  ={KOL:12.4f}")
    print(f" Viscosity   ={visc:12.4f}")
    print(f" dx/Kol.     ={dxKol:12.4f}")
    print(f" 2Pi/Nlamda  ={NLAM:12.4f}")
    print(f" 2Pi/Lux     ={Lux:12.4f}")
    print(f" 2Pi/Luz     ={Luz:12.4f}")
    print(f" 2Pi/Lwx     ={Lwx:12.4f}")
    print(f" 2Pi/Lwz     ={Lwz:12.4f}")
    print(f" Deps.       ={De:12.4f}")
    print(f" Ceps2       ={Ceps2:12.4f}")
    print(f" E1          ={float(E1):12.4f}")
    print(f" E3          ={float(E3):12.4f}")
    print(f" PAO seed    ={seed[0]:12d}")

    for comp in range(2):
        for z in range(NED2 - 1, -1, -1):
            for x in range(ND2):
                UR[x, N - NED2 + z, comp] = UR[x, z + NED2, comp]
                if z <= (N - NE - 1):
                    UR[x, z + NED2, comp] = 0.0 + 0.0j

    NK = S.NK
    UC_host = np.zeros((NK, NE, 3), dtype=np.complex64)

    for z in range(NE):
        for x in range(ND2):
            for c in range(2):
                UC_host[x, z, c] = UR[x, z, c]

    NK_full = S.NK_full
    NZ_full = S.NZ_full
    UC_full_host = np.zeros((NK_full, NZ_full, 3), dtype=np.complex64)
    for z in range(NE):
        for x in range(ND2):
            for c in range(2):
                UC_full_host[x, z, c] = UR[x, z, c]

    print(f" PAO initialization OK. VISC={float(S.visc):.7g}")

    S.alfa = xp.asarray(alfa, dtype=xp.float32)
    S.gamma = xp.asarray(gamma, dtype=xp.float32)

    UC_xp = xp.asarray(UC_host)
    S.uc[...] = xp.transpose(UC_xp, (1, 0, 2))

    UC_full_xp = xp.asarray(UC_full_host)
    S.uc_full[...] = xp.transpose(UC_full_xp, (2, 1, 0))

    vfft_full_inverse_uc_full_to_ur_full(S)
    dns_calcom_from_uc_full(S)
    S.fnm1[...] = xp.zeros_like(S.om2)


# ---------------------------------------------------------------------------
# FFT helpers
# ---------------------------------------------------------------------------
def vfft_full_inverse_uc_full_to_ur_full(S: DnsState) -> None:
    xp = S.xp
    UC = S.uc_full
    fft = S.fft

    UC01 = UC[0:2, :, :]

    if S.backend == "cpu":
        ur01 = fft.irfft2(UC01, s=(S.NZ_full, S.NX_full), axes=(1, 2))
    else:
        plan = S.fft_plan_irfft2_uc01
        if plan is not None:
            with plan:
                ur01 = fft.irfft2(UC01, s=(S.NZ_full, S.NX_full), axes=(1, 2))
        else:
            ur01 = fft.irfft2(UC01, s=(S.NZ_full, S.NX_full), axes=(1, 2))

    # Match cuFFT-style unnormalized inverse
    ur01 *= xp.float32(S.NZ_full * S.NX_full)

    S.ur_full[0:2, :, :] = xp.asarray(ur01, dtype=xp.float32)
    S.ur_full[2, :, :] = xp.float32(0.0)


def vfft_full_forward_ur_full_to_uc_full(S: DnsState) -> None:
    UR = S.ur_full
    fft = S.fft

    if S.backend == "cpu":
        UC = fft.rfft2(UR, s=(S.NZ_full, S.NX_full), axes=(1, 2))
    else:
        plan = S.fft_plan_rfft2_ur_full
        if plan is not None:
            with plan:
                UC = fft.rfft2(UR, s=(S.NZ_full, S.NX_full), axes=(1, 2))
        else:
            UC = fft.rfft2(UR, s=(S.NZ_full, S.NX_full), axes=(1, 2))

    S.uc_full[...] = UC


# ---------------------------------------------------------------------------
# CALCOM — spectral vorticity from UC_full
# ---------------------------------------------------------------------------
def dns_calcom_from_uc_full(S: DnsState) -> None:
    xp = S.xp

    Nbase = int(S.Nbase)
    NX_half = Nbase // 2
    NZ = Nbase

    alfa_1d = S.alfa.astype(xp.float32)
    gamma_1d = S.gamma.astype(xp.float32)

    uc1_full = S.uc_full[0]
    uc2_full = S.uc_full[1]

    uc1 = uc1_full[:NZ, :NX_half]
    uc2 = uc2_full[:NZ, :NX_half]

    ax = alfa_1d[None, :]
    gz = gamma_1d[:, None]

    diff = gz * uc1 - ax * uc2

    diff_r = diff.real
    diff_i = diff.imag

    om_r = -diff_i
    om_i = diff_r

    S.om2[...] = xp.asarray(om_r + 1j * om_i, dtype=xp.complex64)


# ---------------------------------------------------------------------------
# STEP2B
# ---------------------------------------------------------------------------
def dns_step2b(S: DnsState) -> None:
    xp = S.xp

    N = S.Nbase
    NZ_full = S.NZ_full
    NK_full = S.NK_full

    UR = S.ur_full
    UC = S.uc_full

    u = UR[0]
    w = UR[1]

    xp.multiply(u, w, out=UR[2])
    xp.multiply(u, u, out=UR[0])
    xp.multiply(w, w, out=UR[1])

    vfft_full_forward_ur_full_to_uc_full(S)

    NX_half = N // 2
    z_mid = N
    kx_max = min(NX_half, NK_full)

    if z_mid < NZ_full and kx_max > 0:
        UC[0:3, z_mid, 0:kx_max] = xp.complex64(0.0 + 0.0j)


# ---------------------------------------------------------------------------
# STEP3
# ---------------------------------------------------------------------------
def dns_step3(S: DnsState) -> None:
    xp = S.xp

    om2 = S.om2
    fnm1 = S.fnm1
    alfa = S.alfa
    gamma = S.gamma
    uc_full = S.uc_full

    Nbase = int(S.Nbase)
    NX_half = Nbase // 2
    NZ = Nbase

    visc = xp.float32(S.visc)
    dt = xp.float32(S.dt)
    cn = xp.float32(S.cn)
    cnm1 = xp.float32(S.cnm1)

    z_spec = S.step3_z_spec
    divxz = S.step3_divxz
    GA = S.step3_GA
    G2mA2 = S.step3_G2mA2
    K2 = S.step3_K2

    uc0_low = uc_full[0, :, :NX_half]
    uc1_low = uc_full[1, :, :NX_half]
    uc2_low = uc_full[2, :, :NX_half]

    uc1_th = S.step3_uc1_th
    uc2_th = S.step3_uc2_th
    uc3_th = S.step3_uc3_th
    xp.take(uc0_low, z_spec, axis=0, out=uc1_th)
    xp.take(uc1_low, z_spec, axis=0, out=uc2_th)
    xp.take(uc2_low, z_spec, axis=0, out=uc3_th)

    tmp_FN = S.scratch1
    tmp_c = S.scratch2
    xp.subtract(uc1_th, uc2_th, out=tmp_FN)
    xp.multiply(tmp_FN, GA, out=tmp_FN)
    xp.multiply(uc3_th, G2mA2, out=tmp_c)
    xp.add(tmp_FN, tmp_c, out=tmp_FN)
    tmp_FN *= divxz

    VT = xp.float32(0.5) * visc * dt
    ARG = S.step3_ARG
    DEN = S.step3_DEN
    xp.multiply(K2, VT, out=ARG)
    xp.add(ARG, xp.float32(1.0), out=DEN)

    c2 = xp.float32(0.5) * dt * (xp.float32(2.0) + cnm1)
    c3 = -xp.float32(0.5) * dt * cnm1

    NUM = S.step3_NUM
    NUM[...] = om2

    xp.multiply(om2, ARG, out=tmp_c)
    NUM -= tmp_c

    xp.multiply(tmp_FN, c2, out=tmp_c)
    NUM += tmp_c
    xp.multiply(fnm1, c3, out=tmp_c)
    NUM += tmp_c

    xp.divide(NUM, DEN, out=om2)
    fnm1[...] = tmp_FN

    out1 = S.scratch1
    out2 = S.scratch2
    out1[...] = 0
    out2[...] = 0

    if NX_half > 1:
        invK2_sub = S.step3_invK2_sub

        out1[:, 1:] = om2[:, 1:]
        out1[:, 1:] *= invK2_sub
        out1[:, 1:] *= gamma[:, None]
        out1[:, 1:] *= xp.complex64(-1.0j)

        out2[:, 1:] = om2[:, 1:]
        out2[:, 1:] *= invK2_sub
        out2[:, 1:] *= alfa[1:][None, :]
        out2[:, 1:] *= xp.complex64(1.0j)

    out1[:, 0] = 0
    mask0 = S.step3_mask_ix0
    out1[mask0, 0] = xp.complex64(-1.0j) * (om2[mask0, 0] / gamma[mask0])

    uc_full[0, :NZ, :NX_half] = out1
    uc_full[1, :NZ, :NX_half] = out2

    S.cnm1 = float(cn)


# ===============================================================
# STEP2A core
# ===============================================================
def dns_step2a(S: DnsState) -> None:
    xp = S.xp
    N = S.Nbase
    NX = S.NX
    NZ = S.NZ
    NX_full = S.NX_full
    NZ_full = S.NZ_full
    NK_full = S.NK_full

    UC = S.uc_full

    hi_start = N // 2
    hi_end = min(3 * N // 4, NK_full - 1)
    if hi_start <= hi_end:
        UC[0:2, :, hi_start:hi_end + 1] = xp.complex64(0.0 + 0.0j)

    halfN = N // 2
    k_max = min(halfN, NK_full)
    if k_max > 0:
        z_mid_start = halfN
        z_mid_end = N
        z_top_start = N
        z_top_end = N + halfN
        UC[0:2, z_top_start:z_top_end, :k_max] = UC[0:2, z_mid_start:z_mid_end, :k_max]
        UC[0:2, z_mid_start:z_mid_end, :k_max] = xp.complex64(0.0 + 0.0j)

    fft = S.fft
    UC01 = UC[0:2, :, :]

    if S.backend == "cpu":
        ur01 = fft.irfft2(UC01, s=(NZ_full, NX_full), axes=(1, 2))
    else:
        plan = S.fft_plan_irfft2_uc01
        if plan is not None:
            with plan:
                ur01 = fft.irfft2(UC01, s=(NZ_full, NX_full), axes=(1, 2))
        else:
            ur01 = fft.irfft2(UC01, s=(NZ_full, NX_full), axes=(1, 2))

    # Match cuFFT-style unnormalized inverse
    ur01 *= xp.float32(NZ_full * NX_full)

    S.ur_full[0:2, :, :] = ur01.astype(xp.float32, copy=False)
    S.ur_full[2, :, :] = xp.float32(0.0)

    off_x = (NX_full - NX) // 2
    off_z = (NZ_full - NZ) // 2

    S.ur[:, :, 0] = S.ur_full[0, off_z:off_z + N, off_x:off_x + N]
    S.ur[:, :, 1] = S.ur_full[1, off_z:off_z + N, off_x:off_x + N]
    S.ur[:, :, 2] = 0.0


# ---------------------------------------------------------------------------
# NEXTDT — CFL
# ---------------------------------------------------------------------------
def compute_cflm(S: DnsState) -> float:
    xp = S.xp

    NX3D2 = S.NX_full
    NZ3D2 = S.NZ_full

    u = S.ur_full[0, :NZ3D2, :NX3D2]
    w = S.ur_full[1, :NZ3D2, :NX3D2]

    tmp = S.cfl_tmp[:NZ3D2, :NX3D2]
    absw = S.cfl_absw[:NZ3D2, :NX3D2]

    xp.abs(u, out=tmp)
    xp.abs(w, out=absw)
    xp.add(tmp, absw, out=tmp)

    CFLM = float(xp.max(tmp) * S.inv_dx)
    return CFLM


def next_dt(S: DnsState) -> None:
    PI = math.pi

    CFLM = compute_cflm(S)
    if CFLM <= 0.0 or S.dt <= 0.0:
        return

    CFL = CFLM * S.dt * PI
    S.cn = 0.8 + 0.2 * (S.cflnum / CFL)
    S.dt = S.dt * S.cn


# ===============================================================
# Python equivalent of dnsCudaDumpFieldAsPGMFull
# ===============================================================
def dump_field_as_pgm_full(S: DnsState, comp: int, filename: str) -> None:
    """
    Python/CuPy version of:

        void dnsCudaDumpFieldAsPGMFull(DnsDeviceState *S, int comp,
                                       const char *filename)

    Uses S.ur_full (3, NZ_full, NX_full), SoA layout:
      ur_full[comp, z, x]

    Writes an 8-bit binary PGM (P5) file with values mapped
    linearly from [minv, maxv] → [1, 255], same as the CUDA code.
    """
    NX_full = S.NX_full
    NZ_full = S.NZ_full

    # ------------------------------------------------------------
    # Bring UR_full to host as float32, layout [comp][z][x]
    # ------------------------------------------------------------
    if S.backend == "gpu":
        ur_full_host = _np.asarray(S.ur_full.get(), dtype=_np.float32)
    else:
        ur_full_host = _np.asarray(S.ur_full, dtype=_np.float32)

    # Selected component plane: shape (NZ_full, NX_full)
    field = ur_full_host[comp, :, :]

    # ------------------------------------------------------------
    # Compute min and max over the selected component
    # layout: [comp][z][x]
    # ------------------------------------------------------------
    minv = float(field.min())
    maxv = float(field.max())

    try:
        f = open(filename, "wb")
    except OSError as e:
        print(f"[DUMP] fopen failed for {filename!r}: {e}")
        return

    # P5 header: binary grayscale
    header = f"P5\n{NX_full} {NZ_full}\n255\n"
    f.write(header.encode("ascii"))

    rng = maxv - minv

    # ------------------------------------------------------------
    # Map [minv, maxv] → [1, 255]
    # If nearly constant field, use mid-grey 128
    # ------------------------------------------------------------
    if abs(rng) <= 1.0e-12:
        # field is essentially constant
        c = bytes([128])
        row = c * NX_full
        for _ in range(NZ_full):
            f.write(row)
    else:
        for j in range(NZ_full):
            for i in range(NX_full):
                val = float(field[j, i])

                # normalize to [0,1]
                norm = (val - minv) / rng   # 0 .. 1

                # scale to [1,255]
                pixf = 1.0 + norm * 254.0
                pix = int(pixf + 0.5)
                if pix < 1:
                    pix = 1
                if pix > 255:
                    pix = 255

                f.write(bytes([pix]))

    f.close()
    print(f"[DUMP] Wrote {filename} (PGM, {NX_full}x{NZ_full}, "
          f"comp={comp}, min={minv:g}, max={maxv:g})")

# ---------------------------------------------------------------------------
# Helpers for visualization fields (energy, vorticity, streamfunction)
# These are Python/xp equivalents of:
#   FIELD2KIN  + DNS_KINETIC
#   OM2PHYS    + DNS_OM2PHYS
#   STREAMFUNC + DNS_STREAMFUNC
#
# They follow the "dns_all" convention used by your GUI:
#   - dns_kinetic(S)     → fills S.ur_full[2, :, :] with |u|
#   - dns_om2_phys(S)    → fills S.ur_full[2, :, :] with ω(x,z)
#   - dns_stream_func(S) → fills S.ur_full[2, :, :] with φ(x,z)
#
# The GUI then does:
#     dns_all.dns_kinetic(S)
#     field = (cp.asnumpy or np.asarray)(S.ur_full[2, :, :])
#     plane = self._float_to_pixels(field)
# ---------------------------------------------------------------------------


def dns_kinetic(S: DnsState) -> None:
    """
    Fill S.ur_full[2, :, :] with the kinetic-energy magnitude |u|.

    Fortran FIELD2KIN computes:
        K = sqrt(UR(:,:,1)^2 + UR(:,:,2)^2)

    Here we compute the same quantity on the full 3/2 grid:
        ur_full[0] → u, ur_full[1] → w
        ur_full[2] ← sqrt(u^2 + w^2)
    """
    xp = S.xp

    u = S.ur_full[0, :, :]  # component 1
    w = S.ur_full[1, :, :]  # component 2

    ke = xp.sqrt(u * u + w * w)
    S.ur_full[2, :, :] = ke.astype(xp.float32)


def _spectral_band_to_phys_full_grid(S: DnsState, band) -> any:
    """
    Internal helper: take a compact spectral band band(z, kx) with
    shape (NZ, NX/2) and map it to a full 3/2-grid physical field
    using the same de-aliasing + reshuffle + inverse FFT sequence
    as STEP2A / OM2PHYS / STREAMFUNC.

    Returns a real array of shape (NZ_full, NX_full), dtype float32.
    """
    xp = S.xp

    N       = S.Nbase
    NX_full = S.NX_full      # 3*N/2
    NZ_full = S.NZ_full      # 3*N/2
    NK_full = S.NK_full      # 3*N/4 + 1

    NX_half = N // 2
    NZ      = N

    # 2D spectral buffer: layout [z, kx]
    uc_tmp = xp.zeros((NZ_full, NK_full), dtype=xp.complex64)

    # Copy compact band into low-k strip (Z=0..NZ-1, kx=0..NX/2-1)
    uc_tmp[:NZ, :NX_half] = band

    # ----------------------------------------------------------
    # Dealias high-kx: zero band [N/2 .. 3N/4] as in STEP2A
    # ----------------------------------------------------------
    hi_start = N // 2
    hi_end   = min(3 * N // 4, NK_full - 1)
    if hi_start <= hi_end:
        uc_tmp[:, hi_start:hi_end + 1] = xp.complex64(0.0 + 0.0j)

    # ----------------------------------------------------------
    # Z-reshuffle low-kz strip (STEP2A-style)
    #   for z_low = 0..N/2-1:
    #       z_mid = z_low + N/2
    #       z_top = z_low + N
    # ----------------------------------------------------------
    halfN = N // 2
    k_max = min(halfN, NK_full)

    for z_low in range(halfN):
        z_mid = z_low + halfN
        z_top = z_low + N
        if z_top >= NZ_full:
            break

        slice_mid = uc_tmp[z_mid, :k_max].copy()
        uc_tmp[z_top, :k_max] = slice_mid
        uc_tmp[z_mid, :k_max] = xp.complex64(0.0 + 0.0j)

    # Zero the "middle" Fourier coefficient Z = NZ+1 (1-based)
    z_mid = NZ  # 0-based index
    if z_mid < NZ_full:
        uc_tmp[z_mid, :NX_half] = xp.complex64(0.0 + 0.0j)

    # ----------------------------------------------------------
    # Inverse transforms (match CUFFT scaling used elsewhere):
    #   1) inverse along z  (complex→complex)
    #   2) inverse along x  (complex→real)
    # ----------------------------------------------------------
    tmp  = xp.fft.ifft(uc_tmp, axis=0) * NZ_full           # (NZ_full, NK_full)
    phys = xp.fft.irfft(tmp, n=NX_full, axis=1) * NX_full  # (NZ_full, NX_full)

    return phys.astype(xp.float32)


def dns_om2_phys(S: DnsState) -> None:
    """
    Fill S.ur_full[2, :, :] with the physical vorticity ω(x,z).
    """
    xp = S.xp

    band = S.om2
    phys = _spectral_band_to_phys_full_grid(S, band)

    S.ur_full[2, :, :] = phys


def dns_stream_func(S: DnsState) -> None:
    """
    Fill S.ur_full[2, :, :] with the streamfunction φ(x,z).
    """
    xp = S.xp

    N       = S.Nbase
    NX_half = N // 2
    NZ      = N

    alfa_1d  = S.alfa.astype(xp.float32)   # (NX_half,)
    gamma_1d = S.gamma.astype(xp.float32)  # (NZ,)

    ax = alfa_1d[None, :]                  # (1, NX_half)
    gz = gamma_1d[:, None]                 # (NZ, 1)

    K2 = ax * ax + gz * gz
    K2 = K2 + xp.float32(1.0e-30)

    phi_hat = S.om2 / K2

    phys = _spectral_band_to_phys_full_grid(S, phi_hat)
    S.ur_full[2, :, :] = phys

# ---------------------------------------------------------------------------
# Main driver (Python version of main in dns_all.cu)
# ---------------------------------------------------------------------------
def run_dns(
    N: int = 8,
    Re: float = 100,
    K0: float = 10.0,
    STEPS: int = 2,
    CFL: float = 0.75,
    backend: Literal["cpu", "gpu", "auto"] = "auto",
) -> None:
    print("--- INITIALIZING PYTHON (NumPy/CuPy) ---")
    print(f" N   = {N}")
    print(f" Re  = {Re}")
    print(f" K0  = {K0}")
    print(f" Steps = {STEPS}")
    print(f" CFL  = {CFL}")
    print(f" requested = {backend}")

    S = create_dns_state(N=N, Re=Re, K0=K0, CFL=CFL, backend=backend)
    print(f" effective = {S.backend} (xp = {'cupy' if S.backend == 'gpu' else 'numpy'})")

    dns_step2a(S)

    CFLM = compute_cflm(S)
    S.dt = S.cflnum / (CFLM * math.pi)
    S.cn = 1.0
    S.cnm1 = 0.0

    print(f" [NEXTDT INIT] CFLM={CFLM:11.4f} DT={S.dt:11.7f} CN={S.cn:11.7f}")
    print(f" Initial DT={S.dt:11.7f} CN={S.cn:11.7f}")

    S.sync()
    t0 = time.perf_counter()

    for it in range(1, STEPS + 1):
        S.it = it
        dt_old = S.dt

        dns_step2b(S)
        dns_step3(S)
        dns_step2a(S)

        S.t += dt_old

        if (it % 100) == 0 or it == 1 or it == STEPS:
            next_dt(S)
            print(f" ITERATION {it:6d} T={S.t:12.10f} DT={S.dt:10.8f} CN={S.cn:10.8f}")

    S.sync()
    t1 = time.perf_counter()

    elap = t1 - t0
    fps = (STEPS / elap) if elap > 0 else 0.0

    print(f" Elapsed CPU time for {STEPS} steps (s) = {elap:8g}")
    print(f" Final T={S.t:8g}  CN={S.cn:8g}  DT={S.dt:8g}")
    print(f" FPS = {fps:7g}")


def main():
    args = sys.argv[1:]
    N = int(args[0]) if len(args) > 0 else 512
    Re = float(args[1]) if len(args) > 1 else 10000
    K0 = float(args[2]) if len(args) > 2 else 10.0
    STEPS = int(args[3]) if len(args) > 3 else 101
    CFL = float(args[4]) if len(args) > 4 else 0.75

    BACK = args[5].lower() if len(args) > 5 else "auto"
    if BACK not in ("cpu", "gpu", "auto"):
        BACK = "auto"

    run_dns(N=N, Re=Re, K0=K0, STEPS=STEPS, CFL=CFL, backend=BACK)


if __name__ == "__main__":
    main()