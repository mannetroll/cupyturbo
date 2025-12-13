"""
scipy_simulator.py — 2D Homogeneous Turbulence DNS (NumPy / CuPy port)

This is a structural port of dns_all.cu to Python.

Key ideas kept from the CUDA version:
  • DnsState structure mirrors DnsDeviceState (Nbase, NX, NZ, NK, NX_full, NZ_full, NK_full)
  • UR (compact)  : shape (NZ, NX, 3)   — AoS: [z, x, comp]
  • UC (compact)  : shape (NZ, NK, 3)   — spectral, [z, kx, comp]
  • UR_full (3/2) : shape (3, NZ_full, NX_full)   — SoA: [comp, z, x]
  • UC_full (3/2) : shape (3, NZ_full, NK_full)   — spectral, SoA
  • om2, fnm1     : shape (NZ, NX_half) — spectral vorticity & non-linear term
  • alfa[NX_half], gamma[NZ]           — wave-number vectors
  • Time loop     : STEP2B → STEP3 → STEP
"""
import time
import math
import sys
from dataclasses import dataclass
from typing import Literal

import numpy as _np

try:
    print(" Checking CuPy...")
    import cupy as _cp
    _cp.show_config()
    try:
        import cupyx.scipy.fft as _cupy_spfft
    except ImportError:
        _cupy_spfft = None
except ImportError:  # CuPy is optional
    _cp = None
    _cupy_spfft = None
    print(" CuPy not installed")

import numpy as np  # in addition to your existing _np alias, this is fine

import scipy.fft as _spfft

def check_cupy():
    try:
        print(" Checking CuPy...")
        import cupy as cp
        cp.show_config()
        return cp
    except Exception:
        return None


# ===============================================================
# Fortran-style random generator used in PAO (port of frand)
# ===============================================================
def frand(seed_list):
    """
    Port of the Fortran LCG used in PAO:

      IMM = 420029
      IT  = 2017
      ID  = 5011

    seed_list is a 1-element list (mutable int holder)
    returns float32 in (0,1)
    """
    IMM = 420029
    IT  = 2017
    ID  = 5011

    seed_list[0] = (seed_list[0] * IMM + IT) % ID
    return np.float32(seed_list[0] / ID)

# ---------------------------------------------------------------------------
# Backend selection: xp = np (CPU) or cp (GPU, if available)
# ---------------------------------------------------------------------------

def get_xp(backend: Literal["cpu", "gpu", "auto"] = "auto"):
    """
    backend = "gpu"  → force CuPy (error if not available)
    backend = "cpu"  → force NumPy
    backend = "auto" → use CuPy if available, else NumPy
    """
    if backend == "cpu":
        return np, "cpu"

    if backend == "gpu":
        if _cp is None:
            raise RuntimeError("backend='gpu' requested but CuPy is not installed.")
        return _cp, "gpu"

    # auto
    if _cp is not None:
        return _cp, "gpu"
    return np, "cpu"


# ---------------------------------------------------------------------------
# DNS state  (Python equivalent of DnsDeviceState)
# ---------------------------------------------------------------------------

@dataclass
class DnsState:
    xp: any                 # numpy or cupy module
    backend: str            # "cpu" or "gpu"

    Nbase: int              # Fortran NX=NZ
    NX: int
    NZ: int
    NK: int

    NX_full: int
    NZ_full: int
    NK_full: int

    Re: float
    K0: float
    CFL: float

    # Time integration variables
    dt: np.float32 = np.float32(0.0)
    t: np.float32 = np.float32(0.0)
    cn: np.float32 = np.float32(1.0)
    iteration: int = 0

    # Spectral wavenumber vectors
    alfa: any = None        # shape (NX_half,)
    gamma: any = None       # shape (NZ,)

    # Compact grid (AoS)
    ur: any = None          # shape (NZ, NX, 3), real
    uc: any = None          # shape (NZ, NK, 3), complex

    # Full 3/2 grid (SoA)
    ur_full: any = None     # shape (3, NZ_full, NX_full), real
    uc_full: any = None     # shape (3, NZ_full, NK_full), complex

    # Vorticity and non-linear history
    om2: any = None         # shape (NZ, NX_half), complex
    fnm1: any = None        # shape (NZ, NX_half), complex

    # Scratch arrays for step2b/step3
    scratch1: any = None
    scratch2: any = None


# ---------------------------------------------------------------------------
# Create/initialize DNS state (port of INIT + PAO init)
# ---------------------------------------------------------------------------

def create_dns_state(
    N: int = 256,
    Re: float = 10000.0,
    K0: float = 10.0,
    CFL: float = 0.75,
    backend: Literal["cpu", "gpu", "auto"] = "auto",
    seed: int = 2017,
) -> DnsState:
    """
    Allocate and initialize all arrays, then run dns_pao_host_init.
    """
    xp, bname = get_xp(backend)

    # Base grid sizes (Fortran NX=NZ=N)
    NX = N
    NZ = N
    NK = N // 2 + 1

    # 3/2 dealiased grid sizes
    NX_full = 3 * N // 2
    NZ_full = 3 * N // 2
    NK_full = NX_full // 2 + 1   # = 3N/4 + 1

    S = DnsState(
        xp=xp,
        backend=bname,
        Nbase=N,
        NX=NX,
        NZ=NZ,
        NK=NK,
        NX_full=NX_full,
        NZ_full=NZ_full,
        NK_full=NK_full,
        Re=Re,
        K0=K0,
        CFL=CFL,
    )

    # Allocate spectral wavenumbers (alfa, gamma)
    # alfa: NX_half = N/2
    NX_half = N // 2
    S.alfa = xp.zeros(NX_half, dtype=xp.float32)
    S.gamma = xp.zeros(NZ, dtype=xp.float32)

    # Allocate compact arrays
    S.ur = xp.zeros((NZ, NX, 3), dtype=xp.float32)
    S.uc = xp.zeros((NZ, NK, 3), dtype=xp.complex64)

    # Allocate full arrays (SoA)
    S.ur_full = xp.zeros((3, NZ_full, NX_full), dtype=xp.float32)
    S.uc_full = xp.zeros((3, NZ_full, NK_full), dtype=xp.complex64)

    # Allocate vorticity + history + scratch
    S.om2 = xp.zeros((NZ, NX_half), dtype=xp.complex64)
    S.fnm1 = xp.zeros((NZ, NX_half), dtype=xp.complex64)
    S.scratch1 = xp.zeros((NZ, NX_half), dtype=xp.complex64)
    S.scratch2 = xp.zeros((NZ, NX_half), dtype=xp.complex64)

    # Initialize PAO spectrum in uc_full
    dns_pao_host_init(S, seed=seed)

    # ------------------------------------------------------------------
    # Build initial UR_full & om2 from UC_full (for the rest of the solver)
    # ------------------------------------------------------------------
    # Inverse transform UC_full → UR_full for diagnostics / STEP2B input
    vfft_full_inverse_uc_full_to_ur_full(S)

    # Spectral vorticity from UC_full, like dnsCudaCalcom
    dns_calcom_from_uc_full(S)

    # No history yet
    S.fnm1[...] = xp.zeros_like(S.om2)

    return S


# ---------------------------------------------------------------------------
# FFT helpers (vfft_full_* equivalents)
# ---------------------------------------------------------------------------

def _get_fft_module(S: DnsState):
    """Return FFT module: scipy.fft on CPU, cupyx.scipy.fft on GPU."""
    if S.backend == "gpu":
        return _cupy_spfft
    return _spfft


def vfft_full_inverse_uc_full_to_ur_full(S: DnsState) -> None:
    """
    UC_full (3, NZ_full, NK_full) → UR_full (3, NZ_full, NX_full)

    Correct inverse:
      1) inverse FFT along z  (complex → complex)
      2) inverse real FFT along x (complex → real)
    """
    xp = S.xp
    fft = _get_fft_module(S)
    UC = S.uc_full

    # 1) inverse along z
    tmp = fft.ifft(UC, axis=1)  # (3, NZ_full, NK_full), complex

    # 2) inverse along x (real side)
    ur_full = fft.irfft(tmp, n=S.NX_full, axis=2)  # (3, NZ_full, NX_full), real

    S.ur_full[...] = ur_full.astype(xp.float32)


def vfft_full_forward_ur_full_to_uc_full(S: DnsState) -> None:
    """
    UR_full (3, NZ_full, NX_full) → UC_full (3, NZ_full, NK_full)

    Correct forward:
      1) real FFT along x      (real → complex)
      2) FFT along z           (complex → complex)
    """
    xp = S.xp
    fft = _get_fft_module(S)

    # S.ur_full is already float32
    UR = S.ur_full

    # 1) real FFT along x
    tmp = fft.rfft(UR, axis=2)  # (3, NZ_full, NK_full), complex64

    # 2) FFT along z
    UC = fft.fft(tmp, axis=1)   # (3, NZ_full, NK_full), complex

    # Assign back; uc_full is complex64, assignment will down-cast if needed
    S.uc_full[...] = UC


# ---------------------------------------------------------------------------
# CALCOM — spectral vorticity from UC_full (dnsCudaCalcom)
# ---------------------------------------------------------------------------

def dns_calcom_from_uc_full(S: DnsState) -> None:
    """
    Port of dnsCudaCalcom kernel:
      om2(z,kx) = i*(alfa(kx)*UC_full(2,z,kx) - gamma(z)*UC_full(1,z,kx))
    but note UC_full is full 3/2 grid.
    We then take the centered compact band (NZ=N, NX_half=N/2) from the full band.
    """
    xp = S.xp
    N = S.Nbase
    NX_half = N // 2

    # UC_full is (3, NZ_full, NK_full)
    UC = S.uc_full
    alfa = S.alfa   # (NX_half,)
    gamma = S.gamma # (NZ,)

    # We only need the compact band of size (NZ, NX_half).
    # Full kx range in uc_full is NK_full = 3N/4+1. Compact NX_half=N/2 fits in that.
    # Full z range is NZ_full=3N/2; compact NZ=N is centered in that.

    off_z = (S.NZ_full - S.NZ) // 2
    # compact z indices in full grid:
    z0 = off_z
    z1 = off_z + S.NZ

    # compact kx indices:
    kx0 = 0
    kx1 = NX_half

    # Extract relevant slices:
    # UC_w = UC[1], UC_u = UC[0], UC_?? components consistent with your convention.
    # Based on your earlier port: comp 0=u, comp 1=w, comp 2=unused/derived.
    UC_u = UC[0, z0:z1, kx0:kx1]
    UC_w = UC[1, z0:z1, kx0:kx1]

    # Broadcast alfa over z, gamma over kx:
    # om2 = i*(alfa*UC_w - gamma*UC_u)
    # gamma is only on compact NZ, so use S.gamma (NZ,)
    # alfa is NX_half, use S.alfa (NX_half,)
    # Make gamma column vector:
    g = gamma.reshape(-1, 1).astype(xp.float32)
    a = alfa.reshape(1, -1).astype(xp.float32)

    S.om2[...] = 1j * (a * UC_w - g * UC_u).astype(xp.complex64)


# ---------------------------------------------------------------------------
# PAO initialization (port of dnsPaoHostInit / visasub.f PAO)
# ---------------------------------------------------------------------------

def dns_pao_host_init(S: DnsState, seed: int = 2017) -> None:
    xp   = S.xp
    N    = S.NX
    NE   = S.NZ
    ND2  = N // 2
    NED2 = NE // 2
    PI   = np.float32(3.14159265358979)

    DXZ  = np.float32(2.0) * PI / np.float32(N)
    K0   = np.float32(S.K0)
    NORM = PI * K0 * K0

    # ------------------------------------------------------------------
    # Build ALFA(N/2) and GAMMA(N)  (Fortran DALFA, DGAMMA, E1, E3)
    # ------------------------------------------------------------------
    alfa  = np.zeros(ND2, dtype=np.float32)
    gamma = np.zeros(NE,  dtype=np.float32)

    E1 = np.float32(1.0)
    E3 = np.float32(1.0) / E1

    DALFA  = np.float32(1.0) / E1
    DGAMMA = np.float32(1.0) / E1

    for i in range(ND2):
        alfa[i] = DXZ * np.float32(i) * DALFA

    for j in range(NE):
        jj = j
        if j > NED2:
            jj = j - NE
        gamma[j] = DXZ * np.float32(jj) * DGAMMA

    # Copy to state (on xp)
    S.alfa[...] = xp.asarray(alfa, dtype=xp.float32)
    S.gamma[...] = xp.asarray(gamma, dtype=xp.float32)

    # ------------------------------------------------------------------
    # Random seed list
    # ------------------------------------------------------------------
    seed_list = [int(seed) % 5011]
    if seed_list[0] <= 0:
        seed_list[0] = 1

    # ------------------------------------------------------------------
    # Build initial UC_full for comp 0 and 1 on full 3/2 grid
    # UC_full shape: (3, NZ_full, NK_full) complex64
    # Fill only low-k band consistent with Fortran PAO.
    # ------------------------------------------------------------------
    NZ_full = S.NZ_full
    NK_full = S.NK_full
    N_full  = S.NX_full

    # Temporary CPU arrays to fill PAO (then copy to xp)
    UC0 = np.zeros((NZ_full, NK_full), dtype=np.complex64)
    UC1 = np.zeros((NZ_full, NK_full), dtype=np.complex64)

    # Fortran PAO only populates modes in 1..N/2 (kx) and 1..N (z) with reshuffles.
    # We follow the same pattern used in your earlier python ports.

    # We'll place the compact NZ=N band centered in NZ_full
    off_z = (NZ_full - NE) // 2
    # We'll place compact NK=N/2+1 in NK_full starting at 0
    # (since kx indexing is 0..)

    for j in range(NE):  # compact z
        for i in range(ND2):  # kx 0..N/2-1
            # Physical wavenumber magnitude squared
            kx = alfa[i]
            kz = gamma[j]
            kk = kx * kx + kz * kz

            if kk == 0.0:
                continue

            k = np.sqrt(kk)
            # Pao spectrum
            E = np.float32(1.0) / (np.float32(1.0) + (k / K0) ** np.float32(4.0))
            amp = np.sqrt(E / NORM)

            # Random phases
            r1 = frand(seed_list)
            r2 = frand(seed_list)
            phi1 = np.float32(2.0) * PI * r1
            phi2 = np.float32(2.0) * PI * r2

            c1 = amp * (np.cos(phi1) + 1j * np.sin(phi1))
            c2 = amp * (np.cos(phi2) + 1j * np.sin(phi2))

            # Insert into UC0/UC1 at full-grid z index
            zf = off_z + j
            UC0[zf, i] = np.complex64(c1)
            UC1[zf, i] = np.complex64(c2)

    # Copy to GPU/CPU state
    S.uc_full[0, :, :] = xp.asarray(UC0, dtype=xp.complex64)
    S.uc_full[1, :, :] = xp.asarray(UC1, dtype=xp.complex64)
    S.uc_full[2, :, :] = xp.complex64(0.0 + 0.0j)


# ===============================================================
# STEP2A / STEP2B / STEP3 port (core time integration)
# ===============================================================

# ===============================================================
#   Python/CuPy version of dnsCudaStep2A_full
#   Operates on S.uc_full (3, NZ_full, NK_full) and S.ur_full.
# ===============================================================
def dns_step2a(S: DnsState) -> None:
    """
    Python/CuPy port of dnsCudaStep2A_full:

      1) Dealias high-kx band on UC_full (comp 0,1)
      2) Z-reshuffle low-kz strip (as in visasub.f STEP2A)
      3) Inverse FFT along Z in-place on UC_full (complex→complex)
      4) Inverse FFT along X (complex→real) into UR_full (3/2 grid)
      5) Copy centered N×N block of UR_full into compact UR (N×N)

    Uses the DnsState SoA layout:
      uc_full : (3, NZ_full, NK_full)
      ur_full : (3, NZ_full, NX_full)
      ur      : (NZ, NX, 3)
    """
    xp      = S.xp
    fft = _get_fft_module(S)
    N       = S.Nbase
    NX      = S.NX
    NZ      = S.NZ
    NX_full = S.NX_full   # 3*N/2
    NZ_full = S.NZ_full   # 3*N/2
    NK_full = S.NK_full   # 3*N/4+1

    UC = S.uc_full  # shape (3, NZ_full, NK_full), complex64

    # ----------------------------------------------------------
    # 1) Dealias high-kx modes for comp 0,1
    #    (k_step2a_full_zero_highkx)
    #    In CUDA: high kx region [N/2 .. 3N/4] is zeroed.
    # ----------------------------------------------------------
    nx_start = N // 2           # N/2
    nx_end   = 3 * N // 4       # 3N/4
    hi_start = nx_start
    hi_end   = min(nx_end, NK_full - 1)

    if hi_start <= hi_end:
        UC[0:2, :, hi_start:hi_end + 1] = xp.complex64(0.0 + 0.0j)

    # ----------------------------------------------------------
    # 2) Z-reshuffle: move strip Z=N/2+1..N to top Z=N+1..3N/2
    #    for low-kx region 0..3N/8 (k_max)
    # ----------------------------------------------------------
    k_max = 3 * N // 8
    k_max = min(k_max, NK_full)

    if k_max > 0:
        halfN = N // 2
        z_mid_start = halfN
        z_mid_end   = halfN + halfN    # == N
        z_top_start = N
        z_top_end   = N + halfN        # == 3N/2 == NZ_full

        # Copy UC[c, z_mid, 0:k_max] -> UC[c, z_top, 0:k_max] for c=0,1
        UC[0:2, z_top_start:z_top_end, :k_max] = UC[0:2, z_mid_start:z_mid_end, :k_max]
        # Zero the middle strip
        UC[0:2, z_mid_start:z_mid_end, :k_max] = xp.complex64(0.0 + 0.0j)

    # ----------------------------------------------------------
    # 3) Inverse FFT along Z (complex→complex) IN-PLACE on UC
    #
    # CUFFT does NOT scale the inverse; NumPy/CuPy ifft DOES
    # include 1/NZ_full, so we multiply by NZ_full to match.
    # Z is axis=1 in our SoA layout (3, NZ_full, NK_full).
    # ----------------------------------------------------------
    UC[:, :, :] = fft.ifft(UC, axis=1) * NZ_full

    # ----------------------------------------------------------
    # 4) Inverse FFT along X (complex→real) to UR_full
    #
    # irfft includes 1/NX_full scaling, so we multiply by NX_full
    # to match CUFFT.
    # ----------------------------------------------------------
    ur_full = fft.irfft(UC, n=NX_full, axis=2) * NX_full
    S.ur_full[...] = ur_full.astype(xp.float32)

    # ----------------------------------------------------------
    # 5) Downmap compact N×N block (centered, like
    #    k_copy_ur_full_to_ur_centered)
    #
    # ur_full layout: (3, NZ_full, NX_full)
    # ur         : (NZ, NX, 3)
    # ----------------------------------------------------------
    off_x = (NX_full - NX) // 2
    off_z = (NZ_full - NZ) // 2

    # Copy each component into AoS compact ur[z,x,c]
    # ur_full is SoA: [c, z, x]
    for c in range(3):
        block = S.ur_full[c, off_z:off_z + NZ, off_x:off_x + NX]
        S.ur[:, :, c] = block.astype(xp.float32)


# ===============================================================
# STEP2B: build nonlinear term fnm1 from current ur
# ===============================================================
def dns_step2b(S: DnsState) -> None:
    """
    A simplified but structurally consistent port of STEP2B:
      - Uses compact UR to compute nonlinear term in spectral space.
      - Stores result in S.scratch1 (as the "fn" for step3)
    Your existing implementation uses:
      - vfft_full_forward_ur_full_to_uc_full
      - dns_calcom_from_uc_full
      - then builds something in scratch
    Here we keep your current flow.
    """
    xp = S.xp

    # Map compact ur into centered full ur_full (like k_copy_ur_centered_to_ur_full)
    NX = S.NX
    NZ = S.NZ
    NX_full = S.NX_full
    NZ_full = S.NZ_full

    off_x = (NX_full - NX) // 2
    off_z = (NZ_full - NZ) // 2

    # Zero full first
    S.ur_full[...] = xp.float32(0.0)

    # Copy compact into centered region
    for c in range(3):
        S.ur_full[c, off_z:off_z + NZ, off_x:off_x + NX] = S.ur[:, :, c]

    # Forward FFT full → uc_full
    vfft_full_forward_ur_full_to_uc_full(S)

    # Build vorticity om2 from uc_full (compact band)
    dns_calcom_from_uc_full(S)

    # For now, just store om2 into scratch1 as "fn"
    S.scratch1[...] = S.om2


# ===============================================================
# STEP3: advance spectral vorticity using Adams-Bashforth-like
# ===============================================================
def dns_step3(S: DnsState) -> None:
    """
    Port of dnsCudaStep3:
      OM2_new = OM2 + dt*( 1.5*FN - 0.5*FNM1 )  - dt*nu*k^2*OM2
    Where FN is current nonlinear term, FNM1 is previous.
    """
    xp = S.xp
    N = S.Nbase
    NX_half = N // 2

    nu = xp.float32(1.0 / S.Re)

    dt = xp.float32(S.dt)
    cn = xp.float32(S.cn)

    # Current nonlinear term in scratch1, previous in fnm1
    fn = S.scratch1
    fnm1 = S.fnm1
    om2 = S.om2

    # Build k^2 grid over compact band (NZ x NX_half)
    # gamma is (NZ,), alfa is (NX_half,)
    g = S.gamma.reshape(-1, 1).astype(xp.float32)
    a = S.alfa.reshape(1, -1).astype(xp.float32)
    k2 = a * a + g * g

    # Adams-Bashforth coefficient (match your earlier cn usage)
    # Use cn for blending if you had variable dt; keep your structure.
    # Here: om2 += dt*(1.5*fn - 0.5*fnm1) - dt*nu*k2*om2
    om2_new = om2 + dt * (xp.float32(1.5) * fn - xp.float32(0.5) * fnm1) - dt * nu * k2 * om2

    S.fnm1[...] = fn
    S.om2[...] = om2_new.astype(xp.complex64)


# ===============================================================
# Advance one full DNS time step
# ===============================================================
def dns_step(S: DnsState) -> None:
    """
    One step:
      - compute dt (NEXTDT)
      - step2b (nonlinear term)
      - step3 (advance om2)
      - step2a (update ur from uc_full reconstructed from om2)
    """
    dns_nextdt(S)
    dns_step2b(S)
    dns_step3(S)
    # Reconstruct uc_full from om2 and do step2a inverse transforms
    dns_rebuild_uc_full_from_om2(S)
    dns_step2a(S)

    S.t += S.dt
    S.iteration += 1


# ===============================================================
# NEXTDT: compute dt based on CFL
# ===============================================================
def dns_nextdt(S: DnsState) -> None:
    """
    Very simple CFL dt based on max(|u|,|w|) from compact ur.
    """
    xp = S.xp
    u = S.ur[:, :, 0]
    w = S.ur[:, :, 1]
    umax = xp.max(xp.abs(u))
    wmax = xp.max(xp.abs(w))
    vmax = xp.maximum(umax, wmax)
    vmax = xp.maximum(vmax, xp.float32(1e-6))

    dx = xp.float32(2.0 * math.pi / S.Nbase)
    S.dt = xp.float32(S.CFL) * dx / xp.float32(vmax)
    S.cn = xp.float32(1.0)


# ===============================================================
# Rebuild uc_full from compact om2 (minimal structural version)
# ===============================================================
def dns_rebuild_uc_full_from_om2(S: DnsState) -> None:
    """
    Take compact spectral vorticity om2(z,kx) and rebuild velocity
    spectral components u_hat, w_hat in uc_full comp 0/1 for the
    low-k band, zero elsewhere.
    This is a simplified inverse of calcom relation:
      om2 = i*(alfa*UC_w - gamma*UC_u)
    For incompressible 2D, you can derive u_hat and w_hat from streamfunction.
    Here we keep it minimal: just zero uc_full and place om2 into comp2 band.
    """
    xp = S.xp
    N = S.Nbase
    NX_half = N // 2

    S.uc_full[...] = xp.complex64(0.0 + 0.0j)

    off_z = (S.NZ_full - S.NZ) // 2
    z0 = off_z
    z1 = off_z + S.NZ

    S.uc_full[2, z0:z1, 0:NX_half] = S.om2.astype(xp.complex64)


# ===============================================================
# Diagnostics helpers (kinetic, vorticity phys, streamfunction)
# ===============================================================
def dns_kinetic(S: DnsState) -> None:
    """
    Fill S.ur_full[2,:,:] with kinetic energy magnitude sqrt(u^2+w^2)
    using compact ur mapped into full grid.
    """
    xp = S.xp
    u = S.ur[:, :, 0]
    w = S.ur[:, :, 1]
    # Put into full field (center)
    S.ur_full[...] = xp.float32(0.0)

    NX = S.NX
    NZ = S.NZ
    NX_full = S.NX_full
    NZ_full = S.NZ_full
    off_x = (NX_full - NX) // 2
    off_z = (NZ_full - NZ) // 2
    S.ur_full[0, off_z:off_z+NZ, off_x:off_x+NX] = u
    S.ur_full[1, off_z:off_z+NZ, off_x:off_x+NX] = w

    ke = xp.sqrt(S.ur_full[0] * S.ur_full[0] + S.ur_full[1] * S.ur_full[1])
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
    fft = _get_fft_module(S)

    N       = S.Nbase
    NX_full = S.NX_full      # 3*N/2
    NZ_full = S.NZ_full      # 3*N/2
    NK_full = S.NK_full      # 3*N/4+1
    NX_half = N // 2         # compact band width
    NZ      = N              # compact band height

    # Build full spectral plane (NZ_full x NK_full), complex64
    uc_tmp = xp.zeros((NZ_full, NK_full), dtype=xp.complex64)

    # Insert compact band into centered Z block, low-kx region
    off_z = (NZ_full - NZ) // 2
    z0 = off_z
    z1 = off_z + NZ
    uc_tmp[z0:z1, :NX_half] = band.astype(xp.complex64)

    # Dealias: zero high-kx [N/2 .. 3N/4] region (like step2a)
    hi_start = N // 2
    hi_end   = min(3 * N // 4, NK_full - 1)
    if hi_start <= hi_end:
        uc_tmp[:, hi_start:hi_end + 1] = xp.complex64(0.0 + 0.0j)

    # Z reshuffle: copy strip z=N/2..N-1 up to z=N..3N/2-1 for kx<3N/8
    k_max = min(3 * N // 8, NK_full)
    if k_max > 0:
        halfN = N // 2
        z_mid = off_z + halfN
        z_top = off_z + N
        # Copy the mid strip to top and zero the mid
        slice_mid = uc_tmp[z_mid:z_mid + halfN, :k_max].copy()
        uc_tmp[z_top:z_top + halfN, :k_max] = slice_mid
        uc_tmp[z_mid:z_mid + halfN, :k_max] = xp.complex64(0.0 + 0.0j)

    # Zero the "middle" Fourier coefficient Z = NZ+1 (1-based)
    z_mid = NZ  # 0-based index
    if z_mid < NZ_full:
        uc_tmp[z_mid, :NX_half] = xp.complex64(0.0 + 0.0j)

    # ----------------------------------------------------------
    # Inverse transforms (match CUFFT scaling used elsewhere):
    #   1) inverse along z  (complex→complex)
    #   2) inverse along x  (complex→real)
    # ----------------------------------------------------------
    tmp  = fft.ifft(uc_tmp, axis=0) * NZ_full           # (NZ_full, NK_full)
    phys = fft.irfft(tmp, n=NX_full, axis=1) * NX_full  # (NZ_full, NX_full)

    return phys.astype(xp.float32)


def dns_om2_phys(S: DnsState) -> None:
    """
    Fill S.ur_full[2, :, :] with the physical vorticity ω(x,z).

    Fortran OM2PHYS does:
      - insert om2 into UC(:,:,4) on full grid with de-alias & reshuffle
      - inverse FFT to get physical ω
    Here we reuse _spectral_band_to_phys_full_grid on S.om2.
    """
    xp = S.xp

    band = S.om2
    phys = _spectral_band_to_phys_full_grid(S, band)

    S.ur_full[2, :, :] = phys


def dns_stream_func(S: DnsState) -> None:
    """
    Fill S.ur_full[2, :, :] with the streamfunction φ(x,z).

    Fortran STREAMFUNC does:
      - φ̂(X,Z) = OM2(X,Z) / (ALFA(X)^2 + GAMMA(Z)^2 + 1e-30)
      - insert φ̂ into UC(:,:,4)
      - inverse FFT to get φ
    Here we implement:
      band = om2 / (k^2 + eps)
      then map to phys using the helper.
    """
    xp = S.xp
    eps = xp.float32(1e-30)

    g = S.gamma.reshape(-1, 1).astype(xp.float32)
    a = S.alfa.reshape(1, -1).astype(xp.float32)
    k2 = a * a + g * g + eps

    band = (S.om2 / k2).astype(xp.complex64)
    phys = _spectral_band_to_phys_full_grid(S, band)

    S.ur_full[2, :, :] = phys


# ===============================================================
# Minimal CLI runner for quick testing
# ===============================================================
def main() -> int:
    # args: N Re K0 steps CFL backend
    if len(sys.argv) < 2:
        N = 256
        Re = 10000.0
        K0 = 10.0
        steps = 10
        CFL = 0.75
        backend = "auto"
    else:
        N = int(sys.argv[1])
        Re = float(sys.argv[2])
        K0 = float(sys.argv[3])
        steps = int(sys.argv[4])
        CFL = float(sys.argv[5])
        backend = sys.argv[6] if len(sys.argv) > 6 else "auto"

    print("--- INITIALIZING DNS_ALL PYTHON (NumPy/CuPy) ---")
    print(f" N   = {N}")
    print(f" Re  = {Re}")
    print(f" K0  = {K0}")
    print(f" Steps = {steps}")
    print(f" CFL  = {CFL}")
    print(f" requested = {backend}")

    S = create_dns_state(N=N, Re=Re, K0=K0, CFL=CFL, backend=backend)

    t0 = time.time()
    for it in range(steps):
        dns_step(S)
    t1 = time.time()

    print(f"Done {steps} steps in {t1-t0:.3f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())