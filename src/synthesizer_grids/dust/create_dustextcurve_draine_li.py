#!/usr/bin/env python3
"""
Create dust extinction curves from Draine optical properties.

Parse Draine Sil_81.gz, Gra_81.gz, PAHion_30.gz, PAHneu_30.gz files,
construct grain-size distributions (either MRN or the custom lognormal form)
and compute extinction cross-sections (sigma).

Currently this is done only for 2 binsizes (small and large) for silicate
and graphite, and 1 bin for PAHs (ionized and neutral).
"""

import io
import gzip
import math
import argparse
from pathlib import Path
from typing import List, Tuple
from numpy.typing import NDArray
from datetime import date

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import requests
import h5py

from synthesizer.emission_models.attenuation import Calzetti2000, GrainsWD01
from unyt import Angstrom, um


def download_bytes(url: str, timeout: float = 30.0) -> bytes:
    """
    Download bytes from URL with timeout.
    Returns content as bytes.

    Args:
        url (string)
            URL to download
        timeout (float)
            Timeout in seconds (default: 30.0)
    Returns:
        bytes
    """

    print(f"Downloading {url} ...")
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content


def read_gz_lines(gzbytes: bytes, savename: str) -> List[str]:
    """
    Read lines from gzipped bytes, save to savename.
    Returns list of lines as strings.

    Args:
        gzbytes (bytes)
            Gzipped content
        savename (string)
            Filename to save uncompressed text
    Returns:
        string with line breaks
    """
    with gzip.GzipFile(fileobj=io.BytesIO(gzbytes)) as gf:
        txt = gf.read().decode("utf-8", errors="ignore")
    with open(savename, "w") as file:
        file.writelines(txt)
    return txt


def parse_draine_file_lines(
    filename: str,
) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """
    Parse Draine optical property file (e.g., Sil_81.txt) and extract
    radius grid (micron), wavelength grid (micron), and Qext, Qabs, Qsca arrays.

    Args:
        filename (string)
            path to Draine optical property file
    Returns:
        a_grid (NDArray)
            radius grid (micron)
        wav_grid (NDArray)
            wavelength grid (micron)
        Qext (NDArray)
            extinction efficiency array, shape (len(wav_grid), len(a_grid))
        Qabs (NDArray)
            absorption efficiency array, shape (len(wav_grid), len(a_grid))
        Qsca (NDArray)
            scattering efficiency array, shape (len(wav_grid), len(a_grid))
    """

    NRAD = None
    NWAV = None
    a_min = None
    a_max = None
    line_num_NWAV = None

    with open(filename, "r") as file:
        lines = file.read().splitlines()

    # Find NRAD, NWAV, a_min, a_max in early header lines
    for ii, L in enumerate(lines[:50]):
        if "NRAD" in L:
            left = L.split("=")[0].strip()
            toks = left.split()
            if len(toks) >= 1:
                try:
                    NRAD = int(toks[0])
                    a_min = float(toks[1])
                    a_max = float(toks[2])
                except Exception:
                    pass
        if "NWAV" in L:
            left = L.split("=")[0].strip()
            toks = left.split()
            if len(toks) >= 1:
                try:
                    NWAV = int(toks[0])
                    line_num_NWAV = ii
                except Exception:
                    pass
    if NRAD is None or NWAV is None:
        raise RuntimeError(
            "Failed to find NRAD (number of radii) or NWAV (number of wavelength) in header. Inspect file header manually, and retry."
        )

    print(f"NRAD = {NRAD}, NWAV = {NWAV}, amin ={a_min}, amax={a_max}")

    skip_ftr = 3  # lines to skip after NWAV line (usually 3)
    first_skip_hdr = (
        1 + line_num_NWAV + skip_ftr
    )  # start parsing after this line

    a_grid = 10 ** np.linspace(
        np.log10(a_min), np.log10(a_max), num=NRAD, endpoint=True
    )

    Qabs: NDArray = np.zeros((NRAD, NWAV))
    Qsca: NDArray = np.zeros((NRAD, NWAV))
    Qext: NDArray = np.zeros((NRAD, NWAV))
    # gcos: NDArray = np.zeros((NRAD, NWAV))

    for ii in range(NRAD):
        if "PAH" in filename:
            add = 1
            usecols = (0, 1, 2, 3)
        else:
            add = 0
            usecols = (0, 1, 2)

        tmp = np.genfromtxt(
            filename,
            skip_header=first_skip_hdr + ii * NWAV + ii * skip_ftr,
            max_rows=NWAV,
            dtype=float,
            usecols=usecols,
        )

        if ii == 0:
            wav_grid = tmp[:, 0]  # micron
        Qabs[ii] = tmp[:, 1 + add]
        Qsca[ii] = tmp[:, 2 + add]
        if "PAH" in filename:
            Qext[ii] = tmp[:, 1]
        else:
            Qext[ii] = tmp[:, 1 + add] + tmp[:, 2 + add]
        # gcos[ii] = tmp[:,3+add]

    return a_grid, wav_grid, Qext.T, Qabs.T, Qsca.T


def build_mrn_component(
    a_grid_micron: NDArray,
    a_min: float,
    a_max: float,
    power: float,
    mass_per_H: float,
    rho: float,
) -> NDArray:
    """
    Build a truncated MRN-like grain-size distribution component
    with dn/da propto a^power over [a_min, a_max] (micron).
    n(a) = C * a^(power+1), where C is a normalisation constant

    Args:
        a_grid_micron (NDArray)
            radius grid (micron) where n(a) is evaluated
        a_min (float)
            minimum grain radius (micron)
        a_max (float)
            maximum grain radius (micron)
        power (float)
            power-law exponent (dn/da ∝ a^power)
        mass_per_H (float)
            total mass per H for this component (g per H)
        rho (float)
            material density (g cm^-3)
    Returns:
        n(a) (NDArray)
            grain size distribution n(a) on a_grid_micron
            defined as number per Hydrogen nucleus in the range
            of grain radii between a and a+da.
    """
    # S = ∫ a^p da over [a_min, a_max] (a in micron)
    if abs(power + 1.0) < 1e-12:
        S = math.log(a_max / a_min)
    else:
        S = (a_max ** (power + 1.0) - a_min ** (power + 1.0)) / (power + 1.0)

    p4 = power + 4.0
    if abs(p4) < 1e-12:
        I_a3 = math.log(a_max / a_min)
    else:
        I_a3 = (a_max**p4 - a_min**p4) / p4

    a3_mean_micron3 = I_a3 / S  # <a^3> in micron^3
    micron_to_cm = (1.0 * um).to("cm").value  # 1e-4
    a3_mean_cm3 = a3_mean_micron3 * (micron_to_cm**3)
    mass_per_grain = (4.0 / 3.0) * math.pi * rho * a3_mean_cm3
    if mass_per_grain <= 0:
        raise ValueError("Non-positive mass per grain; check parameters.")
    N = mass_per_H / mass_per_grain  # number per H total for component
    # shape s(a) = a^p / S (units 1/micron)
    s = np.zeros_like(a_grid_micron)
    mask = (a_grid_micron >= a_min) & (a_grid_micron <= a_max)
    s[mask] = (a_grid_micron[mask] ** power) / S
    n_a = N * s
    return n_a


def build_lognormal_component(
    a_grid_micron: NDArray,
    a0_micron: float,
    sigma_ln: float,
    mass_per_H: float,
    rho: float,
) -> NDArray:
    """
    Build the lognormal-type component with the form
    n(a) = C / a^4 * exp( - (ln(a/a0))^2 / (2 sigma^2) ),
    where a is in micron and n(a) is number per H per micron.
    C is chosen so the component total mass per H = mass_per_H (g per H).
    Reference: Hirashita 2015 (https://arxiv.org/abs/1412.3866)
    See eq 31 in that paper.
    implies C = 3 * mass_per_H / (4 pi rho sqrt(2 pi) sigma_ln
    for the lmit 0 -> infinity (change of variable to x=ln(a/a0))

    Args:
        a_grid_micron (NDArray)
            radius grid (micron) where n(a) is evaluated
        a0_micron (float)
            center of lognormal (micron)
        sigma_ln (float)
            width of lognormal in ln-space
        mass_per_H (float)
            total mass per H for this component (g per H)
        rho (float)
            material density (g cm^-3)
    Returns:
        n(a) (NDArray)
            grain size distribution n(a) on a_grid_micron
            defined as number per Hydrogen nucleus in the range
            of grain radii between a and a+da.
    """
    if sigma_ln <= 0:
        raise ValueError("sigma_ln must be > 0")

    # compute C using analytic integral:
    micron_to_cm = (1.0 * um).to("cm").value  # 1e-4
    prefac = (4.0 / 3.0) * math.pi * rho * (micron_to_cm**3) * micron_to_cm  #
    denom = prefac * math.sqrt(2.0 * math.pi) * sigma_ln
    if denom <= 0:
        raise ValueError("Denominator for C non-positive.")
    C = mass_per_H / denom

    exponent = -0.5 * ((np.log(a_grid_micron / a0_micron)) / sigma_ln) ** 2
    n_a = (C / (a_grid_micron**4)) * np.exp(exponent)

    return n_a


def calculate_Alam_over_NH(
    wav_micron: NDArray,
    radii_micron: NDArray,
    Qext: NDArray,
    a_grid_micron: NDArray,
    n_a: NDArray,
) -> NDArray:
    """
    A(lam)/N_H = 2.5 ln(e) integral pi a^2 Q_ext(a,lam) n(a) da
    Usually the dust-to-gas ratio used to calculate Alam/N_H is
    calculated for a line of sight, hence the N_H can be cancelled
    out. If the dust-to-gas ratio is an average for the galaxy, then
    this gives the average Alam over N_H.

    Args:
        wav_micron (NDArray)
            wavelength grid (micron)
        radii_micron (NDArray)
            radius grid (micron) corresponding to Qext
        Qext (NDArray)
            extinction efficiency array, shape (len(wav_micron), len(radii_micron))
        a_grid_micron (NDArray)
            radius grid (micron) where n(a) is defined
        n_a (NDArray)
            grain size distribution n(a) on a_grid_micron
            defined as number per Hydrogen nucleus in the range
            of grain radii between a and a+da.
    Returns:
        A(lam)/N_H (NDArray)
            Attenuation curve A(lam)/N_H on wav_micron (units: mag cm^2)
    """
    micron_to_cm = (1.0 * um).to("cm").value  # 1e-4
    a_grid_cm = a_grid_micron * micron_to_cm
    # n_a is number per H per micron -> convert to number per H per cm: n_a_per_cm = n_a / micron_to_cm
    n_a_per_cm = n_a / micron_to_cm

    Nwav = wav_micron.size
    Alam_by_N_H = np.zeros(Nwav, dtype=float)
    for iw in range(Nwav):
        q = Qext[iw, :]
        f_lin = interp1d(radii_micron, q, bounds_error=False, fill_value=0)
        q_interp = f_lin(a_grid_micron)
        # integrand: pi a^2 Q * n(a) da  (a in cm, n in per cm, da in cm)
        integrand = math.pi * (a_grid_cm**2) * q_interp * n_a_per_cm
        Alam_by_N_H[iw] = (
            2.5 * np.log(np.e) * np.trapezoid(integrand, a_grid_cm)
        )

    return Alam_by_N_H


def interp_Q_to_grid(
    wav_orig: NDArray,
    radii_orig: NDArray,
    Q_orig: NDArray,
    wav_target: NDArray,
) -> NDArray:
    """Interpolate Qext (or Qabs, Qsca) from original wavelength grid to target wavelength grid.
    Args:
        wav_orig (NDArray)
            original wavelength grid (micron)
        radii_orig (NDArray)
            original radius grid (micron)
        Q_orig (NDArray)
            original Q array, shape (len(wav_orig), len(radii_orig))
        wav_target (NDArray)
            target wavelength grid (micron)
    Returns:
        Qw (NDArray)
            interpolated Q array on target wavelength grid, shape (len(wav_target), len(radii_orig))
    """
    Nr = radii_orig.size
    Qw = np.zeros((wav_target.size, Nr))
    for j in range(Nr):
        f = interp1d(
            wav_orig,
            Q_orig[:, j],
            kind="linear",
            bounds_error=False,
            fill_value=0,
        )
        Qw[:, j] = f(wav_target)

    return Qw


def plot_extinction_curve(
    wav_micron: NDArray,
    A_over_Av: NDArray,
    mode: str,
    DTG_sil: float,
    DTG_gra: float,
    DTG_PAH: float,
    out_prefix: str,
    savefig: bool = True,
    show: bool = False,
):
    """
    Plot extinction curve A(λ)/A(V).

    Args:
        wav_micron (NDArray)
            wavelength grid (micron)
        A_over_Av (NDArray)
            extinction curve A(λ)/A(V)
        mode (str)
            size-distribution mode ("mrn" or "lognormal")
        DTG_sil (float)
            dust-to-gas mass ratio for silicate
        DTG_gra (float)
            dust-to-gas mass ratio for graphite
        DTG_PAH (float)
            dust-to-gas mass ratio for PAH
        out_prefix (str)
            output file prefix for saving plot
        savefig (bool)
            whether to save the figure (default: True)
        show (bool)
            whether to show the figure (default: False)
    """
    plt.figure(figsize=(8, 5))
    wav_angstrom = wav_micron * 1e4
    plt.plot(wav_angstrom, A_over_Av, label="Attenuation curve")
    curve = Calzetti2000(ampl=10.0)
    plt.plot(
        wav_angstrom,
        curve.get_tau(wav_angstrom * Angstrom),
        ls="--",
        label="Calzetti+2000",
    )
    for c in ["MW", "LMC", "SMC"]:
        curve_wd01 = GrainsWD01(model=c)
        plt.plot(
            wav_angstrom,
            curve_wd01.get_tau(wav_angstrom * Angstrom),
            ls=":",
            label=f"{c}",
        )
    curve_wd01 = GrainsWD01(model="MW")
    plt.xlabel("Wavelength (AA)")
    plt.ylabel("A(λ) / A(V)")
    plt.title(
        f"Extinction ({mode}) — sil DTG={DTG_sil:.3g}, gra DTG={DTG_gra:.3g}, PAH DTG={DTG_PAH:.3g}"
    )
    plt.xlim(800, 1e4)
    plt.xticks(np.arange(1000, 10001, 1000))
    # plt.ylim(1e-3, max(5.0, np.max(A_over_Av)*2.0))
    plt.grid(which="both", ls="--", alpha=0.4)
    plt.legend(fontsize="small", ncol=2)
    plt.tight_layout()
    if savefig:
        png_name = f"{out_prefix}_{mode}.png"
        plt.savefig(png_name, dpi=200)
        print(f"Saved plot: {png_name}")
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build extinction curves from Draine optical files with MRN or lognormal-size components."
    )
    parser.add_argument(
        "--mode",
        choices=["mrn", "lognormal"],
        default="lognormal",
        help="size-distribution mode",
    )
    parser.add_argument(
        "--grid-loc",
        type=str,
        default=".",
        help="directory to save the extinction curve grid",
    )
    parser.add_argument(
        "--grid-name",
        type=str,
        default="dust_extcurve_draine_li",
        help="name of the extinction curve grid file (without extension)",
    )
    parser.add_argument(
        "--dtg-min",
        type=float,
        default=1e-10,
        help="minimum of dust-to-gas mass ratio used for curve calculation",
    )
    parser.add_argument(
        "--dtg-max",
        type=float,
        default=1e-1,
        help="maximum of dust-to-gas mass ratio used for curve calculation",
    )
    parser.add_argument(
        "--n-dtg",
        type=int,
        default=1000,
        help="number of dust-to-gas mass ratio points between min and max",
    )
    parser.add_argument(
        "--small-centre",
        type=float,
        default=0.01,
        help="centre of small bin (micron)",
    )
    parser.add_argument(
        "--large-centre",
        type=float,
        default=0.1,
        help="centre of large bin (micron)",
    )
    parser.add_argument(
        "--pah-centre",
        type=float,
        default=0.005,
        help="centre of PAH bin (micron)",
    )
    parser.add_argument(
        "--sigma-lognormal",
        type=float,
        default=0.75,
        help="lognormal sigma_ln (width in ln-space)",
    )
    # Needed if mode is MRN
    parser.add_argument(
        "--a-min-mrn",
        type=float,
        default=0.005,
        help="MRN a_min (micron) when using MRN",
    )
    parser.add_argument(
        "--a-max-mrn",
        type=float,
        default=10.0,
        help="MRN a_max (micron) when using MRN",
    )
    parser.add_argument(
        "--a-min-mrn_pah",
        type=float,
        default=0.0001,
        help="MRN a_min (micron) when using MRN",
    )
    parser.add_argument(
        "--a-max-mrn_pah",
        type=float,
        default=0.01,
        help="MRN a_max (micron) when using MRN",
    )
    parser.add_argument(
        "--power-mrn",
        type=float,
        default=-3.5,
        help="MRN power-law exponent (dn/da ∝ a^power)",
    )
    parser.add_argument(
        "--plot_example",
        action="store_true",
        help="plot example extinction curves",
    )

    args = parser.parse_args()

    # Download URLs for Draine optical property files
    # We use the high-resolution versions (81 radii for
    # silicate/graphite, 30 for PAH)
    SIL_URL = "https://www.astro.princeton.edu/~draine/dust/diel/Sil_81.gz"
    saveSIL = "./Sil_81.txt"  # save
    GRA_URL = "https://www.astro.princeton.edu/~draine/dust/diel/Gra_81.gz"
    saveGRA = "./Gra_81.txt"  # save
    PAH_ion = "https://www.astro.princeton.edu/~draine/dust/diel/PAHion_30.gz"
    savePAHion = "./PAHion_30.txt"  # save
    PAH_neu = "https://www.astro.princeton.edu/~draine/dust/diel/PAHneu_30.gz"
    savePAHneu = "./PAHneu_30.txt"  # save

    # Define dust-to-gas ratio range
    DTG_MIN = args.dtg_min
    DTG_MAX = args.dtg_max
    N_DTG = args.n_dtg
    dtg_grid = np.logspace(
        math.log10(DTG_MIN), math.log10(DTG_MAX), N_DTG, endpoint=True
    )

    # Material densities (g cm^-3)
    RHO_SIL = 3.5
    RHO_GRA = 2.24
    RHO_PAH = 2.24

    # Gas mass per H (g) with helium correction
    MU = 1.4
    M_H = 1.6738e-24  # in grams
    GAS_MASS_PER_H = MU * M_H  # g per H

    # Integration grid for grain size in micron
    A_INT_MIN = 1e-4
    A_INT_MAX = 10.0
    N_A_INT = 500
    a_grid_int = np.logspace(
        math.log10(A_INT_MIN), math.log10(A_INT_MAX), N_A_INT, endpoint=True
    )

    # V wavelength for A(V)
    V_WAVELENGTH = 0.55  # micron

    # Download and process files
    if Path(saveSIL).exists():
        radii_sil, wav_sil, Qext_sil, Qabs_sil, Qsca_sil = (
            parse_draine_file_lines(saveSIL)
        )
    else:
        sil_bytes = download_bytes(SIL_URL)
        sil_lines = read_gz_lines(sil_bytes, savename=saveSIL)
        radii_sil, wav_sil, Qext_sil, Qabs_sil, Qsca_sil = (
            parse_draine_file_lines(saveSIL)
        )

    if Path(saveGRA).exists():
        radii_gra, wav_gra, Qext_gra, Qabs_gra, Qsca_gra = (
            parse_draine_file_lines(saveGRA)
        )
    else:
        gra_bytes = download_bytes(GRA_URL)
        gra_lines = read_gz_lines(gra_bytes, savename=saveGRA)
        radii_gra, wav_gra, Qext_gra, Qabs_gra, Qsca_gra = (
            parse_draine_file_lines(saveGRA)
        )

    if Path(savePAHion).exists():
        (
            radii_pah_ion,
            wav_pah_ion,
            Qext_pah_ion,
            Qabs_pah_ion,
            Qsca_pah_ion,
        ) = parse_draine_file_lines(savePAHion)
    else:
        pah_ion_bytes = download_bytes(PAH_ion)
        pah_ion_lines = read_gz_lines(pah_ion_bytes, savename=savePAHion)
        (
            radii_pah_ion,
            wav_pah_ion,
            Qext_pah_ion,
            Qabs_pah_ion,
            Qsca_pah_ion,
        ) = parse_draine_file_lines(savePAHion)

    if Path(savePAHneu).exists():
        (
            radii_pah_neu,
            wav_pah_neu,
            Qext_pah_neu,
            Qabs_pah_neu,
            Qsca_pah_neu,
        ) = parse_draine_file_lines(savePAHneu)
    else:
        pah_neu_bytes = download_bytes(PAH_neu)
        pah_neu_lines = read_gz_lines(pah_neu_bytes, savename=savePAHneu)
        (
            radii_pah_neu,
            wav_pah_neu,
            Qext_pah_neu,
            Qabs_pah_neu,
            Qsca_pah_neu,
        ) = parse_draine_file_lines(savePAHneu)

    # unify wavelength grid
    all_wav = np.unique(
        np.concatenate([wav_sil, wav_gra, wav_pah_ion, wav_pah_neu])
    )
    all_wav.sort()
    # Interpolate Qext to unified wavelength grid
    Qs_sil = interp_Q_to_grid(wav_sil, radii_sil, Qext_sil, all_wav)
    Qs_gra = interp_Q_to_grid(wav_gra, radii_gra, Qext_gra, all_wav)
    Qs_pahion = interp_Q_to_grid(
        wav_pah_ion, radii_pah_ion, Qext_pah_ion, all_wav
    )
    Qs_pahneu = interp_Q_to_grid(
        wav_pah_neu, radii_pah_neu, Qext_pah_neu, all_wav
    )

    # compute mass per H for each material
    mass_per_H_grain_grid = dtg_grid * GAS_MASS_PER_H

    # Define arrays to hold size distribution results
    n_s_small = np.zeros((N_DTG, N_A_INT))  # silicate small
    n_s_large = np.zeros((N_DTG, N_A_INT))  # silicate large
    n_g_small = np.zeros((N_DTG, N_A_INT))  # graphite small
    n_g_large = np.zeros((N_DTG, N_A_INT))  # graphite large
    n_pahion = np.zeros((N_DTG, N_A_INT))  # PAH ionized
    n_pahneu = np.zeros((N_DTG, N_A_INT))  # PAH neutral

    # Compute grain size distributions for each DTG
    if args.mode == "lognormal":
        for ii, mass_per_H_grain in enumerate(mass_per_H_grain_grid):
            n_s_small[ii] = build_lognormal_component(
                a_grid_int,
                args.small_centre,
                args.sigma_lognormal,
                mass_per_H_grain,
                RHO_SIL,
            )
            n_s_large[ii] = build_lognormal_component(
                a_grid_int,
                args.large_centre,
                args.sigma_lognormal,
                mass_per_H_grain,
                RHO_SIL,
            )

            n_g_small[ii] = build_lognormal_component(
                a_grid_int,
                args.small_centre,
                args.sigma_lognormal,
                mass_per_H_grain,
                RHO_GRA,
            )
            n_g_large[ii] = build_lognormal_component(
                a_grid_int,
                args.large_centre,
                args.sigma_lognormal,
                mass_per_H_grain,
                RHO_GRA,
            )

            n_pahion[ii] = build_lognormal_component(
                a_grid_int,
                args.pah_centre,
                args.sigma_lognormal,
                mass_per_H_grain,
                RHO_PAH,
            )
            n_pahneu[ii] = build_lognormal_component(
                a_grid_int,
                args.pah_centre,
                args.sigma_lognormal,
                mass_per_H_grain,
                RHO_PAH,
            )

    else:
        for ii, mass_per_H_grain in enumerate(mass_per_H_grain_grid):
            # choose MRN bin bounds around the centers specified
            small_a_min = args.a_min_mrn
            small_a_max = args.small_centre + (
                args.small_centre - args.a_min_mrn
            )
            large_a_min = small_a_max
            large_a_max = args.a_max_mrn

            pah_a_min = args.a_min_mrn_pah
            pah_a_max = args.a_max_mrn_pah

            if ii == 0:
                print(
                    f"MRN bins (micron): small [{small_a_min:.4g},"
                    f"{small_a_max:.4g}] large [{large_a_min:.4g},"
                    f"{large_a_max:.4g}], PAH [{pah_a_min:.4g},"
                    f"{pah_a_max:.4g}]"
                )

            n_s_small[ii] = build_mrn_component(
                a_grid_int,
                small_a_min,
                small_a_max,
                args.power_mrn,
                mass_per_H_grain,
                RHO_SIL,
            )
            n_s_large[ii] = build_mrn_component(
                a_grid_int,
                large_a_min,
                large_a_max,
                args.power_mrn,
                mass_per_H_grain,
                RHO_SIL,
            )
            n_g_small[ii] = build_mrn_component(
                a_grid_int,
                small_a_min,
                small_a_max,
                args.power_mrn,
                mass_per_H_grain,
                RHO_GRA,
            )
            n_g_large[ii] = build_mrn_component(
                a_grid_int,
                large_a_min,
                large_a_max,
                args.power_mrn,
                mass_per_H_grain,
                RHO_GRA,
            )

            n_pahion[ii] = build_mrn_component(
                a_grid_int,
                pah_a_min,
                pah_a_max,
                args.power_mrn,
                mass_per_H_grain,
                RHO_PAH,
            )
            n_pahneu[ii] = build_mrn_component(
                a_grid_int,
                pah_a_min,
                pah_a_max,
                args.power_mrn,
                mass_per_H_grain,
                RHO_PAH,
            )

    # Define array to hold extinction curves
    Alam_s_small = np.zeros((N_DTG, len(all_wav)))
    Alam_s_large = np.zeros((N_DTG, len(all_wav)))
    Alam_g_small = np.zeros((N_DTG, len(all_wav)))
    Alam_g_large = np.zeros((N_DTG, len(all_wav)))
    Alam_pahion = np.zeros((N_DTG, len(all_wav)))
    Alam_pahneu = np.zeros((N_DTG, len(all_wav)))

    # compute sigma for each component and total
    for ii in range(N_DTG):
        Alam_s_small[ii] = calculate_Alam_over_NH(
            all_wav, radii_sil, Qs_sil, a_grid_int, n_s_small[ii]
        )
        Alam_s_large[ii] = calculate_Alam_over_NH(
            all_wav, radii_sil, Qs_sil, a_grid_int, n_s_large[ii]
        )
        Alam_g_small[ii] = calculate_Alam_over_NH(
            all_wav, radii_gra, Qs_gra, a_grid_int, n_g_small[ii]
        )
        Alam_g_large[ii] = calculate_Alam_over_NH(
            all_wav, radii_gra, Qs_gra, a_grid_int, n_g_large[ii]
        )
        Alam_pahion[ii] = calculate_Alam_over_NH(
            all_wav, radii_pah_ion, Qs_pahion, a_grid_int, n_pahion[ii]
        )
        Alam_pahneu[ii] = calculate_Alam_over_NH(
            all_wav, radii_pah_neu, Qs_pahneu, a_grid_int, n_pahneu[ii]
        )

    # Filter and write
    if args.mode == "lognormal":
        avoid = "mrn"
    else:
        avoid = "lognormal"
    args_dict = vars(args)
    filtered_args = {
        k: str(v)
        for k, v in args_dict.items()
        if v is not None and avoid not in k.lower() and "plot" not in k.lower()
    }

    header = []
    for key, value in filtered_args.items():
        header.append(f"{key}: {value}")
    header = "\n".join(header)
    print("Header:", header)

    grid_name = (
        f"{args.grid_name}_{args.mode}"
        f"_asmall{str(args.small_centre).replace('.', 'p')}"
        f"_alarge{str(args.large_centre).replace('.', 'p')}"
        f"_apah{str(args.pah_centre).replace('.', 'p')}"
    )

    with h5py.File(f"{args.grid_loc}/{grid_name}.hdf5", "w") as hf:
        hf.attrs["Header"] = header

        hf.attrs["date_created"] = str(date.today())

        hf.attrs["axes"] = ["DTG"]
        axes = hf.create_group("axes")
        axes["DTG"] = dtg_grid
        axes["DTG"].attrs["Description"] = (
            "Dust-to-gas mass ratio of the species"
        )
        axes["DTG"].attrs["Units"] = "dimensionless"

        hf.create_group("extinction_curves")
        ext_curves = hf["extinction_curves"]
        ext_curves["wavelength"] = all_wav
        ext_curves["wavelength"].attrs["Description"] = (
            "Wavelength grid for extinction curves"
        )
        ext_curves["wavelength"].attrs["Units"] = "um"

        ext_curves["silicate_small"] = Alam_s_small
        ext_curves["silicate_small"].attrs["Description"] = (
            "Extinction curve A(lam)/N_H for silicate small grain component"
        )
        ext_curves["silicate_small"].attrs["Units"] = "cm^2 per H"

        ext_curves["silicate_large"] = Alam_s_large
        ext_curves["silicate_large"].attrs["Description"] = (
            "Extinction curve A(lam)/N_H for silicate large grain component"
        )
        ext_curves["silicate_large"].attrs["Units"] = "cm^2 per H"

        ext_curves["graphite_small"] = Alam_g_small
        ext_curves["graphite_small"].attrs["Description"] = (
            "Extinction curve A(lam)/N_H for graphite small grain component"
        )
        ext_curves["graphite_small"].attrs["Units"] = "cm^2 per H"

        ext_curves["graphite_large"] = Alam_g_large
        ext_curves["graphite_large"].attrs["Description"] = (
            "Extinction curve A(lam)/N_H for graphite large grain component"
        )
        ext_curves["graphite_large"].attrs["Units"] = "cm^2 per H"

        ext_curves["pah_ionized"] = Alam_pahion
        ext_curves["pah_ionized"].attrs["Description"] = (
            "Extinction curve A(lam)/N_H for PAH ionized component"
        )
        ext_curves["pah_ionized"].attrs["Units"] = "cm^2 per H"

        ext_curves["pah_neutral"] = Alam_pahneu
        ext_curves["pah_neutral"].attrs["Description"] = (
            "Extinction curve A(lam)/N_H for PAH neutral component"
        )
        ext_curves["pah_neutral"].attrs["Units"] = "cm^2 per H"

    print(f"Saved extinction curve grid to {args.grid_loc}/{grid_name}.hdf5")

    if args.plot_example:
        # Plot example extinction curves for specific DTG values
        iv = np.argmin(np.abs(all_wav - V_WAVELENGTH))

        DTG_sil = 0.006  # example dust-to-gas ratio for plotting
        DTG_gra = 0.002  # example dust-to-gas ratio for plotting
        DTG_PAH = 0.0001  # example dust-to-gas ratio for plotting
        idx_dtg_sil = np.argmin(np.abs(dtg_grid - DTG_sil))
        idx_dtg_gra = np.argmin(np.abs(dtg_grid - DTG_gra))
        idx_dtg_pah = np.argmin(np.abs(dtg_grid - DTG_PAH))
        A_total = (
            Alam_s_small[idx_dtg_sil]
            + Alam_s_large[idx_dtg_sil]
            + Alam_g_small[idx_dtg_gra]
            + Alam_g_large[idx_dtg_gra]
            + Alam_pahion[idx_dtg_pah]
            + Alam_pahneu[idx_dtg_pah]
        )
        A_V = A_total[iv]
        A_over_Av = A_total / A_V
        out_prefix = "dust_extinction_curve"
        plot_extinction_curve(
            all_wav,
            A_over_Av,
            args.mode,
            dtg_grid[idx_dtg_sil],
            dtg_grid[idx_dtg_gra],
            dtg_grid[idx_dtg_pah],
            out_prefix,
            savefig=True,
            show=True,
        )
