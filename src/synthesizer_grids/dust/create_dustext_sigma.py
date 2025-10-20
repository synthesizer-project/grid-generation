#!/usr/bin/env python3
"""
Create dust extinction curves from Draine optical properties.

Parse Draine Sil_81.gz, Gra_81.gz, PAHion_30.gz, PAHneu_30.gz files,
construct grain-size distributions (either MRN or the custom lognormal form)
and compute extinction cross-sections (sigma).
"""

import io
import gzip
import math
import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from numpy.typing import NDArray

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.special import erf
import matplotlib.pyplot as plt
import requests

from synthesizer.emission_models.attenuation import Calzetti2000
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
        list of strings with line breaks
    """
    with gzip.GzipFile(fileobj=io.BytesIO(gzbytes)) as gf:
        txt = gf.read().decode('utf-8', errors='ignore')   
    with open(savename, "w") as file:
        file.writelines(txt)
    return txt


def parse_draine_file_lines(filename: str) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """
    Parse Draine optical property file (e.g., Sil_81.txt) and extract
    radius grid (micron), wavelength grid (micron), and Qext, Qabs, Qsca matrices.
    
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
        if 'NRAD' in L:
            left = L.split('=')[0].strip()
            toks = left.split()
            if len(toks) >= 1:
                try:
                    NRAD = int(toks[0])
                    print (toks)
                    a_min = float(toks[1])
                    a_max = float(toks[2])
                except Exception:
                    pass
        if 'NWAV' in L:
            left = L.split('=')[0].strip()
            toks = left.split()
            if len(toks) >= 1:
                try:
                    NWAV = int(toks[0])
                    line_num_NWAV = ii
                except Exception:
                    pass
    if NRAD is None or NWAV is None:
        raise RuntimeError("Failed to find NRAD (number of radii) or NWAV (number of wavelength) in header. Inspect file header manually, and retry.")
    
    print (F'NRAD = {NRAD}, NWAV = {NWAV}, amin ={a_min}, amax={a_max}')

    skip_ftr = 3 # lines to skip after NWAV line (usually 3)
    first_skip_hdr = 1 + line_num_NWAV + skip_ftr  # start parsing after this line
    
    a_grid = 10**np.linspace(np.log10(a_min), np.log10(a_max), num=NRAD, endpoint=True) 
    
    Qabs: NDArray = np.zeros((NRAD, NWAV))
    Qsca: NDArray = np.zeros((NRAD, NWAV))
    Qext: NDArray = np.zeros((NRAD, NWAV))
    # gcos: NDArray = np.zeros((NRAD, NWAV))  
    
    for ii in range(NRAD):        
        if 'PAH' in filename:
            add = 1
            usecols=(0,1,2,3)
        else:
            add = 0   
            usecols=(0,1,2)
         
        tmp = np.genfromtxt(
            filename,
            skip_header = first_skip_hdr + ii*NWAV + ii*skip_ftr,
            max_rows=NWAV,
            dtype=float,
            usecols=usecols,
        )
        
        if ii==0: wav_grid = tmp[:,0]  # micron            
        Qabs[ii] = tmp[:,1+add]
        Qsca[ii] = tmp[:,2+add]
        if 'PAH' in filename:
            Qext[ii] = tmp[:,1]
        else:
            Qext[ii] = tmp[:,1+add] + tmp[:,2+add]
        # gcos[ii] = tmp[:,3+add]
        

    return a_grid, wav_grid, Qext.T, Qabs.T, Qsca.T


def build_mrn_component(a_grid_micron: NDArray,
                        a_min: float,
                        a_max: float,
                        power: float,
                        mass_per_H: float,
                        rho: float) -> NDArray:
    """
    Build a truncated MRN-like grain-size distribution component
    with dn/da \propto a^power over [a_min, a_max] (micron).
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
        S = (a_max**(power + 1.0) - a_min**(power + 1.0)) / (power + 1.0)

    p4 = power + 4.0
    if abs(p4) < 1e-12:
        I_a3 = math.log(a_max / a_min)
    else:
        I_a3 = (a_max**p4 - a_min**p4) / p4

    a3_mean_micron3 = I_a3 / S  # <a^3> in micron^3
    micron_to_cm = (1.0*um).to("cm").value  # 1e-4
    a3_mean_cm3 = a3_mean_micron3 * (micron_to_cm**3)
    mass_per_grain = (4.0/3.0) * math.pi * rho * a3_mean_cm3
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
    rho: float) -> NDArray:
    """
    Build the lognormal-type component with the form
    n(a) = C / a^4 * exp( - (ln(a/a0))^2 / (2 sigma^2) ),
    where a is in micron and n(a) is number per H per micron.
    C is chosen so the component total mass per H = mass_per_H (g per H).
    Reference: Hirashita 2015 (https://arxiv.org/abs/1412.3866)
    See eq 31 in that paper.
    implies C = 3 * mass_per_H / (4 pi rho sqrt(2 pi) sigma_ln 
    for the lmit 0 -> infinity 
    
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
    micron_to_cm = (1.0*um).to("cm").value  # 1e-4
    prefac = (4.0/3.0) * math.pi * rho * (micron_to_cm**3) * micron_to_cm  # 
    denom = prefac * math.sqrt(2.0 * math.pi) * sigma_ln 
    if denom <= 0:
        raise ValueError("Denominator for C non-positive.")
    C = mass_per_H / denom

    # build n(a) on grid
    a = a_grid_micron.copy()
    # avoid division issues at a==0 (grid doesn't include 0)
    exponent = -0.5 * ((np.log(a) - math.log(a0_micron)) / sigma_ln) ** 2
    n_a = (C / (a ** 4)) * np.exp(exponent)
    # n_a units: number per H per micron (because a in micron, and C has correct units)
    return n_a

# ---------------------------
# Integration: sigma_ext per H
# ---------------------------
def integrate_sigma_per_H(
    wav_micron: NDArray,
    radii_micron: NDArray,
    Qext: NDArray,
    a_grid_micron: NDArray,
    n_a: NDArray) -> NDArray:
    """
    A(lam)/N_H = 2.5 ln(e) integral π a^2 Q_ext(a,lam) n(a) da
    
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
            extinction cross-section per H at each wavelength (cm^2 per H)
    """
    micron_to_cm = (1.0*um).to("cm").value  # 1e-4
    a_grid_cm = a_grid_micron * micron_to_cm
    # n_a is number per H per micron -> convert to number per H per cm: n_a_per_cm = n_a / micron_to_cm
    n_a_per_cm = n_a / micron_to_cm

    log_r_input = np.log(radii_micron)
    Nwav = wav_micron.size
    Alam_by_N_H = np.zeros(Nwav, dtype=float)
    for iw in range(Nwav):
        q = Qext[iw, :]
        pos = q > 0
        if np.sum(pos) >= 2:
            f_log = interp1d(log_r_input[pos], np.log(q[pos]), bounds_error=False, fill_value=0)
            q_interp = np.exp(f_log(np.log(a_grid_micron)))
        else:
            f_lin = interp1d(radii_micron, q, bounds_error=False, fill_value=0)
            q_interp = f_lin(a_grid_micron)
        # integrand: pi a^2 Q * n(a) da  (a in cm, n in per cm, da in cm)
        integrand = math.pi * (a_grid_cm ** 2) * q_interp * n_a_per_cm
        Alam_by_N_H[iw] = 2.5 * np.log(np.e) * np.trapz(integrand, a_grid_cm)
    
    return Alam_by_N_H


def interp_Q_to_grid(
    wav_orig: NDArray,
    radii_orig: NDArray,
    Q_orig: NDArray,
    wav_target: NDArray
) -> NDArray:
    """ Interpolate Qext (or Qabs, Qsca) from original wavelength grid to target wavelength grid.  
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
        f = interp1d(wav_orig, Q_orig[:, j], kind='linear', bounds_error=False, fill_value=0)
        Qw[:, j] = f(wav_target)
    
    return Qw


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Build extinction curves from Draine optical files with MRN or lognormal-size components.")
    parser.add_argument("--mode", choices=["mrn", "lognormal"], default="lognormal", help="size-distribution mode")
    # parser.add_argument("--sil-file", type=str, default=None, help="local Sil_81.gz path (if omitted, will download)")
    # parser.add_argument("--gra-file", type=str, default=None, help="local Gra_81.gz path (if omitted, will download)")
    # parser.add_argument("--d2g-sil", type=float, default=D2G_SIL_DEFAULT, help="dust-to-gas mass ratio (silicate)")
    # parser.add_argument("--d2g-gra", type=float, default=D2G_GRA_DEFAULT, help="dust-to-gas mass ratio (graphite)")
    parser.add_argument("--a-min-mrn", type=float, default=0.005, help="MRN a_min (micron) when using MRN")
    parser.add_argument("--a-max-mrn", type=float, default=0.25, help="MRN a_max (micron) when using MRN")
    parser.add_argument("--power-mrn", type=float, default=-3.5, help="MRN power-law exponent (dn/da ∝ a^power)")
    parser.add_argument("--small-center", type=float, default=0.01, help="center of small bin (micron)")
    parser.add_argument("--large-center", type=float, default=0.1, help="center of large bin (micron)")
    parser.add_argument("--sigma-ln", type=float, default=0.4, help="sigma (ln-space) for lognormal (or custom) components")
    parser.add_argument("--f-small-sil", type=float, default=0.5, help="fraction of silicate dust mass in the small bin")
    parser.add_argument("--f-small-gra", type=float, default=0.5, help="fraction of graphite dust mass in the small bin")
    # parser.add_argument("--out-prefix", default=OUT_PREFIX, help="output file prefix")
    # parser.add_argument("--na", type=int, default=N_A_INT, help="number of radii in integration grid")
    args = parser.parse_args()
    
    
    # ---------------------------
    # Default resources / params
    # ---------------------------
    SIL_URL = "https://www.astro.princeton.edu/~draine/dust/diel/Sil_81.gz"
    GRA_URL = "https://www.astro.princeton.edu/~draine/dust/diel/Gra_81.gz"
    PAH_ion = "https://www.astro.princeton.edu/~draine/dust/diel/PAHion_30.gz"
    PAH_neu = "https://www.astro.princeton.edu/~draine/dust/diel/PAHneu_30.gz"

    # default outputs
    OUT_PREFIX = "extinction"

    # default dust-to-gas mass ratios (mass dust / mass gas)
    D2G_SIL_DEFAULT = 0.006
    D2G_C_DEFAULT = 0.004
    D2G_PAHion_DEFAULT = D2G_C_DEFAULT * 0.1
    D2G_PAHnue_DEFAULT = D2G_C_DEFAULT * 0.1
    D2G_GRA_DEFAULT = D2G_C_DEFAULT * 0.8

    # material densities (g cm^-3)
    RHO_SIL = 3.5
    RHO_GRA = 2.24
    RHO_PAH = 2.24

    # gas mass per H (g) with helium correction
    MU = 1.4
    M_H = 1.6738e-24 # in grams
    GAS_MASS_PER_H = MU * M_H  # g per H

    # integration a-grid (micron)
    A_INT_MIN = 1e-4
    A_INT_MAX = 10.0
    N_A_INT = 500

    # V wavelength for A(V)
    V_WAVELENGTH = 0.55  # micron
    
    
    # download and process files
    saveGRA = './Gra_81.txt'  # save
    if Path(saveGRA).exists():
        radii_gra, wav_gra, Qext_gra, Qabs_gra, Qsca_gra = parse_draine_file_lines(saveGRA)
    else:    
        gra_bytes = download_bytes(GRA_URL)
        gra_lines = read_gz_lines(gra_bytes, savename='Gra_81.txt')
        radii_gra, wav_gra, Qext_gra, Qabs_gra, Qsca_gra = parse_draine_file_lines(saveGRA)
        
    saveSIL = './Sil_81.txt'  # save
    if Path(saveSIL).exists():
        radii_sil, wav_sil, Qext_sil, Qabs_sil, Qsca_sil = parse_draine_file_lines(saveSIL)
    else:    
        sil_bytes = download_bytes(SIL_URL)
        sil_lines = read_gz_lines(sil_bytes, savename='Sil_81.txt')
        radii_sil, wav_sil, Qext_sil, Qabs_sil, Qsca_sil = parse_draine_file_lines(saveSIL)
    
    savePAHion = './PAHion_30.txt'  # save
    if Path(savePAHion).exists():
        radii_pah_ion, wav_pah_ion, Qext_pah_ion, Qabs_pah_ion, Qsca_pah_ion = parse_draine_file_lines(savePAHion)
    else:    
        pah_ion_bytes = download_bytes(PAH_ion)
        pah_ion_lines = read_gz_lines(pah_ion_bytes, savename='PAHion_30.txt')
        radii_pah_ion, wav_pah_ion, Qext_pah_ion, Qabs_pah_ion, Qsca_pah_ion = parse_draine_file_lines(savePAHion)
    
    savePAHneu = './PAHneu_30.txt'  # save
    if Path(savePAHneu).exists():
        radii_pah_neu, wav_pah_neu, Qext_pah_neu, Qabs_pah_neu, Qsca_pah_neu = parse_draine_file_lines(savePAHneu)
    else:    
        pah_neu_bytes = download_bytes(PAH_neu)
        pah_neu_lines = read_gz_lines(pah_neu_bytes, savename='PAHneu_30.txt')
        radii_pah_neu, wav_pah_neu, Qext_pah_neu, Qabs_pah_neu, Qsca_pah_neu = parse_draine_file_lines(savePAHneu)

    # unify wavelength grid
    all_wav = np.unique(np.concatenate([wav_sil, wav_gra, wav_pah_ion, wav_pah_neu]))
    all_wav.sort()

    # build integration a_grid (micron)
    a_grid_int = np.logspace(math.log10(A_INT_MIN), math.log10(A_INT_MAX), N_A_INT)
    
    # compute mass per H for each material
    mass_per_H_sil = D2G_SIL_DEFAULT * GAS_MASS_PER_H
    mass_per_H_gra = D2G_GRA_DEFAULT * GAS_MASS_PER_H
    mass_per_H_pahion = D2G_PAHion_DEFAULT * GAS_MASS_PER_H
    mass_per_H_pahneu = D2G_PAHnue_DEFAULT * GAS_MASS_PER_H
    print(f"Mass per H (sil): {mass_per_H_sil:.3e} g/H ; (gra): {mass_per_H_gra:.3e} g/H")

    # Build component n(a) depending on MRN or lognormal mode
    # Assign mass fractions to small/large bins per material using f_small args.
    ms_s_small = args.f_small_sil * mass_per_H_sil
    ms_s_large = (1.0 - args.f_small_sil) * mass_per_H_sil
    mg_g_small = args.f_small_gra * mass_per_H_gra
    mg_g_large = (1.0 - args.f_small_gra) * mass_per_H_gra
    
    mg_pahion_small = mass_per_H_pahion  #PAH only in small grains
    mg_pahneu_small = mass_per_H_pahneu  #PAH only in small grains

    
    if args.mode == "mrn":
        # choose MRN bin bounds around the centers specified
        small_a_min = max(A_INT_MIN, args.small_center / 3.0)
        small_a_max = min(args.large_center / 3.0, args.small_center * 3.0)
        large_a_min = max(small_a_max, args.large_center / 3.0)
        large_a_max = min(A_INT_MAX, args.large_center * 10.0)

        print("MRN bins (micron): small [{:.4g}, {:.4g}] large [{:.4g}, {:.4g}]".format(small_a_min, small_a_max, large_a_min, large_a_max))

        n_s_small = build_mrn_component(a_grid_int, small_a_min, small_a_max, args.power_mrn, ms_s_small, RHO_SIL)
        n_s_large = build_mrn_component(a_grid_int, large_a_min, large_a_max, args.power_mrn, ms_s_large, RHO_SIL)
        n_g_small = build_mrn_component(a_grid_int, small_a_min, small_a_max, args.power_mrn, mg_g_small, RHO_GRA)
        n_g_large = build_mrn_component(a_grid_int, large_a_min, large_a_max, args.power_mrn, mg_g_large, RHO_GRA)
        
        n_pahion_small = build_mrn_component(a_grid_int, small_a_min, small_a_max, args.power_mrn, mg_pahion_small, RHO_PAH)
        n_pahneu_small = build_mrn_component(a_grid_int, small_a_min, small_a_max, args.power_mrn, mg_pahneu_small, RHO_PAH)
    else:
        # lognormal custom form
        n_s_small = build_lognormal_component(a_grid_int, args.small_center, args.sigma_ln, ms_s_small, RHO_SIL)
        n_s_large = build_lognormal_component(a_grid_int, args.large_center, args.sigma_ln, ms_s_large, RHO_SIL)
        n_g_small = build_lognormal_component(a_grid_int, args.small_center, args.sigma_ln, mg_g_small, RHO_GRA)
        n_g_large = build_lognormal_component(a_grid_int, args.large_center, args.sigma_ln, mg_g_large, RHO_GRA)
        
        n_pahion_small = build_lognormal_component(a_grid_int, args.small_center, args.sigma_ln, mg_pahion_small, RHO_PAH)
        n_pahneu_small = build_lognormal_component(a_grid_int, args.small_center, args.sigma_ln, mg_pahneu_small, RHO_PAH)

    n_s_total = n_s_small + n_s_large
    n_g_total = n_g_small + n_g_large
    
    n_pahion_total = n_pahion_small  # PAH only small
    n_pahneu_total = n_pahneu_small  # PAH only small


    Qs_sil = interp_Q_to_grid(wav_sil, radii_sil, Qext_sil, all_wav)
    Qs_gra = interp_Q_to_grid(wav_gra, radii_gra, Qext_gra, all_wav)
    Qs_pahion = interp_Q_to_grid(wav_pah_ion, radii_pah_ion, Qext_pah_ion, all_wav)
    Qs_pahneu = interp_Q_to_grid(wav_pah_neu, radii_pah_neu, Qext_pah_neu, all_wav)

    # compute sigma for each component and total
    sigma_s_small = integrate_sigma_per_H(all_wav, radii_sil, Qs_sil, a_grid_int, n_s_small)
    sigma_s_large = integrate_sigma_per_H(all_wav, radii_sil, Qs_sil, a_grid_int, n_s_large)
    sigma_g_small = integrate_sigma_per_H(all_wav, radii_gra, Qs_gra, a_grid_int, n_g_small)
    sigma_g_large = integrate_sigma_per_H(all_wav, radii_gra, Qs_gra, a_grid_int, n_g_large)
    sigma_pahion_small = integrate_sigma_per_H(all_wav, radii_pah_ion, Qs_pahion, a_grid_int, n_pahion_small)
    sigma_pahneu_small = integrate_sigma_per_H(all_wav, radii_pah_neu, Qs_pahneu, a_grid_int, n_pahneu_small)

    sigma_s_total = sigma_s_small + sigma_s_large
    sigma_g_total = sigma_g_small + sigma_g_large
    sigma_pah_total = sigma_pahion_small + sigma_pahneu_small
    sigma_total = sigma_s_total + sigma_g_total + sigma_pah_total

    A_per_NH = 1.086 * sigma_total  # mag * cm^2 / H
    iv = np.argmin(np.abs(all_wav - V_WAVELENGTH))
    A_over_Av = A_per_NH / A_per_NH[iv]


    # Save Qext per-material (already saved earlier) and size arrays (also saved)
    # Plot A(λ)/A(V)
    # all_wav*=1e4  # convert to AA
    plt.figure(figsize=(8,5))
    plt.plot(all_wav, A_over_Av, label='total')
    plt.plot(all_wav, (1.086 * sigma_s_total) / (1.086 * sigma_s_total[iv]), linestyle='--', label='silicate total')
    plt.plot(all_wav, (1.086 * sigma_g_total) / (1.086 * sigma_g_total[iv]), linestyle=':', label='graphite total')
    
    plt.plot(all_wav, (1.086 * sigma_pah_total) / (1.086 * sigma_pah_total[iv]), linestyle=(0,(2,2)), label='PAH')

    
    calzetti = Calzetti2000(ampl=10.0)
    plt.plot(all_wav, calzetti.get_tau(all_wav*um), 'k--', label='Calzetti+2000')

    plt.xlabel("Wavelength (um)")
    plt.ylabel("A(λ) / A(V)")
    plt.title(f"Extinction ({args.mode}) — sil D/G={D2G_SIL_DEFAULT:.3g}, gra D/G={D2G_GRA_DEFAULT:.3g}")
    # plt.xlim(all_wav.min(), all_wav.max())
    # plt.ylim(1e-3, max(5.0, np.max(A_over_Av)*2.0))
    plt.xlim(0.09, 1)
    plt.ylim(0, 10)
    plt.grid(which='both', ls='--', alpha=0.4)
    plt.legend(fontsize='small', ncol=2)
    plt.tight_layout()
    png_name = f"{OUT_PREFIX}_{args.mode}.png"
    plt.savefig(png_name, dpi=200)
    print(f"Saved plot: {png_name}")
    plt.show()
