import numpy as np
from wqsat_format import s2_reader

def temporal_composite(tiles_path, coordinates=None):
    """
    Generates composite water quality proxies and bathymetric indexes from multiple satellite images.
    """
    num_tiles = len(tiles_path)
    data_bands, coord = s2_reader.S2Reader(tiles_path[0], coordinates).read_bands()
    h, w = data_bands['B2'].shape
    
    # Preallocate arrays
    Zgreen, Zred = np.zeros((h, w, num_tiles)), np.zeros((h, w, num_tiles))
    Rs492, Rs559, Rs704, chl = (np.zeros((h, w, num_tiles)) for _ in range(4))
    
    for t, tiles in enumerate(tiles_path):
        tile_path = tiles_path[t]
        data_bands, _ = s2_reader.S2Reader(tiles_path[0], coordinates).read_bands()
        
        # Extract bands and calculate bathymetric proxies
        Rs492[:, :, t], Rs559[:, :, t], Rs704[:, :, t] = data_bands['B2'], data_bands['B3'], data_bands['B5']
        chl[:, :, t] = data_bands['chl']
        Zgreen[:, :, t] = np.log(3140 * data_bands['B2']) / np.log(3140 * data_bands['B3'])
        Zred[:, :, t] = np.log(3140 * data_bands['B2']) / np.log(3140 * data_bands['B4'])
    
    # Compute max values and indices
    Zgr_max, Zr_max = np.nanmax(Zgreen, axis=2), np.nanmax(Zred, axis=2)
    idx_Zgr_max = np.argmax(np.nan_to_num(Zgreen, nan=float('-inf')), axis=2)
    
    # Select composite values using indices
    m, n = idx_Zgr_max.shape
    i, j = np.ogrid[:m, :n]
    Rs492_c, Rs559_c, Rs704_c, chl_c = Rs492[i, j, idx_Zgr_max], Rs559[i, j, idx_Zgr_max], Rs704[i, j, idx_Zgr_max], chl[i, j, idx_Zgr_max]
    
    return Zgr_max, Zr_max, Rs492_c, Rs559_c, Rs704_c, chl_c, coord

def switching_model(pSDBgreen, pSDBred, Zgr_coef=3.5, Zr_coef=2.0):
    """
    Implements a switching model for satellite-derived bathymetry (SDB).
    """
    a = (Zgr_coef - pSDBred) / (Zgr_coef - Zr_coef)
    b = 1 - a
    SDBw = a * pSDBred + b * pSDBgreen
    
    # Create the SDB array with NaNs
    SDB = np.full_like(pSDBred, np.nan)
    SDB = np.where(pSDBred < Zr_coef, pSDBred, SDB)
    SDB = np.where((pSDBred > Zr_coef) & (pSDBgreen > Zgr_coef), pSDBgreen, SDB)
    SDB = np.where((pSDBred >= Zr_coef) & (pSDBgreen <= Zgr_coef), SDBw, SDB)
    SDB[SDB < 0] = np.nan  # Remove negative values
    
    return SDB

def odw_model(SDB, Rs492, Rs559, Rs704):
    """
    Applies the Optical Deep Water (ODW) model for bathymetric correction.
    """
    mask_clear = (Rs492 <= 0.003) | (Rs559 <= 0.003)
    SDB[mask_clear] = np.nan  # Remove clear water pixels
    
    # Compute Ymax for turbid waters
    Ymax = (-0.251 * np.log(Rs704)) + 0.8
    Ymax[Ymax < 0] = np.nan
    
    # Apply constraints based on Ymax
    y = np.log(SDB)
    SDB[y > Ymax] = np.nan  # Remove invalid depth estimates
    
    return SDB