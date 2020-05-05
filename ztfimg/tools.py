import numpy as np
from astropy import constants

# --------------------------- #
# - Conversion Tools        - #
# --------------------------- #
def counts_to_flux(counts, dcounts, magzp, wavelength):
    """ converts counts into flux (erg/s/cm2/A) """
    flux = counts * 10**(-(2.406+magzp) / 2.5 ) / (wavelength**2)
    if dcounts is not None:
        dflux = dcounts * 10**(-(2.406+magzp) / 2.5 ) / (wavelength**2)
    else:
        dflux = None
    return flux, dflux

def flux_to_counts(flux, dflux, magzp, wavelength):
    """ converts flux (erg/s/cm2/A) into counts """
    counts = flux / (10**(-(2.406+magzp) / 2.5 ) / (wavelength**2))
    if dflux is not None:
        dcounts = dflux / (10**(-(2.406+magzp) / 2.5 ) / (wavelength**2))
    else:
        dcounts = None
    return counts, dcounts

def counts_to_mag(counts, dcounts, magzp, wavelength):
    """ counts into ABmag (trhough flux) """
    return flux_to_mag(*counts_to_flux(counts,dcounts, magzp, wavelength), wavelength=wavelength)

def mag_to_counts(mag, dmag, magzp, wavelength):
    """ ABmag to counts (trhough flux) """
    return flux_to_counts(*mag_to_flux(mag, dmag, wavelength=wavelength), magzp, wavelength)

def flux_to_mag(flux, dflux, wavelength=None, zp=None, inhz=False):
    """ Converts fluxes (erg/s/cm2/A) into AB or zp magnitudes


    Parameters
    ----------
    flux, fluxerr: [float or array]
        flux and its error 

    wavelength: [float or array] -optional-
        central wavelength [in AA] of the photometric filter.
        // Ignored if inhz=True //

    zp: [float or array] -optional-
        zero point of for flux;
        // Ignored if inhz=True //
        // Ignored if wavelength is provided //

    inhz:
        set to true if the flux (and flux) are given in erg/s/cm2/Hz
        (False means in erg/s/cm2/AA)
        
    Returns
    -------
    - float or array (if magerr is None)
    - float or array, float or array (if magerr provided)
    
    """
    if inhz:
        zp = -48.598 # instaad of -48.60 such that hz to aa is correct
        wavelength = 1
    else:
        if zp is None and wavelength is None:
            raise ValueError("zp or wavelength must be provided")
        if zp is None:
            zp = -2.406 
        else:
            wavelength=1
            
    mag_ab = -2.5*np.log10(flux*wavelength**2) + zp
    if dflux is None:
        return mag_ab, None
    
    dmag_ab = +2.5/np.log(10) * dflux / flux
    return mag_ab, dmag_ab

def mag_to_flux(mag, magerr=None, wavelength=None, zp=None, inhz=False):
    """ converts magnitude into flux

    Parameters
    ----------
    mag: [float or array]
        AB magnitude(s)

    magerr: [float or array] -optional-
        magnitude error if any

    wavelength: [float or array] -optional-
        central wavelength [in AA] of the photometric filter.
        // Ignored if inhz=True //

    zp: [float or array] -optional-
        zero point of for flux;
        // Ignored if inhz=True //
        // Ignored if wavelength is provided //

    inhz:
        set to true if the flux (and flux) are given in erg/s/cm2/Hz
        (False means in erg/s/cm2/AA)

    Returns
    -------
    - float or array (if magerr is None)
    - float or array, float or array (if magerr provided)
    """
    if inhz:
        zp = -48.598 # instaad of -48.60 such that hz to aa is correct
        wavelength = 1
    else:
        if zp is None and wavelength is None:
            raise ValueError("zp or wavelength must be provided")
        if zp is None:
            zp = -2.406 
        else:
            wavelength=1

    flux = 10**(-(mag-zp)/2.5) / wavelength**2
    if magerr is None:
        return flux, None
    
    dflux = np.abs(flux*(-magerr/2.5*np.log(10))) # df/f = dcounts/counts
    return flux, dflux

def flux_aa_to_hz(flux_aa, wavelength):
    """ """
    return flux_aa * (wavelength**2 / constants.c.to("AA/s").value)
    
def flux_hz_to_aa(flux_hz, wavelength):
    """ """
    return flux_hz / (wavelength**2 / constants.c.to("AA/s").value)

# --------------------------- #
#    Array Tools              #
# --------------------------- #

def restride(arr, binfactor, squeezed=True, flattened=False):
    """ Rebin nd-array `arr` by `binfactor`.

    Let `arr.shape = (s1, s2, ...)` and `binfactor = (b1, b2, ...)` (same
    length), new shape will be `(s1/b1, s2/b2, ... b1, b2, ...)` (squeezed).
    * If `binfactor` is an iterable of length < `arr.ndim`, it is prepended
      with 1's.
    * If `binfactor` is an integer, it is considered as the bin factor for all
      axes.
    If `flattened`, the bin axes are explicitely flattened into a single
    axis. Note that this will probably induce a copy of the array.
    Bin 2D-array by a factor 2:
    >>> restride(N.ones((6, 8)), 2).shape
    (3, 4, 2, 2)
    Bin 2D-array by a factor 2, with flattening of the last 2 bin axes:
    >>> restride(N.ones((6, 8)), 2, flattened=True).shape
    (3, 4, 4)
    Bin 2D-array by uneven factor (3, 2):
    >>> restride(N.ones((6, 8)), (3, 2)).shape
    (2, 4, 3, 2)
    Bin 3D-array by factor 2 over the last 2 axes, and take bin average:
    >>> q = N.arange(2*4*6).reshape(2, 4, 6)
    >>> restride(q, (2, 2)).mean(axis=(-1, -2))
    array([[[ 3.5,  5.5,  7.5],
            [15.5, 17.5, 19.5]],
           [[27.5, 29.5, 31.5],
            [39.5, 41.5, 43.5]]])
    Bin 3D-array by factor 2, and take bin average:
    >>> restride(q, 2).mean(axis=(-1, -2, -3))
    array([[15.5, 17.5, 19.5],
           [27.5, 29.5, 31.5]])
    .. Note:: for a 2D-array, `restride(arr, (3, 2))` is equivalent to::
         np.moveaxis(arr.ravel().reshape(arr.shape[1]/3, arr.shape[0]/2, 3, 2), 1, 2)
    """
    try:                        # binfactor is list-like
        # Convert binfactor to [1, ...] + binfactor
        binshape = [1] * (arr.ndim - len(binfactor)) + list(binfactor)
    except TypeError:           # binfactor is not list-like
        binshape = [binfactor] * arr.ndim
    assert len(binshape) == arr.ndim, "Invalid bin factor (shape)."
    assert (~np.mod(arr.shape, binshape).astype('bool')).all(), \
        "Invalid bin factor (modulo)."
    # New shape
    rshape = [ d // b for d, b in zip(arr.shape, binshape) ] + binshape
    # New stride
    rstride = [ d * b for d, b in zip(arr.strides, binshape) ] + list(arr.strides)
    rarr = np.lib.stride_tricks.as_strided(arr, rshape, rstride)
    if flattened:               # Flatten bin axes, which may induce a costful copy!
        rarr = rarr.reshape(rarr.shape[:-(rarr.ndim - arr.ndim)] + (-1,))
    return rarr.squeeze() if squeezed else rarr  # Remove length-1 axes
