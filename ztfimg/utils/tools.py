import numpy as np
from astropy import constants

import dask
import dask.array as da
import dask.dataframe as dd


def ccdid_qid_to_rcid(ccdid, qid):
    """ """
    return 4*(ccdid - 1) + qid - 1

def rcid_to_ccdid_qid(rcid):
    """ computes the rcid """
    qid = (rcid%4)+1
    ccdid  = int((rcid-(qid - 1))/4 +1)
    return ccdid,qid

def fit_polynome(x, y, degree, variance=None):
    """ """
    from scipy.optimize import fmin
    from scipy.special import legendre
    
    xdata = (x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))*2-1.
    basemodel = np.asarray([legendre(i)(xdata) for i in range(degree)])
    
    def get_model( parameters ):
        """ """
        return np.dot(basemodel.T, parameters.T).T
    
    def get_chi2( parameters ):
        res = (y - get_model(parameters) )**2
        if variance is not None:
            res /= variance
            
        return np.nansum(res)
        
    guess = np.zeros(degree)
    guess[0] = np.nanmedian(y)
    param = fmin(get_chi2, guess, disp=0)
    return get_model(param)


def rebin_arr(arr, bins, use_dask=False):
    ccd_bins = arr.ravel().reshape( int(arr.shape[0]/bins[0]), 
                                    bins[0],
                                    int(arr.shape[1]/bins[1]), 
                                    bins[1])
    if use_dask:
        return da.moveaxis(ccd_bins, 1,2)
    return np.moveaxis(ccd_bins, 1,2)

def parse_vmin_vmax(data, vmin, vmax):
    """ Parse the input vmin vmax given the data.\n    
    If float or int given, this does nothing \n
    If string given, this computes the corresponding percentile of the data.\n
    e.g. parse_vmin_vmax(data, 40, '90')\n
    -> the input vmin is not a string, so it is returned as such\n
    -> the input vmax is a string, so the returned vmax corresponds to the 
       90-th percent value of data.

    Parameters
    ----------
    data: array     
        data (float array)

    vmin,vmax: string or float/int
        If string, the corresponding percentile is computed\n
        Otherwise, nothing happends.

    Returns
    -------
    float, float
    """
    if vmax is None: vmax="99"
    if vmin is None: vmin = "1"
                
    if type(vmax) == str:
        vmax=np.nanpercentile(data, float(vmax))
        
    if type(vmin) == str:
        vmin=np.nanpercentile(data, float(vmin))
        
    return vmin, vmax

def get_aperture(data, x0, y0, radius, err=None, mask=None,
                         bkgann=None, subpix=0, use_dask=False, **kwargs ):
    """ 
    Returns
    -------
    (dask)array 
    (counts, counterr, flag) of size (3, len(radius), len(x0))
    """
    if use_dask:
        return da.from_delayed( dask.delayed(get_aperture)(data, x0, y0, radius),
                                        shape=(3, radius.size, x0.size), dtype="float"
                                  )
    
    
    from sep import sum_circle
    out = sum_circle(data.astype("float32"), np.atleast_1d(x0), np.atleast_1d(y0), radius,
                         err=err, mask=mask, bkgann=bkgann, subpix=subpix,
                         **kwargs)
    return np.asarray(out)

def extract_sources(data, thresh_=2, err=None, mask=None, use_dask=False, **kwargs):
        """ uses sep.extract to extract sources 'a la Sextractor' """
        #
        # Dask
        #
        import pandas        
        if use_dask:
            columns = ['thresh', 'npix', 'tnpix', 'xmin', 'xmax', 'ymin', 'ymax', 'x', 'y',
                       'x2', 'y2', 'xy', 'errx2', 'erry2', 'errxy', 'a', 'b', 'theta', 'cxx',
                       'cyy', 'cxy', 'cflux', 'flux', 'cpeak', 'peak', 'xcpeak', 'ycpeak',
                       'xpeak', 'ypeak', 'flag']
            
            meta = pandas.DataFrame(columns=columns, dtype="float")
            meta = meta.astype({k:"int" for k in ['npix', 'tnpix',
                                                  'xmin', 'xmax', 'ymin', 'ymax', 'xcpeak', 'ycpeak',
                                                  'xpeak', 'ypeak', 'flag']})
            return dd.from_delayed(
                        dask.delayed(extract_sources)(data, thresh_=thresh_, err=err, mask=mask, use_dask=False, **kwargs),
                                  meta=meta)
        #
        # No Dask
        #
        from sep import extract
        sout = extract(data.astype("float32"),
                        thresh_, err=err, mask=mask, **kwargs)

        return pandas.DataFrame(sout)

def get_source_mask(sourcedf, shape, r=5, use_dask=False):
    """ """
    from sep import mask_ellipse
    if use_dask:
        return da.from_delayed( dask.delayed(get_source_mask)(sourcedf, shape, r=r, use_dask=False),
                                shape=shape, dtype="bool"
                              )
    
        
    mask = np.zeros(shape).astype("bool")
    ellipsemask = mask_ellipse(mask, *sourcedf[["x","y","a","b","theta"]].astype("float").values.T, r=r)
    return mask



# ----------------------------- #
#  Hierarchical Triangular Mesh #
# ----------------------------- #
def get_htm_intersect(ra, dec, radius, depth=7, **kwargs):
    """ Module to get htm overlap (ids)  
    = Based on HMpTy (pip install HMpTy) =

    Parameters
    ----------
    ra, dec: [floatt]
        central point coordinates in decimal degrees or sexagesimal
    
    radius: [float]
        radius of circle in degrees

    depth: [int] -optional-
        depth of the htm
        
    **kwags goes to HMpTy.HTM.intersect 
         inclusive:
             include IDs of triangles that intersect the circle as well as 
             those completely inclosed by the circle. Default True

    Returns
    -------
    list of ID (htm ids overlapping with the input circle.)
    """
    from HMpTy import HTM
    return HTM(depth=depth).intersect(ra, dec, radius, **kwargs)

# --------------------------- #
# - Conversion Tools        - #
# --------------------------- #
def njy_to_mag(njy_, njyerr_=None):
    """ get AB magnitudes corresponding to the input nJy fluxes.
    Returns
    -------
    mags (or mags, dmags if njyerr_ is not None)
    """
    mags = -2.5*np.log10(njy_*10**(-9)/3631)
    if njyerr_ is None:
        return mags
    dmags = +2.5/np.log(10) * njyerr_ / njy_
    return mags, dmags
        

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

def project(radec, radec0, projection="gnomonic"):
    """ project a radec coordinates given a reference coordinate radec0

    Parameters
    ----------
    radec:
        Coordinate to be projected [in deg]

    radec0:
        Reference coordinate [in deg]

    projection: [string] -optional-
        What kind of projection do you want?
        - gnomonic: k = 1 / cos(c)
        - lambert: k = sqrt( 2  / ( 1 + cos(c) ) )
        - stereographic: k = 2 / ( 1 + cos(c) )
        - postel: k = c / sin(c)
        for 
        x = k cos(dec) sin(ra-ra0)
        y = k ( cos(dec0) sin(dec) - sin(dec0) cos(dec) cos(ra-ra0) )


    Returns
    -------
    u, v 

    Credits:
    --------
    Adapted from LSST DESC coord.
    """
    # The equations are given at the above mathworld websites.  They are the same except
    # for the definition of k:
    #
    # x = k cos(dec) sin(ra-ra0)
    # y = k ( cos(dec0) sin(dec) - sin(dec0) cos(dec) cos(ra-ra0) )
    #
    # Lambert:
    #   k = sqrt( 2  / ( 1 + cos(c) ) )
    # Stereographic:
    #   k = 2 / ( 1 + cos(c) )
    # Gnomonic:
    #   k = 1 / cos(c)
    # Postel:
    #   k = c / sin(c)
    # where cos(c) = sin(dec0) sin(dec) + cos(dec0) cos(dec) cos(ra-ra0)

    # cos(dra) = cos(ra-ra0) = cos(ra0) cos(ra) + sin(ra0) sin(ra)

    

    # cos and sin of angles
    # - ref
    ra0,dec0 = np.asarray(radec0)*np.pi/180
    _cosra, _sinra = np.cos(ra0), np.sin(ra0)
    _cosdec, _sindec = np.cos(dec0), np.sin(dec0)
    # other
    ra,dec   = np.asarray(radec)*np.pi/180    
    cosra, sinra   = np.cos(ra), np.sin(ra)
    cosdec, sindec   = np.cos(dec), np.sin(dec)
    
    ###
    
    cosdra = _cosra * cosra
    cosdra += _sinra * sinra

    # sin(dra) = -sin(ra - ra0)
    # Note: - sign here is to make +x correspond to -ra,
    #       so x increases for decreasing ra.
    #       East is to the left on the sky!
    # sin(dra) = -cos(ra0) sin(ra) + sin(ra0) cos(ra)
    sindra = _sinra * cosra
    sindra -= _cosra * sinra

    # Calculate k according to which projection we are using
    cosc = cosdec * cosdra
    cosc *= _cosdec
    cosc += _sindec * sindec
    if projection is None or projection[0] == 'g':
        k = 1. / cosc
    elif projection[0] == 's':
        k = 2. / (1. + cosc)
    elif projection[0] == 'l':
        k = np.sqrt( 2. / (1.+cosc) )
    else:
        c = np.arccos(cosc)
            # k = c / np.sin(c)
            # np.sinc is defined as sin(pi x) / (pi x)
            # So need to divide by pi first.
        k = 1. / np.sinc(c / np.pi)

        # u = k * cosdec * sindra
        # v = k * ( self._cosdec * sindec - self._sindec * cosdec * cosdra )
    u = cosdec * sindra
    v = cosdec * cosdra
    v *= -_sindec
    v += _cosdec * sindec
    u *= k
    v *= k
    
    return u, v

def deproject(uv, radec0, projection="gnomonic"):
    """ project a uv coordinates back to radec given a reference coordinate radec0

    Parameters
    ----------
    uv:
        u, v tangent plane coordinates

    radec0:
        Reference coordinate [in deg]

    projection: [string] -optional-
        What kind of projection do you want?
        - gnomonic: k = 1 / cos(c)
        - lambert: k = sqrt( 2  / ( 1 + cos(c) ) )
        - stereographic: k = 2 / ( 1 + cos(c) )
        - postel: k = c / sin(c)

    Returns
    -------
    ra, dec

    Credits:
    --------
    Adapted from LSST DESC coord.
    """
    # - other
    u, v = np.asarray(uv)/(180/np.pi * 3600)

    # - ref
    ra0,dec0 = np.asarray(radec0)*np.pi/180
    _cosra, _sinra = np.cos(ra0), np.sin(ra0)
    _cosdec, _sindec = np.cos(dec0), np.sin(dec0)
    
    rsq = u*u
    rsq += v*v
    if projection is None or projection[0] == 'g':
        # c = arctan(r)
        # cos(c) = 1 / sqrt(1+r^2)
        # sin(c) = r / sqrt(1+r^2)
        cosc = sinc_over_r = 1./np.sqrt(1.+rsq)
    elif projection[0] == 's':
        # c = 2 * arctan(r/2)
        # Some trig manipulations reveal:
        # cos(c) = (4-r^2) / (4+r^2)
        # sin(c) = 4r / (4+r^2)
        cosc = (4.-rsq) / (4.+rsq)
        sinc_over_r = 4. / (4.+rsq)
    elif projection[0] == 'l':
        # c = 2 * arcsin(r/2)
        # Some trig manipulations reveal:
        # cos(c) = 1 - r^2/2
        # sin(c) = r sqrt(4-r^2) / 2
        cosc = 1. - rsq/2.
        sinc_over_r = np.sqrt(4.-rsq) / 2.
    else:
        r = np.sqrt(rsq)
        cosc = np.cos(r)
        sinc_over_r = np.sinc(r/np.pi)

    # Compute sindec, tandra
    # Note: more efficient to use numpy op= as much as possible to avoid temporary arrays.
    
    # sindec = cosc * self._sindec + v * sinc_over_r * self._cosdec
    sindec = v * sinc_over_r
    sindec *= _cosdec
    sindec += cosc * _sindec
    # Remember the - sign so +dra is -u.  East is left.
    tandra_num = u * sinc_over_r
    tandra_num *= -1.
    # tandra_denom = cosc * self._cosdec - v * sinc_over_r * self._sindec
    tandra_denom = v * sinc_over_r
    tandra_denom *= -_sindec
    tandra_denom += cosc * _cosdec
    
    dec = np.arcsin(sindec)
    ra = ra0 + np.arctan2(tandra_num, tandra_denom)

    return ra, dec
