
import os
import pandas
import numpy as np
import dask
import dask.array as da
import dask.dataframe as dd
import warnings
from astropy.nddata import bitmask

from .base import Quadrant, CCD, FocalPlane
from .utils.tools import rebin_arr, parse_vmin_vmax, rcid_to_ccdid_qid, ccdid_qid_to_rcid
from .utils.astrometry import WCSHolder


__all__ = ["ScienceQuadrant", "ScienceCCD", "ScienceFocalPlane"]


class ComplexImage( object ):
    """ Image that has a mask and enables to get background and noise """

    BITMASK_KEY = ["tracks", "sexsources", "lowresponsivity", "highresponsivity",
                   "noisy", "ghosts", "spillage", "spikes", "saturated",
                   "dead", "nan", "psfsources", "brightstarhalo"]
    
    def set_mask(self, mask):
        """ set the mask to this instance.

        = most likely you do not want to use this method =

        Parameters
        ----------
        mask: 2d array
            numpy or dask array.
        """
        self._mask = mask

    # --------- #
    #  GETTER   #
    # --------- #
    def get_mask(self, reorder=True, from_sources=None,
                 tracks=True, ghosts=True, spillage=True, spikes=True,
                 dead=True, nan=True, saturated=True, brightstarhalo=True,
                 lowresponsivity=True, highresponsivity=True, noisy=True,
                 sexsources=False, psfsources=False,
                 alltrue=False, flip_bits=True,
                 verbose=False, getflags=False,
                 rebin=None, rebin_stat="nanmean",
                 **kwargs):
        """ get a boolean mask (or associated flags). You have the chooce of
        what you want to mask out.

        Image pixels to be mask are set to True.

        A pixel is masked if it corresponds to any of the requested entry.
        For instance if a bitmask is '3', it corresponds to condition 1(2^0) et 2(2^1).
        Since 0 -> tracks and 1 -> sexsources, if tracks or sexsources (or both) is (are) true,
        then the pixel will be set to True.

        Uses: astropy.nddata.bitmask.bitfield_to_boolean_mask

        Parameters
        ----------

        from_source: bool or DataFrame
            A mask will be extracted from the given source.
            (This uses, sep.mask_ellipse)
            Accepted format:
            - None or False: ignored.
            - True: this uses the self.sources
            - DataFrame: this will using this dataframe as source.
                         this dataframe must contain: x,y,a,b,theta

            => accepted kwargs: 'r' the scale (diameter) of the ellipse (5 default)

        // If from_source is used, rest is ignored.



        // These corresponds to the bitmasking definition for the IPAC pipeline //

        Special parameters
        alltrue: bool
            Short cut to set everything to true. Supposedly only background left

        flip_bits: bool
            This should be True to have the aforementioned masking proceedure.
            See astropy.nddata.bitmask.bitfield_to_boolean_mask

        verbose: bool
            Shall this print what you requested ?

        getflags: bool
            Get the bitmask power of 2 you requested instead of the actual masking.

        Returns
        -------
        boolean mask (or list of int, see getflags)
        """
        # // BitMasking
        npda = da if self.use_dask else np
        if alltrue and not getflags:
            return self.mask > 0

        locals_ = locals()
        if verbose:
            print({k: locals_[k] for k in self.BITMASK_KEY})

        flags = [2**i for i,
                 k in enumerate(self.BITMASK_KEY) if locals_[k] or alltrue]
        if getflags:
            return flags

        if self.use_dask:
            mask_ = dask.delayed(bitmask.bitfield_to_boolean_mask)(self.mask,
                                                                   ignore_flags=flags,
                                                                   flip_bits=flip_bits)
            mask = da.from_delayed(mask_, self.shape, dtype="bool")
        else:
            mask = bitmask.bitfield_to_boolean_mask(self.mask,
                                                    ignore_flags=flags, flip_bits=flip_bits)
        # Rebin
        if rebin is not None:
            mask = getattr(da if self.use_dask else np, rebin_stat)(
                rebin_arr(mask, (rebin, rebin), use_dask=self.use_dask), axis=(-2, -1))

        if reorder:
            print("reorder mask")
            mask = mask[::-1, ::-1]
            
        return mask
        
    def get_noise(self, method="sep"):
        """ get a noise image or value
        
        Parameters
        ----------
        method: str
            method used to estimate the background
            - sep or rms: use the sep-background 2d rms image
            - globalrms: mean sep-background rms (float)
            - std: std of the source-masked image (nanstd)

        Returns
        -------
        array or float
            dasked or not depending on data.
            float is method is globalrms
        """
        if method in ["std", "nanstd"]:
            npda = da if self.use_dask else np
            mask = self.get_mask(psfsources=True, sexsources=True)
            bkgd = self.get_background(method="sep")
            data = self.get_data(apply_mask=mask, rm_bkgd=bkgd)
            noise = npda.nanstd(datam)

        elif method in ["sep", "backgroundrms", "rms"]:
            sepbackground = self._get_sepbackound()
            noise = sepbackground.rms()
            if "delayed" in str( type( noise )):
                noise = da.from_delayed(noise,
                                        shape=self.shape,
                                        dtype="float32")
            
        elif method in ["globalrms"]:
            sepbackground = self._get_sepbackound()
            noise = sepbackground.globalrms
            
        else:
            raise ValueError(f"which should be nanstd, globalrms or rms ; {which} given")
        
        return noise
    
    def get_background(self, method="sep", data=None):
        """ get an estimation of the image's background

        Parameters
        ----------
        method: str
            if None, method ="default"
            - "default": returns the background store as self.background (see set_background)
            - "median": gets the median of the fully masked data (self.get_mask(alltrue=True))
            - "sep": returns the sep estimation of the background image (Sextractor-like)

        rm_bkgd: bool
            // ignored if method != median //
            shall the median background estimation be made on default-background subtraced image ?

        Returns
        -------
        float/array (see method)
        """
        # Ready for additional background methods.
        print("calling get_background")
        # Method
        if method == "sep":
            print("using sep background")
            sepbackground = self._get_sepbackound()
            bkgd = sepbackground.back()
            if "delayed" in str( type(bkgd) ):
                bkgd = da.from_delayed(bkgd, shape=self.SHAPE, dtype="float32")

        elif method == "median":
            print("using median background")
            if data is None:
                mask = self.get_mask(psfsources=True, sexsources=True)
                data = self.get_data(rm_bkgd=False, apply_mask=mask)
                
            if "dask" in str( type(data) ):
                # median no easy to massively //
                bkgd = data_.map_blocks(np.nanmedian)
            else:
                bkgd = np.nanmedian( data )
                            
        else:
            raise NotImplementedError("Only median background implemented")

        return bkgd    

    def _get_sepbackound(self, bw=192, bh=192, update=False, **kwargs):
        """ """
        if not hasattr(self, "_sepbackground") or update:
            print("creating _sepbackground")
            from sep import Background
            data = self.get_data().copy().astype("float32")
            smask = self.get_mask(psfsources=True, sexsources=True)
            data[smask] = np.NaN
            bkgs_prop = {**dict(mask=smask, bh=bh, bw=bw), **kwargs}
            if self.use_dask:
                self._sepbackground = dask.delayed(Background)(data, **bkgs_prop)
            else:
                self._sepbackground = Background(np.ascontiguousarray(data), **bkgs_prop)

        return self._sepbackground

    # ============== #
    #   Properties   #
    # ============== #
    @property
    def mask(self):
        """ mask image. """
        return self._mask


class ScienceQuadrant(Quadrant, WCSHolder, ComplexImage):

    # "family"
    _CCDCLASS = "ScienceCCD"
    _FocalPlaneCLASS = "ScienceFocalPlane"

    def __init__(self, data=None, mask=None, header=None, meta=None):
        """ Science Quadrant. You most likely want to load it using from_* class method

        See also
        --------
        from_filename: load the instance using a quadrant filename
        
        
        """
        _ = super().__init__(data=data, header=header)

        if mask is not None:
            self.set_mask(mask)

        self._meta = meta

    @classmethod
    def from_filename(cls, filename, filename_mask=None,
                          download=True, as_path=True,
                          use_dask=False, persist=False, **kwargs):
        """
        Parameters
        ----------
        filename: str
            name of the file. 
            This could be a full path or a ztf filename. 
            (see as_path option)
            
        filename_mask: str
            name of the file containing the mask. 
            This could be a full path or a ztf filename. 
            (see as_path option).
            If None, filename_mask will be built based on filename.

        download: bool
             Downloads the file (filename or filename_masl) if necessary.

        as_path: bool
            Set to True if the filename [filename_mask] are path and not just ztf filename, 
            hence bypassing ``filename = ztfquery.io.get_file(filename)``

        use_dask: bool
            Shall this use dask to load the image data ?

        persist: bool
            = ignored if use_dask=False = 
            shall this run .persist() on data ?

        **kwargs goes to ztfquery.io.get_file() = ignored if as_path=True =
            

        Returns
        -------
        instance
        """
        from ztfquery import io, buildurl
        from astropy.io import fits
            
        meta = io.parse_filename(filename)
        # If mask image not given, let's check if it exist somewhere either as mskimg.fits or mskimg.fits.gz
        # trying mskimg.fits.gz first
        if filename_mask is None:
            filename_mask = buildurl.filename_to_url(filename, suffix="mskimg.fits.gz", source="local")
            if not os.path.isfile(filename_mask): # trying mskimg.fits next
                filename_mask = buildurl.filename_to_url(filename, suffix="mskimg.fits", source="local")
            if not os.path.isfile(filename_mask): # ok, nothing exist, back to None
                filename_mask = None 

        # -> at this point if a mskimg.fits.gz or a mskimg.fits exists, filename_mask is it.
        
        if use_dask:
            # Getting the filenames, download if needed
            if not as_path:
                filepath = dask.delayed(io.get_file)(filename, suffix="sciimg.fits",
                                                   downloadit=download,
                                                   show_progress=False, maxnprocess=1,
                                                   **kwargs)
            else:
                filepath = filename

            if filename_mask is None:
                filepath_mask = dask.delayed(io.get_file)(filename, suffix="mskimg.fits",
                                                                  downloadit=download,
                                                                  show_progress=False, maxnprocess=1,
                                                                  **kwargs)
            else: # Both given
                filepath_mask = filename_mask
                
            # Getting the filenames
            # - Data
            data = da.from_delayed(dask.delayed(fits.getdata)(filepath),
                                   shape=cls.SHAPE, dtype="float32")
            header = dask.delayed(fits.getheader)(filepath)
            # - Mask
            mask = da.from_delayed(dask.delayed(fits.getdata)(filepath_mask),
                                   shape=cls.SHAPE, dtype="int16")
            if persist:
                data = data.persist()
                header = header.persist()
                mask = mask.persist()

        else:
            if not as_path:
                filepath = io.get_file(filename, suffix="sciimg.fits",
                                     downloadit=download, **kwargs)
            else:
                filepath = filename
                
            if filename_mask is None:
                filepath_mask = io.get_file(filename, suffix="mskimg.fits",
                                     downloadit=download, **kwargs)
            else: # both given
                filepath_mask = filename_mask

                
            data = fits.getdata(filepath)
            header = fits.getheader(filepath)
            # - Mask
            mask = fits.getdata(filepath_mask)

        # self

        this = cls(data=data, header=header,
                   mask=mask, meta=meta)
        
        this._filename = filename
        this._filepath = filepath
        return this

    def load_wcs(self, header=None):
        """ loads the wcs solution from the header
        
        Parameters
        ----------
        header: fits.Header
            header containing the wcs information. 
            If None, self.header will be used.
        """
        if header is None:
            header = self.get_header(compute=True)

        super().load_wcs(header)

    # -------- #
    #  GETTER  #
    # -------- #
    def get_rawimage(self, use_dask=None, which="quadrant", as_path=False, **kwargs):
        """ get the raw image of the given science quadrant
        
        This uses ztfquery to fetch the raw image path 
        and inputs it to RawQuadrant or RawCCD .from_filename method
        (see which)

        Parameters
        ----------

        Returns
        -------
        """
        from . import raw        
        from ztfquery.buildurl import get_rawfile_of_filename
        rawfile = get_rawfile_of_filename(self.filename)

        if use_dask is None:
            use_dask = self._use_dask
        
        if which == "quadrant":
            rawimg = raw.RawQuadrant.from_filename(rawfile, qid=self.qid,
                                                    use_dask=use_dask,
                                                    as_path=as_path,**kwargs)
            
        elif which == "ccd":
            rawimg = raw.RawCCD.from_filename(rawfile, use_dask=use_dask,
                                                  as_path=as_path, **kwargs)
            
        else:
            raise ValueError(f"Cannot parse input which {which} (quadrant or ccd implemented)")

        return rawimg
    
    def get_data(self, apply_mask=False,
                     rm_bkgd=False, 
                     rebin=None, rebin_stat="nanmean",
                     maskvalue=np.NaN,
                     zp=None, reorder=True,
                     **kwargs):
        """ get a copy of the data affected by background and/or masking.

        Parameters
        ---------
        apply_mask: bool, array
            Shall a default masking be applied (i.e. all bad pixels to nan)
            if an array is given, this will be the mask used.

        rm_bkgd: bool, array, float
            Should the data be background subtracted ?
            if something else than a bool is given, 
            it will be used as background

        maskvalue: float
            Whick values should the masked out data have ?


        **kwargs goes to super().get_data()

        Returns
        -------
        2d array
            numpy or dask.
        """  
        data_ = super().get_data(reorder=reorder, rebin=None, **kwargs)

        # Mask
        if apply_mask is not False: # accepts bool or array
            if type(apply_mask) is bool:
                apply_mask = self.get_mask()
                
            data_ = data_.copy() # do not affect current data
            data_[apply_mask] = maskvalue  # OK
            
        # Background
        if rm_bkgd is not False: # accepts bool or array
            if type(rm_bkgd) is bool:
                rm_bkgd = self.get_background()
            data_ = data_ - rm_bkgd

        # ZP            
        if zp is not None:
            magzp = self.get_value("MAGZP")
            coef = 10 ** (-0.4*(magzp - zp)) # change zp system
            if "delayed" in str( type( coef ) ):
                if "dask" in str( type( data_ ) ):
                    coef = da.from_delayed(coef, dtype="float32", shape=())
                else: # means numpy data
                    coef = coef.compute()
            
            data_ = data_*coef

        # rebin            
        if rebin is not None:
            data_ = getattr(da if self.use_dask else np, rebin_stat)(
                rebin_arr(data_, (rebin, rebin), use_dask=self.use_dask), axis=(-2, -1))

        return data_

    def get_source_mask(self, reorder=True, thresh=5, r=8):
        """ """
        if not hasattr(self, "_source_mask"):
            from .utils.tools import extract_sources, get_source_mask
            mask = self.get_mask(reorder=reorder)
            noise = self.get_noise(method="nanstd")
            bkgd = self.get_background(method="sep")
            data = self.get_data(apply_mask=mask, rm_bkgd=bkgd)
            sources = extract_sources(
                data, thresh_=thresh, err=noise, mask=mask, use_dask=self.use_dask)
            self._source_mask = get_source_mask(
                sources, self.shape, use_dask=self.use_dask, r=r)

        return self._source_mask

    def get_aperture(self, x0, y0, radius,
                     data=None,
                     bkgann=None, subpix=0,
                     system="xy",
                     mask=None, err=None,
                     as_dataframe=False,
                     **kwargs):
        """ get the aperture (circular) photometry.


        Parameters
        ----------
        x0, y0, radius: 1d-array
            Center coordinates and radius (radii) of aperture(s).
            (could be x,y, ra,dec or u,v ; see system)

        data: 2d-array
            2d image the aperture will be applied on. 
            (self.data otherwise, see also `which` and `dataprop`)

        bkgann: 2d-array
            Length 2 tuple giving the inner and outer radius of a “background annulus”.
            If supplied, the background is estimated by averaging unmasked pixels in this annulus. 
            If supplied, the inner and outer radii obey numpy broadcasting rules along with ``x``,
            ``y`` and ``r``.

        subpix: int
            Subpixel sampling factor. If 0, exact overlap is calculated. 5 is acceptable.

        system: str
            In which system are the input x0, y0:
            - xy (ccd )
            - radec (in deg, sky)
            - uv (focalplane)
            
        dataprop: dict
            = ignored if data is given =
            kwargs for the get_data method
            using ``data = self.get_data(**dataprop)``

        mask: 2d-array
            mask image. ``mask=self.get_mask()`` used if None
        
        err: 2d-array
            error image. ``mask=self.get_noise()`` used if None
        
        as_dataframe: [bool]
            set the returned format.
            If as_dataFrame=True, this will be a dataframe with
            3xn-radius columns (f_0...f_i, f_0_e..f_i_e, f_0_f...f_i_f)
            standing for fluxes, errors, flags.

        **kwargs goes to super().get_aperture(**kwargs)
        
        Returns
        -------
        (3, n) array or `pandas.DataFrame`
            array: with n the number of radius.
            
        """

        if system == "radec":
            x0, y0 = self.radec_to_xy(x0, y0)
        elif system == "uv":
            x0, y0 = self.uv_to_xy(x0, y0)
        elif system != "xy":
            raise ValueError(
                f"system must be xy, radec or uv ;  {system} given")

        # Data
        if data is None:
            data = self.get_data()
            
        # Err
        if err is None:
            err = self.get_noise()
            
        # Mask
        if mask is None:
            mask = self.get_mask()

        # calling back base.get_aperture()
        return super().get_aperture(x0, y0, radius,
                                    data=data, err=err,
                                    bkgann=bkgann, subpix=subpix,
                                    as_dataframe=as_dataframe,
                                    **kwargs)
    # -------- #
    # CATALOGS #
    # -------- #
    # - ZTFCATS
    def get_psfcat(self, show_progress=False, use_dask=None, **kwargs):
        """ get the psf photometry catalog generated by the ztf-pipeline
        
        Parameters
        ----------
        show_progress: bool
            option of io.get_file to display progress while downloading the data.

        use_dask: bool
            should the catalog dataframe be as dask.dataframe ?
        
        **kwargs goes to io.get_file()

        Returns
        -------
        `pandas.DataFrame`
        """
        if use_dask is None:
            use_dask = self.use_dask

        if use_dask:
            columns = ['x', 'y', 'ra', 'dec', 'flux', 'sigflux', 'mag',
                       'sigmag', 'snr', 'chi', 'sharp', 'flags']
            meta = pandas.DataFrame(columns=columns, dtype="float")
            return dd.from_delayed(dask.delayed(self.get_psfcat)(use_dask=False,
                                                                  show_progress=False, **kwargs),
                                   meta=meta)

        from ztfquery import io
        from astropy.io import fits
        from astropy.table import Table
        psffilename = io.get_file(self.filename, suffix="psfcat.fits",
                                  show_progress=show_progress, **kwargs)
        data = Table(fits.open(psffilename)[
                     1].data).to_pandas().set_index("sourceid")
        # Not the same standard as calibrator cats.
        data[["xpos", "ypos"]] -= 1
        return data.rename({"xpos": "x", "ypos": "y"}, axis=1)

    def get_sexcat(self, as_table=False, show_progress=False, use_dask=None, **kwargs):
        """ get the sextractor photometry catalog generated by the ztf-pipeline
        (nested-aperture photometry catalog)
        
        

        Parameters
        ----------
        as_table: bool
            should the returned table be a pandas dataframe or an astropy table.
            careful, nested apertures (MAG_APER, FLUX_APER and 
            associated errors are droped when using pandas.)
        
        show_progress: bool
            option of io.get_file to display progress while downloading the data.

        use_dask: bool
            should the catalog dataframe be as dask.dataframe ?
            
        Returns
        -------
        `pandas.DataFrame` or `astropy.Table`
        """
        if use_dask is None:
            use_dask = self.use_dask

        if use_dask:
            columns = ['FLAGS', 'XWIN_IMAGE', 'YWIN_IMAGE', 'X_WORLD', 'Y_WORLD',
                       'XPEAK_IMAGE', 'YPEAK_IMAGE', 'THETAWIN_IMAGE', 'ERRTHETAWIN_IMAGE',
                       'ALPHAWIN_J2000', 'DELTAWIN_J2000', 'THETAWIN_J2000', 'X2WIN_IMAGE',
                       'Y2WIN_IMAGE', 'XYWIN_IMAGE', 'AWIN_WORLD', 'BWIN_WORLD', 'MAG_ISO',
                       'MAGERR_ISO', 'MAG_AUTO', 'MAGERR_AUTO', 'MAG_ISOCOR', 'MAGERR_ISOCOR',
                       'MAG_PETRO', 'MAGERR_PETRO', 'MAG_BEST', 'MAGERR_BEST', 'MU_THRESHOLD',
                       'MU_MAX', 'BACKGROUND', 'THRESHOLD', 'ELONGATION', 'ISOAREA_WORLD',
                       'ISOAREAF_WORLD', 'ISO0', 'ISO1', 'ISO2', 'ISO3', 'ISO4', 'ISO5',
                       'ISO6', 'ISO7', 'FWHM_IMAGE', 'KRON_RADIUS', 'PETRO_RADIUS',
                       'CLASS_STAR', 'FLUX_BEST', 'FLUXERR_BEST', 'FLUX_AUTO', 'FLUXERR_AUTO',
                       'FLUX_ISO', 'FLUXERR_ISO', 'X_IMAGE', 'Y_IMAGE', 'X2_IMAGE', 'Y2_IMAGE',
                       'XY_IMAGE', 'THETA_IMAGE', 'ERRAWIN_IMAGE', 'ERRBWIN_IMAGE',
                       'THETAWIN_WORLD', 'ERRX2WIN_IMAGE', 'ERRY2WIN_IMAGE', 'ERRXYWIN_IMAGE',
                       'IMAFLAGS_ISO', 'NIMAFLAGS_ISO', 'ERRAWIN_WORLD', 'ERRBWIN_WORLD',
                       'ERRTHETAWIN_WORLD', 'A_IMAGE', 'ERRA_IMAGE', 'B_IMAGE', 'ERRB_IMAGE',
                       'A_WORLD', 'ERRA_WORLD', 'B_WORLD', 'ERRB_WORLD', 'ERRTHETA_IMAGE',
                       'ERRX2_IMAGE', 'ERRY2_IMAGE', 'ERRXY_IMAGE', 'AWIN_IMAGE', 'BWIN_IMAGE',
                       'FLUX_PETRO', 'FLUXERR_PETRO']
                
            meta = pandas.DataFrame(columns=columns, dtype="float")
            return dd.from_delayed(dask.delayed(self.get_sexcat)(use_dask=False, show_progress=False,
                                                                 as_table=False,
                                                                 **kwargs), meta=meta)

        from ztfquery import io
        from astropy.table import Table
        from atropy.io import fits
        psffilename = io.get_file(
            self.filename, suffix="sexcat.fits", show_progress=show_progress, **kwargs)
        # .to_pandas().set_index("sourceid")
        tbl = Table(fits.open(psffilename)[1].data)
        if as_table:
            return tbl

        names = [name for name in tbl.colnames if len(tbl[name].shape) <= 1]
        return tbl[names].to_pandas().set_index("NUMBER")

    # --------- #
    #  INTERNAL #
    # --------- #
    def _setxy_to_cat_(self, cat, drop_outside=True, pixelbuffer=10):
        """ """
        warnings.warn("_setxy_to_cat_ is deprecated")
        x, y = self.radec_to_xy(cat["ra"], cat["dec"])
        u, v = self.radec_to_uv(cat["ra"], cat["dec"])
        cat["x"] = x
        cat["y"] = y
        cat["u"] = u
        cat["v"] = v

        if drop_outside:
            ymax, xmax = self.shape
            cat = cat[cat["x"].between(+pixelbuffer, ymax-pixelbuffer)
                      & cat["y"].between(+pixelbuffer, xmax-pixelbuffer)]
        return cat

    # =============== #
    #  Properties     #
    # =============== #
    @property
    def wcs(self):
        """ astropy wcs solution loaded from the header """
        if not hasattr(self, "_wcs"):
            self.load_wcs()
        return self._wcs

    # // shortcut
    @property
    def filtername(self):
        """ name of the filter (from self.meta) """
        return self.meta.filtercode

    @property
    def filterid(self):
        """ filter number (1: zg, 2: zr, 3:zi) see self.filtername """
        return self.meta.filterid

    @property
    def rcid(self):
        """ id of the quadrant in the focal plane (from meta) (0->63) """
        return self.get_value("rcid", attr_ok=False) # avoid loop

    @property
    def ccdid(self):
        """ id of the ccd (1->16) """
        return self.get_value("ccdid", attr_ok=False) # avoid loop

    @property
    def qid(self):
        """ id of the quadrant (1->4) """
        return self.get_value("qid", attr_ok=False) # avoid loop

    @property
    def fieldid(self):
        """ number of the field (from meta) """
        return self.get_value("field", attr_ok=False) # avoid loop

    @property
    def filefracday(self):
        """ id corresponding to the 'fraction of the day' (from meta) """
        return self.get_value("filefracday", attr_ok=False) # avoid loop

    @property
    def obsdate(self):
        """ observing date with the yyyy-mm-dd format. """
        return "-".join(self.meta[["year", "month", "day"]].values)

class ScienceCCD(CCD, ComplexImage):
    
    _COLLECTION_OF = ScienceQuadrant
    # "family" 
    _QUADRANTCLASS = "ScienceQuadrant"
    _FocalPlaneCLASS = "ScienceFocalPlane"
    
    # =============== #
    #  Properties     #
    # =============== #
    @property
    def meta(self):
        """ pandas.dataframe concatenating meta data from the quadrants. """
        if not hasattr(self, "_meta") or self._meta is None:
            self._meta = pandas.concat(
                {i: q.meta for i, q in self.quadrants.items()}, axis=1).T

        return self._meta

    def get_mask(self, **kwargs):
        """ """
        # 'calling' is the method actually called by each quadrant query
        return self.get_data(calling="get_mask", **kwargs)
    
    def get_data(self, apply_mask=False,
                     rm_bkgd=False, 
                     rebin=None, rebin_stat="mean",
                     maskvalue=np.NaN,
                     zp=None,
                     **kwargs):
        """ 
        
        Parameters
        ----------
        
        rm_bkgd: bool, str, array
            Should the data be background subtracted ? 
            - False: no background subtraction
            - True: background subtraction at the ccd level
            - array: 
                - if shape 4 x quadrant.shape, this mapped at the quadrant-level
                - if shape ccd.shape, this is applied at the ccd-level
            - str:
                - quadrant: removed at the quandrant level (equiv to rm_bkgd=True)
                - ccd: removed at the ccd-level (equiv to rm_bkgd=ccd.get_background())
        

        rebin_stat: str
            = applies only if rebin is not None =
            numpy (dask.array) method used for rebinning the data.
            For instance, if rebin=4 and rebin_stat = median
            the median of a 4x4 pixel will be used to form a new pixel.
            The dimension of the final image will depend on rebin.

        """
        # Step 1. parse the background removal that can be
        # made at the CCD level.
        if rm_bkgd is not False:
            if type( rm_bkgd ) is str:
                if rm_bkgd == "ccd":
                    rm_bkgd = self.get_background() # CCDSHAPE
                elif rm_bkgd == "quadrant":
                    rm_bkgd = self.call_quadrants("get_background") # 4 x QUADRANTSHAPE

            if type( rm_bkgd ) is bool: # means True
                rm_bkgd = self.get_background() # CCDSHAPE

            # here rm_bkgd should be an array
            if np.shape(rm_bkgd) == (4, *self.qshape):
                # 4 x QUADRANTSHAPE -> CCDSHAPE
                rm_bkgd = self._quadrantdata_to_ccddata(qdata=rm_bkgd)

            # here rm_bkgd should be a CCDSHAPE array
            if not list(np.shape(rm_bkgd)) == list(self.shape):
                raise ValueError(f"rm_bkgd (ccd) has the wrong shape {np.shape(rm_bkgd)}")

        # mask
        if apply_mask is not False:
            if type( apply_mask ) is bool: # means True
                apply_mask = self.get_mask() # CCDSHAPE
                
            # here rm_bkgd should be an array
            if np.shape(apply_mask) == (4, *self.qshape):
                # 4 x QUADRANTSHAPE -> CCDSHAPE
                apply_mask = self._quadrantdata_to_ccddata(qdata=apply_mask)

            # here rm_bkgd should be a CCDSHAPE array
            if not list(np.shape(apply_mask)) == list(self.shape):
                raise ValueError(f"apply_mask (ccd) has the wrong shape {np.shape(apply_mask)}")

        
        # Step 2. get data
        ccddata = self._quadrantdata_to_ccddata(zp=zp, **kwargs).copy()
        # background        
        if not rm_bkgd is False: # accepts array
            ccddata -= rm_bkgd
            
        # mask
        if apply_mask is not False: # accepts array
            ccddata[apply_mask] = maskvalue  # OK

        # Step 3: Rebin & persisting
        if rebin is not None:
            npda = da if "dask" in str( type(ccddata) ) else np
            ccddata = getattr(npda, rebin_stat)(rebin_arr(ccddata, (rebin, rebin),
                                                        use_dask=self.use_dask),
                                              axis=(-2, -1))
        if "dask" in str( type(ccddata) ) and persist:
            ccddata = ccddata.persist()

        return ccddata
    

class ScienceFocalPlane(FocalPlane):
    """ """
    _CCDCLASS = "ScienceCCD"
    _COLLECTION_OF = ScienceCCD
    
    def get_files(self, client=None, suffix=["sciimg.fits", "mskimg.fits"], as_dask="futures"):
        """ fetch the files of the focal plane (using ztfquery.io.bulk_get_file) 

        Parameters
        ----------
        client: `dask.distributed.Client`
            client to use to run the bulk downloading

        suffix: str or list of str
            suffix corresponding to the image to download.

        as_dask: str
            kind of dask object to get 
            - delayed or futures (download started)
            
        Returns
        -------
        list
            list of files
        """
        from ztfquery import io
        return io.bulk_get_file(self.filenames, client=client,
                                suffix=suffix, as_dask=as_dask)

    @property
    def meta(self):
        """ pandas.dataframe concatenating meta data from the quadrants. """
        if not hasattr(self, "_meta") or self._meta is None:
            self._meta = pandas.concat({i: ccd.meta for i, ccd in self.ccds.items()}
                                       ).set_index("rcid")

        return self._meta
