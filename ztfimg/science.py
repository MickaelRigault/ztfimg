
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


class ScienceQuadrant(Quadrant, WCSHolder):
    
    BITMASK_KEY = ["tracks", "sexsources", "lowresponsivity", "highresponsivity",
                   "noisy", "ghosts", "spillage", "spikes", "saturated",
                   "dead", "nan", "psfsources", "brightstarhalo"]

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
                          download=True, as_path=False,
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
        from ztfquery import io
        from astropy.io import fits

        if filename_mask is None:
            filename_mask = filename

        meta = io.parse_filename(filename)
        
        if use_dask:
            # Getting the filenames, download if needed
            if not as_path:
                filename = dask.delayed(io.get_file)(filename, suffix="sciimg.fits",
                                                   downloadit=download,
                                                   show_progress=False, maxnprocess=1,
                                                   **kwargs)

                filename_mask = dask.delayed(io.get_file)(filename_mask, suffix="mskimg.fits",
                                                   downloadit=download,
                                                   show_progress=False, maxnprocess=1,
                                                   **kwargs)

            # Getting the filenames
            # - Data
            data = da.from_delayed(dask.delayed(fits.getdata)(filename),
                                   shape=cls.SHAPE, dtype="float32")
            header = dask.delayed(fits.getheader)(filename)
            # - Mask
            mask = da.from_delayed(dask.delayed(fits.getdata)(filename_mask),
                                   shape=cls.SHAPE, dtype="int16")
            if persist:
                data = data.persist()
                header = header.persist()
                mask = mask.persist()

        else:
            if not as_path:
                filename = io.get_file(filename, suffix="sciimg.fits",
                                     downloadit=download, **kwargs)
                filename_mask = io.get_file(filename_mask, suffix="mskimg.fits",
                                     downloadit=download, **kwargs)
            data = fits.getdata(filename)
            header = fits.getheader(filename)
            # - Mask
            mask = fits.getdata(filename_mask)

        # self

        this = cls(data=data, header=header,
                   mask=mask, meta=meta)
        
        this._filename = filename
        return this

    def set_mask(self, mask):
        """ set the mask to this instance.

        = most likely you do not want to use this method =

        Parameters
        ----------
        mask: 2d array
            numpy or dask array.
        """
        self._mask = mask

    def load_wcs(self, header=None):
        """ loads the wcs solution from the header
        
        Parameters
        ----------
        header: fits.Header
            header containing the wcs information. 
            If None, self.header will be used.
        """
        if header is None:
            header = self.header

        super().load_wcs(header)

    # -------- #
    #  CORE    #
    # -------- #

    # -------- #
    #  GETTER  #
    # -------- #
    def get_center(self, system="xy"):
        """ x and y or RA, Dec coordinates of the centroid. (shape[::-1]) 

        Parameters
        ----------
        system: str
            system you want to get the center:
            - 'xy': ccd pixel coordinates
            - 'uv': projecting plane coordinates (center on focal plane center)
            - 'radec': RA, Dec sky coordinates.

        Returns
        -------
        (float, float)
            requested center.
            
        """
        if system in ["xy", "pixel", "pixels", "pxl"]:
            return (np.asarray(self.shape[::-1])+1)/2

        if system in ["uv", "tangent"]:
            return np.squeeze(self.xy_to_uv(*self.get_center(system="xy")))

        if system in ["radec", "coords", "worlds"]:
            return np.squeeze(self.xy_to_radec(*self.get_center(system="xy")))

        raise ValueError(
            f"cannot parse the given system {system}, use xy, radec or uv")

    def get_ccd(self, use_dask=False, **kwargs):
        """ ScienceCCD object containing this quadrant. """
        return ScienceCCD.from_single_filename(self.filename,
                                                      use_dask=use_dask,
                                                      **kwargs)

    def get_focalplane(self, use_dask=False, **kwargs):
        """ FocalPlane (64 quadrants making 16 CCDs) containing this quadrant """
        return ScienceFocalPlane.from_single_filename(self.filename,
                                                              use_dask=use_dask,
                                                              **kwargs)
    
    def get_rawimage(self, use_dask=None, which="quadrant", **kwargs):
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
                                                    use_dask=use_dask, **kwargs)
            
        elif which == "ccd":
            rawimg = raw.RawCCD.from_filename(rawfile, use_dask=use_dask, **kwargs)
            
        else:
            raise ValueError(f"Cannot parse input which {which} (quadrant or ccd implemented)")

        return rawimg
    
    def get_data(self, apply_mask=False, maskvalue=np.NaN,
                     rm_bkgd=False, whichbkgd="median",
                     rebin=None, rebin_stat="nanmean",
                     reorder=True,
                     **kwargs):
        """ get a copy of the data affected by background and/or masking.

        Parameters
        ---------
        which: str
            shortcut to acces the data. This will format the rest of the input.
            - data: copy of the data.
            - 


        rebin:
            None

        rebin_stat:
            "nanmean"


        // If not which is None only //


        clean: [bool] -optional-
            shortcut to get_dataclean()
            // rest is ignored //

        apply_mask: [bool] -optional-
            Shall a default masking be applied (i.e. all bad pixels to nan)

        maskvalue: [float] -optional-
            Whick values should the masked out data have ?

        rm_bkgd: [bool] -optional-
            Should the data be background subtracted ?

        whichbkgd: [bool] -optional-
            // ignored if rm_bkgd=False //
            which background should this use (see self.get_background())

        **kwargs goes to self.get_mask()

        Returns
        -------
        2d array (data)

        """
        data_ = super().get_data(reorder=reorder, rebin=None)

        if apply_mask:
            data_ = data_.copy() # do not affect current data
            data_[self.get_mask(**kwargs)] = maskvalue  # OK

        if rm_bkgd:
            # not data -= to create a copy.
            data_ = data_-self.get_background(method=whichbkgd, rm_bkgd=False)

        if rebin is not None:
            data_ = getattr(da if self.use_dask else np, rebin_stat)(
                rebin_arr(data_, (rebin, rebin), use_dask=self.use_dask), axis=(-2, -1))

        return data_

    def get_mask(self, from_sources=None,
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

        return mask

    def get_background(self, method="median", rm_bkgd=False, backup_default="sep"):
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

        backup_default: str
            If no background has been set yet, which method should be the default backgroud.
            If no background set and backup_default is None an AttributeError is raised.

        Returns
        -------
        float/array (see method)
        """
        # Ready for additional background methods.
        
        # Method
        if method == "median":
            data_ = self.get_data(rm_bkgd=rm_bkgd, apply_mask=True, alltrue=True)
            if "dask" in str( type(data_) ):
                # median no easy to massively //
                bkgd = data_.map_blocks(np.nanmedian)
            else:
                bkgd = np.nanmedian( data_ )

        else:
            raise NotImplementedError("Only median background implemented")

        return bkgd

    def get_dataclean(self):
        """ """
        if not hasattr(self, "_dataclean"):
            self._dataclean = self.get_data(
                apply_mask=True, rm_bkgd=False) - self.get_source_background()

        return self._dataclean

    def get_noise(self, which="nanstd"):
        """ """
        if which == "nanstd":
            npda = da if self.use_dask else np
            datamasked = self.get_data(
                apply_mask=True, rm_bkgd=True, whichbkgd="median", alltrue=True)
            return npda.nanstd(datamasked)

        if which in ["sep", "sextractor", "globalrms"]:
            if not hasattr(self, "_back"):
                self._load_background_()
            return self._back.globalrms

        if which in ["backgroundrms", "rms"]:
            if not hasattr(self, "_back"):
                self._load_background_()
            if self.use_dask:
                return da.from_delayed(self._back.rms(),
                                       shape=self.shape, dtype="float32")
            return self._back.rms()

        raise ValueError(
            f"which should be nanstd, globalrms or rms ; {which} given")

    def get_source_mask(self, thresh=5, r=8):
        """ """
        if not hasattr(self, "_source_mask"):
            from .utils.tools import extract_sources, get_source_mask
            data = self.get_data(
                apply_mask=True, rm_bkgd
                =True, whichbkgd="median")
            mask = self.get_mask()
            noise = self.get_noise(which="nanstd")
            sources = extract_sources(
                data, thresh_=thresh, err=noise, mask=mask, use_dask=self.use_dask)
            self._source_mask = get_source_mask(
                sources, self.shape, use_dask=self.use_dask, r=r)

        return self._source_mask

    def get_source_background(self):
        """ """
        if not hasattr(self, "_source_background"):
            if not hasattr(self, "_back"):
                self._load_background_()
                
            if self.use_dask:
                self._source_background = da.from_delayed(self._back.back(),
                                                          shape=self.shape, dtype="float32")
            else:
                self._source_background = self._back.back()

        return self._source_background

    def get_aperture(self, x0, y0, radius,
                     data=None,
                     bkgann=None, subpix=0,
                     system="xy",
                     which="dataclean",
                     dataprop={},
                     mask=None, maskprop={},
                     err=None, noiseprop={},
                     as_dataframe=False,
                     **kwargs):
        """


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
            mask image. ``mask=self.get_mask(**maskprop)`` used if None
            
        maskprop: dict
            = ignored if mask is given =
            kwargs for the get_mask method
            using ``mask = self.get_mask(**maskprop)``
        
        err: 2d-array
            error image. ``mask=self.get_noise(**noiseprop)`` used if None
        
        noiseprop: dict
            = ignored if mask is given =
            kwargs for the get_noise method
            using ``mask = self.get_noise(**noiseprop)``

        as_dataframe: [bool]
            set the returned format.
            If as_dataFrame=True, this will be a dataframe with
            3xn-radius columns (f_0...f_i, f_0_e..f_i_e, f_0_f...f_i_f)
            standing for fluxes, errors, flags.

        **kwargs goes to super().get_aperture(**kwargs)
        
        Returns
        -------
        2d-array or `pandas.DataFrame`
           (see unit: (counts, dcounts) | (flux, dflux) | (mag, dmag)) + flag (see get_flag option)
        """

        if system == "radec":
            x0, y0 = self.radec_to_xy(x0, y0)
        elif system == "uv":
            x0, y0 = self.uv_to_xy(x0, y0)
        elif system != "xy":
            raise ValueError(
                f"system must be xy, radec or uv ;  {system} given")

        if err is None:
            err = self.get_noise(**noiseprop)

        if mask is None:
            mask = self.get_mask(**maskprop)

        if data is None:
            data = self.get_data(**dataprop)

        # calling back base.get_aperture()
        return super().get_aperture(x0, y0, radius,
                                    data=data,
                                    err=err,
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
    def _load_background_(self):
        """ """
        from sep import Background
        data = self.data.copy()
        smask = self.get_source_mask()
        data[smask] = np.NaN
        if self.use_dask:
            self._back = dask.delayed(Background)(data)
        else:
            self._back = Background(data.astype("float32"))

    def _setxy_to_cat_(self, cat, drop_outside=True, pixelbuffer=10):
        """ """
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
    def dataclean(self):
        """ shortcut to get_dataclean() """
        return self.get_dataclean()

    @property
    def mask(self):
        """ mask image. """
        return self._mask

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


class ScienceCCD(CCD):
    SHAPE = 3080*2, 3072*2
    
    _QUADRANTCLASS = ScienceQuadrant

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

    @property
    def filenames(self):
        """ list of the filename of the different quadrants constituing the data. """
        return [q.filenames for q in self.quandrants]


class ScienceFocalPlane(FocalPlane):
    """ """
    _CCDCLASS = ScienceCCD

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

    @property
    def filenames(self):
        """ list of the filename of the different quadrants constituing the data. """
        return [q.filename for ccdid, ccd in self.ccds.items() for qid, q in ccd.quadrants.items()]
