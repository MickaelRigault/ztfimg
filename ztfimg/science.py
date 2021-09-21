
import pandas
import numpy as np
import dask
import dask.array as da
import dask.dataframe as dd

from astropy.nddata import bitmask

from .base import _Quadrant_, _CCD_, _FocalPlane_
from .tools import rebin_arr, parse_vmin_vmax
from .astrometry import WCSHolder

class ScienceQuadrant( _Quadrant_, WCSHolder ):

    BITMASK_KEY = [ "tracks","sexsources","lowresponsivity","highresponsivity",
                    "noisy","ghosts","spillage","spikes","saturated",
                    "dead","nan","psfsources","brightstarhalo"]

            
    def __init__(self, data=None, mask=None, header=None, use_dask=True, meta=None):
        """ """
        _ = super().__init__(use_dask=use_dask)
        if data is not None:
            self.set_data(data)
            
        if mask is not None:
            self.set_mask(mask)
            
        if header is not None:
            self.set_header(header)
            
        self._meta = meta
    
    @classmethod
    def from_filename(cls, filename, filenamemask=None, download=True,
                          use_dask=True, persist=True, **kwargs):
        """ 
        Parameters
        ----------
        download: [bool] -optional-
             Downloads the maskfile if necessary.

        **kwargs goes to ztfquery.io.get_file()
        """
        from ztfquery import io
        from astropy.io import fits
        
        if filenamemask is None:
            filenamemask = filename
            
        if use_dask:
            # Getting the filenames, download if needed
            sciimgpath = dask.delayed(io.get_file)(filename, suffix="sciimg.fits",
                                                       downloadit=download,
                                                       show_progress=False, maxnprocess=1,
                                                       **kwargs)

            mskimgpath = dask.delayed(io.get_file)(filenamemask, suffix="mskimg.fits",
                                                       downloadit=download,
                                                       show_progress=False, maxnprocess=1,
                                                       **kwargs)
            
            # Getting the filenames
            # - Data            
            data   = da.from_delayed( dask.delayed(fits.getdata)(sciimgpath),
                                            shape=cls.SHAPE, dtype="float")
            header = dask.delayed(fits.getheader)(sciimgpath)
            # - Mask
            mask   = da.from_delayed( dask.delayed(fits.getdata)(mskimgpath),
                                            shape=cls.SHAPE, dtype="float")
            if persist:
                data = data.persist()
                header = header.persist()
                mask = mask.persist()
                
        else:
            sciimgpath = io.get_file(filename, suffix="sciimg.fits",
                                         downloadit=download, **kwargs)
            mskimgpath = io.get_file(filenamemask, suffix="mskimg.fits",
                                         downloadit=download, **kwargs)
            data   = fits.getdata(sciimgpath)
            header = fits.getheader(sciimgpath)
            # - Mask
            mask   = fits.getdata(mskimgpath)

        # self
        meta = io.parse_filename(filename)
        this = cls(data=data, mask=mask, header=header, use_dask=use_dask, meta=meta)
        this._filename = filename
        return this
    
    def set_mask(self, mask):
        """ """
        self._mask = mask

    def load_wcs(self, header=None):
        """ """
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
        """ x and y or RA, Dec coordinates of the centroid. (shape[::-1]) """
        if system in ["xy","pixel","pixels","pxl"]:
            return (np.asarray(self.shape[::-1])+1)/2

        if system in ["uv","tangent"]:
            return np.squeeze(self.xy_to_uv(*self.get_center(system="xy")) )
        
        if system in ["radec","coords","worlds"]:
            return np.squeeze(self.xy_to_radec(*self.get_center(system="xy")) )

        raise ValueError(f"cannot parse the given system {system}, use xy, radec or uv")
    
    def get_ccd(self, use_dask=True, **kwargs):
        """ """
        return ScienceCCD.from_single_filename(self.filename, use_dask=use_dask, **kwargs)

    def get_focalplane(self, use_dask=True, **kwargs):
        """ fetchs the whole focal plan image (64 quadrants making 16 CCDs) and returns a ScienceFocalPlane object """
        return ScienceFocalPlane.from_single_filename(self.filename, use_dask=use_dask, **kwargs)
    
    def get_data(self, applymask=True, maskvalue=np.NaN,
                       rmbkgd=True, whichbkgd="median",
                       rebin=None, rebin_stat="nanmean",
                       **kwargs):
        """ get a copy of the data affected by background and/or masking.

        Parameters
        ---------
        applymask: [bool] -optional-
            Shall a default masking be applied (i.e. all bad pixels to nan)

        maskvalue: [float] -optional-
            Whick values should the masked out data have ?

        rmbkgd: [bool] -optional-
            Should the data be background subtracted ?

        whichbkgd: [bool] -optional-
            // ignored if rmbkgd=False //
            which background should this use (see self.get_background())

        **kwargs goes to self.get_mask()

        Returns
        -------
        2d array (data)

        """ 
        data_ = self.data.copy()
        
        if applymask:
            data_[self.get_mask(**kwargs)] = maskvalue # OK
            
        if rmbkgd:
            data_ -= self.get_background(method=whichbkgd, rmbkgd=False)

        if rebin is not None:
            data_ = getattr(da if self._use_dask else np, rebin_stat)(
                rebin_arr(data_, (rebin,rebin), use_dask=self._use_dask), axis=(-2,-1))

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

        from_source: [None/bool/DataFrame] -optional-
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
        alltrue: [bool] -optional-
            Short cut to set everything to true. Supposedly only background left

        flip_bits: [bool] -optional-
            This should be True to have the aforementioned masking proceedure.
            See astropy.nddata.bitmask.bitfield_to_boolean_mask

        verbose: [bool] -optional-
            Shall this print what you requested ?

        getflags: [bool]
            Get the bitmask power of 2 you requested instead of the actual masking.

        Returns
        -------
        boolean mask (or list of int, see getflags)
        """
        # // BitMasking
        npda = da if self._use_dask else np
        if alltrue and not getflags:
            return self.mask>0

        locals_ = locals()
        if verbose:
            print({k:locals_[k] for k in self.BITMASK_KEY})

        flags = [2**i for i,k in enumerate(self.BITMASK_KEY) if locals_[k] or alltrue]
        if getflags:
            return flags
        
        if self._use_dask:
            mask_ = dask.delayed(bitmask.bitfield_to_boolean_mask)(self.mask,
                                                        ignore_flags=flags, flip_bits=flip_bits)
            mask = da.from_delayed(mask_, self.shape, dtype="bool")
        else:
            mask = bitmask.bitfield_to_boolean_mask(self.mask,
                                                        ignore_flags=flags, flip_bits=flip_bits)
        # Rebin
        if rebin is not None:
            mask = getattr(da if self._use_dask else np, rebin_stat)(
                rebin_arr(mask, (rebin,rebin), use_dask=self._use_dask), axis=(-2,-1))

        return mask

    def get_background(self, method="median", rmbkgd=False, backup_default="sep"):
        """ get an estimation of the image's background

        Parameters
        ----------
        method: [string] -optional-
            if None, method ="default"
            - "default": returns the background store as self.background (see set_background)
            - "median": gets the median of the fully masked data (self.get_mask(alltrue=True))
            - "sep": returns the sep estimation of the background image (Sextractor-like)

        rmbkgd: [bool] -optional-
            // ignored if method != median //
            shall the median background estimation be made on default-background subtraced image ?

        backup_default: [string] -optional-
            If no background has been set yet, which method should be the default backgroud.
            If no background set and backup_default is None an AttributeError is raised.

        Returns
        -------
        float/array (see method)
        """
        if method not in ["median"]:
            raise NotImplementedError("Only median background implemented")
        
        if method in ["median"]:
            if self._use_dask:
                # median no easy to massively //
                return self.get_data(rmbkgd=rmbkgd, applymask=True, alltrue=True
                                      ).map_blocks(np.nanmedian)
            
            return np.nanmedian( self.get_data(rmbkgd=rmbkgd, applymask=True, alltrue=True) )

    def get_source_mask(self, thresh=2):
        """ """
        if not hasattr(self, "_source_mask"):
            from .tools import extract_sources, get_source_mask
            npda = da if self._use_dask else np
            datamasked = self.get_data(applymask=True, rmbkgd=True, whichbkgd="median", alltrue=True)
            noise = da.nanstd(datamasked)
            data = self.get_data(applymask=True, rmbkgd=True, whichbkgd="median")
            mask = self.get_mask()
            sources = extract_sources(data, thresh_=thresh, err=noise, mask=mask, use_dask=self._use_dask)
            self._source_mask = get_source_mask(sources, self.shape, use_dask=self._use_dask)
            
        return self._source_mask

    def get_source_background(self):
        """ """
        if not hasattr(self, "_source_mask"):
            from sep import Background
            data = self.data.copy()
            smask = self.get_source_mask()
            data[smask] = np.NaN
            if self._use_dask:
                self._source_background = da.from_delayed( dask.delayed(Background)(data).back(),
                                                shape = self.shape, dtype="float")
            else:
                self._source_background = Background(data).back()
                
        return self._source_background
        
    # -------- #
    # CATALOGS # 
    # -------- #
    def get_calibrators(self, which=["gaia","ps1"],
                            setxy=True, drop_outside=True, drop_namag=True,
                            pixelbuffer=10, isolation=None, mergehow="inner", seplimit=0.5,
                            use_dask=None, persist=False, **kwargs):
        """ get a DataFrame containing the requested calibrator catalog(s).
        If several catalog are given, a matching will be made and the dataframe merged (in)

        = implemented: gaia, ps1 = 

        Returns
        ------
        DataFrame
        """
        which = np.atleast_1d(which)
        if len(which)==0:
            raise ValueError("At least 1 catalog must be given")

        # Single Catalog
        if len(which) == 1:
            if which[0] == "gaia":
                return self.get_gaia_calibrators(setxy=setxy, drop_namag=drop_namag, drop_outside=drop_outside,
                                                 pixelbuffer=pixelbuffer,
                                                 isolation=isolation, use_dask=use_dask,
                                                 **kwargs)
            elif which[0] == "ps1":
                return self.get_ps1_calibrators( setxy=setxy, drop_outside=drop_outside,
                                                 pixelbuffer=pixelbuffer,
                                                 use_dask=use_dask,
                                                 **kwargs)
            else:
                raise ValueError(f"Only ps1 or gaia calibrator catalog have been implemented, {which} given.")
            
        # Two Catalogs
        if len(which) == 2:
            if which.tolist() in [["gaia","ps1"], ["ps1","gaia"]]:
                from .catalog import match_and_merge
                catps1  = self.get_ps1_calibrators( setxy=setxy,
                                                    drop_outside=drop_outside, pixelbuffer=pixelbuffer,
                                                    use_dask=use_dask,
                                                    **kwargs)
                catgaia = self.get_gaia_calibrators( setxy=setxy, drop_namag=drop_namag,isolation=isolation,
                                                     drop_outside=drop_outside, pixelbuffer=pixelbuffer,
                                                     use_dask=use_dask,
                                                     **kwargs)
                if use_dask is None:
                    use_dask = self._use_dask
                    
                if use_dask:
                    if persist:
                        catgaia = catgaia.persist()
                        catps1 = catps1.persist()
                    return dd.from_delayed(dask.delayed(match_and_merge)(
                        catgaia, catps1, suffixes=('', '_ps1'), mergehow=mergehow, seplimit=seplimit))
                
                return match_and_merge( catgaia, catps1,
                                        suffixes=('', '_ps1'), mergehow=mergehow,
                                        seplimit=seplimit)
            else:
                raise ValueError(f"Only ps1 and gaia calibrators catalog have been implemented, {which} given.")
            
            raise ValueError(f"Only single or pair or catalog (ps1 and/or gaia) been implemented, {which} given.")
        
    def get_ps1_calibrators(self, setxy=True, drop_outside=True, pixelbuffer=10, use_dask=None, **kwargs):
        """ """
        from .io import PS1Calibrators
        if use_dask is None:
            use_dask = self._use_dask
            
        # // Dask
        if use_dask:
            delayed_cat = dask.delayed(self.get_ps1_calibrators)( setxy=setxy, drop_outside=drop_outside,
                                                                  pixelbuffer=pixelbuffer,
                                                                  use_dask=False, # get the df
                                                                  **kwargs)
            columns = ['ra', 'dec', 'gmag', 'e_gmag',
                       'rmag', 'e_rmag', 'imag', 'e_imag',
                       'zmag', 'e_zmag']
            if setxy:
                columns += ["x","y","u","v"]
        
            meta = pandas.DataFrame(columns=columns,  dtype="float64")
            return dd.from_delayed(delayed_cat, meta=meta)

        #
        # Not Dasked
        #
        ps1cat = PS1Calibrators.fetch_data(self.rcid, self.fieldid, radec=self.get_center(system="radec"), **kwargs)
        if setxy and ("ra" in ps1cat.columns and "x" not in ps1cat.columns):
            ps1cat = self._setxy_to_cat_(ps1cat, drop_outside=drop_outside, pixelbuffer=pixelbuffer)

        return ps1cat

    def get_gaia_calibrators(self, setxy=True, drop_namag=True, drop_outside=True, pixelbuffer=10,
                                 isolation=None, use_dask=None, **kwargs):
        """ **kwargs goes to GaiaCalibrators (dl_wait for instance) 

        isolation: [None or positive float] -optional-
            self isolation limit (in arcsec). A True / False flag will be added to the catalog
            
        Returns
        -------
        DataFrame
        """
        from .io import GaiaCalibrators
        if use_dask is None:
            use_dask = self._use_dask
        
        if use_dask:
            delayed_cat = dask.delayed(self.get_gaia_calibrators)( setxy=setxy, drop_namag=drop_namag,
                                                                    drop_outside=drop_outside,
                                                                  pixelbuffer=pixelbuffer,
                                                                  isolation=isolation,
                                                                  use_dask=False, # get the df
                                                                  **kwargs)
            columns = ['Source', 'ps1_id', 'sdssdr13_id', 'ra', 'dec', 'gmag', 'e_gmag', 'gmagcorr',
                        'rpmag', 'e_rpmag', 'bpmag', 'e_bpmag', 'fg', 'e_fg', 'fgcorr', 'frp',
                        'e_frp', 'fbp', 'e_fbp', 'plx', 'e_plx', 'pm', 'pmra', 'e_pmra', 'pmde',
                        'e_pmde', 'colormag']
            spetypes = {"ps1_id":'string',"sdssdr13_id":'string'}
            if setxy:
                columns += ["x","y","u","v"]
            if isolation is not None:
                columns += ["isolated"]
                spetypes["isolated"] = "bool"
            meta = pandas.DataFrame(columns=columns,  dtype="float")
            meta = meta.astype(spetypes)            
            return dd.from_delayed(delayed_cat, meta=meta)
        #
        # Not Dasked
        #
        cat = GaiaCalibrators.fetch_data(self.rcid, self.fieldid, radec=self.get_center(system="radec"), **kwargs).reset_index()
        
        if drop_namag:
            cat = cat[~pandas.isna(cat[["gmag","rpmag","bpmag"]]).any(axis=1)]
        #cat[["ps1_id","sdssdr13_id"]] = cat[["ps1_id","sdssdr13_id"]].fillna("None")
        
        # Set mag as the current band magnitude
        if setxy and ("ra" in cat.columns and "x" not in cat.columns):
            cat = self._setxy_to_cat_(cat, drop_outside=drop_outside, pixelbuffer=pixelbuffer)
        
        if isolation is not None:
            from .catalog import get_isolated
            isolation = float(isolation)
            if isolation<=0:
                raise ValueError(f"isolation should be positive ; {isolation} given")

            cat["isolated"] = get_isolated(cat, seplimit=isolation)
            
        return cat.astype({"ps1_id":'string',"sdssdr13_id":'string'})
    
    # -------- #
    #  DASK    #
    # -------- #
    def _compute(self, client=None):
        """ """
        if client is not None:
            f_ = client.compute([self._header, self._data, self._mask])
            self._header, self._data, self._mask = client.gather(f_)
        else:
            self._header = self.header.compute()
            self._data = self._data.compute()
            self._mask = self._mask.compute()
        
        
    def _compute_data(self):
        """ """
        if self._use_dask and type(self._data) == DaskArray:
            self._data = self._data.compute()
            self._mask = self._mask.compute()

    # --------- #
    #  INTERNAL #
    # --------- #
    def _setxy_to_cat_(self, cat, drop_outside=True, pixelbuffer=10):
        """ """
        x,y = self.radec_to_xy(cat["ra"], cat["dec"])
        u,v = self.radec_to_uv(cat["ra"], cat["dec"])
        cat["x"] = x
        cat["y"] = y
        cat["u"] = u
        cat["v"] = v

        if drop_outside:
            ymax, xmax = self.shape
            cat = cat[cat["x"].between(+pixelbuffer, ymax-pixelbuffer) & \
                      cat["y"].between(+pixelbuffer, xmax-pixelbuffer)]
        return cat

    
    # =============== #
    #  Properties     #
    # =============== #
    @property
    def meta(self):
        """ """
        return self._meta
    
    @property
    def mask(self):
        """ """
        return self._mask

    @property
    def wcs(self):
        """ Astropy WCS solution loaded from the header """
        if not hasattr(self,"_wcs"):
            self.load_wcs()
        return self._wcs

    @property
    def filename(self):
        """ """
        if not hasattr(self,"_filename"):
            return None
        return self._filename

    # // shortcut
    @property
    def filtername(self):
        """ """
        return self.meta.filtercode
    
    @property
    def filterid(self):
        """ filter number (1: zg, 2: zr, 3:zi) see self.filtername """
        return self.meta.filterid

    @property
    def rcid(self):
        """ """
        return self.meta.rcid

    @property
    def ccdid(self):
        """ """
        return self.meta.ccdid

    @property
    def qid(self):
        """ """
        return self.meta.qid

    @property
    def fieldid(self):
        """ """
        return self.meta.field

    @property
    def filefracday(self):
        """ """
        return self.meta.filefracday

    @property
    def obsdate(self):
        """ YYYY-MM-DD"""
        return "-".join(self.meta[["year","month","day"]].values)
        
class ScienceCCD( _CCD_ ):
    SHAPE = 3080*2, 3072*2

    @classmethod
    def from_single_filename(cls, filename, use_dask=True, persist=True, **kwargs):
        """ """
        import re
        qids = range(1,5)
        scimg = [ScienceQuadrant.from_filename(re.sub(r"_o_q[1-5]*",f"_o_q{i}", filename),
                                                   use_dask=use_dask, persist=persist)
                     for i in qids]
        return cls(scimg, qids, use_dask=use_dask)
    

    # =============== #
    #  Properties     #
    # =============== #
    @property
    def meta(self):
        """ """
        if not hasattr(self, "_meta") or self._meta is None:
            self._meta = pandas.concat({i:q.meta for i,q in self.quadrants.items()}, axis=1).T
            
        return self._meta

    @property
    def meta(self):
        """ """
        if not hasattr(self, "_meta") or self._meta is None:
            self._meta = pandas.concat({i:q.meta for i,q in self.quadrants.items()}, axis=1).T
            
        return self._meta

    @property
    def filenames(self):
        """ """
        return [q.filenames for q in self.quandrants]
    
class ScienceFocalPlane( _FocalPlane_ ):
    """ """
    
    @classmethod
    def from_single_filename(cls, filename, use_dask=True, persist=True, **kwargs):
        """ """
        import re        
        ccdids = range(1,17)
        ccds = [ScienceCCD.from_single_filename(re.sub("_c(\d\d)_*",f"_c{i:02d}_",filename),
                                                    use_dask=use_dask, persist=persist, **kwargs)
                     for i in ccdids]
        return cls(ccds, ccdids, use_dask=use_dask)
    
    @property
    def meta(self):
        """ """
        if not hasattr(self, "_meta") or self._meta is None:
            self._meta = pandas.concat({i:ccd.meta for i,ccd in self.ccds.items()}).set_index("rcid")
            
        return self._meta
    
    @property
    def filenames(self):
        """ """
        return [q.filename  for ccdid,ccd in self.ccds.items() for qid,q in ccd.quadrants.items()]
