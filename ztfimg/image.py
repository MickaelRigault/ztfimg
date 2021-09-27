""" """
import pandas
import numpy as np

from astropy.io import fits
from astropy.nddata import bitmask
from .io import PS1Calibrators, GaiaCalibrators
from . import tools


import dask
import dask.array as da

from dask.array.core import Array as DaskArray
from dask.delayed import Delayed
ZTF_FILTERS = {"ztfg":{"wave_eff":4813.97, "fid":1},
               "ztfr":{"wave_eff":6421.81, "fid":2},
               "ztfi":{"wave_eff":7883.06, "fid":3}
                }


from .astrometry import WCSHolder



print("ztfimg.image is DEPRECATED. See ztfimg.science (dasked version of it) ")

class ZTFImage( WCSHolder ):
    """ """
    SHAPE = 3080, 3072
    BITMASK_KEY = [ "tracks","sexsources","lowresponsivity","highresponsivity",
                    "noisy","ghosts","spillage","spikes","saturated",
                    "dead","nan","psfsources","brightstarhalo"]

    def __init__(self, imagefile=None, maskfile=None):
        """ """
        if imagefile is not None:
            self.load_data(imagefile)

        if maskfile is not None:
            self.load_mask(maskfile)

    @classmethod
    def fetch_local(cls):
        """ """
        print("To be done")

    # =============== #
    #  Methods        #
    # =============== #
    def query_associated_data(self, suffix=None, source="irsa", which="science",
                                  verbose=False, **kwargs):
        """ """
        from ztfquery import buildurl
        return getattr(buildurl,f"filename_to_{which}url")(self._filename, source=source,
                                                               suffix=suffix,
                                                               verbose=False, **kwargs)

    # -------- #
    # LOADER   #
    # -------- #
    def load_data(self, imagefile, **kwargs):
        """ """
        self._filename = imagefile
        self._data = fits.getdata(imagefile, **kwargs)
        self._header = fits.getheader(imagefile, **kwargs)

        

    def load_mask(self, maskfile, **kwargs):
        """ """
        self._mask = fits.getdata(maskfile,**kwargs)
        self._maskheader = fits.getheader(maskfile,**kwargs)
        

    def load_wcs(self, header=None):
        """ """
        if header is None:
            header = self.header
            
        super().load_wcs(header)

    def load_source_background(self, r=5, setit=True, datamasked=None, **kwargs):
        """
        kwargs goes to """
        from sep import Background
        if datamasked is None:
            if self.sources is None:
                from_sources = self.extract_sources(update=False, **kwargs)
            else:
                from_sources = self.sources

            datamasked = self.get_data(applymask=True, from_sources=from_sources,
                                        r=r, rmbkgd=False)

        self._sourcebackground = Background(datamasked.byteswap().newbyteorder())
        if setit:
            self.set_background(self._sourcebackground.back())

    def load_ps1_calibrators(self, setxy=True):
        """ """
        self.set_catalog( self.get_ps1_calibrators(setxy=setxy), "ps1cat")

    def load_gaia_calibrators(self, setxy=True):
        """ """
        self.set_catalog( self.get_gaia_calibrators(setxy=setxy), "gaia")

    # -------- #
    # SETTER   #
    # -------- #
    def set_background(self, background, cleardataclean=True):
        """
        Parameters
        ----------
        background: [array/float/str]
            Could be:
            array or float: this will be the background
            str: this will call get_background(method=background)

        """
        if type(background) == str:
            self._background = self.get_background(method=background)
        else:
            self._background = background

        if cleardataclean:
            self._dataclean = None

    def set_catalog(self, dataframe, label):
        """ """
        if "ra" not in dataframe.columns and "x" not in dataframe.columns:
            raise ValueError("The dataframe must contains either (x,y) coords or (ra,dec) coords")

        if "ra" in dataframe.columns and "x" not in dataframe.columns:
            x,y = self.radec_to_xy(dataframe["ra"], dataframe["dec"])
            dataframe["x"] = x
            dataframe["y"] = y

        if "x" in dataframe.columns and "ra" not in dataframe.columns:
            ra,dec = self.xy_to_radec(dataframe["x"], dataframe["y"])
            dataframe["ra"] = ra
            dataframe["dec"] = dec

        if "u" not in dataframe.columns:
            u, v = self.radec_to_uv(dataframe["ra"], dataframe["dec"])
            dataframe["u"] = u
            dataframe["v"] = v
            
        self.catalogs.set_catalog(dataframe, label)

    # -------- #
    # GETTER   #
    # -------- #
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

    def get_psfcat(self, show_progress=False, **kwargs):
        """ 
        psf-fit photometry catalog generated by the ztf-pipeline

        """
        from ztfquery import io
        from astropy.table import Table
        psffilename = io.get_file(self.filename, suffix="psfcat.fits",
                                  show_progress=show_progress, **kwargs)
        data = Table(fits.open(psffilename)[1].data).to_pandas().set_index("sourceid")
        # Not the same standard as calibrator cats.
        data[["xpos","ypos"]] -= 1 
        return data.rename({"xpos":"x", "ypos":"y"}, axis=1)

    def get_sexcat(self, show_progress=False, astable=False,**kwargs):
        """ 
        nested-aperture photometry catalog generated by the ztf-pipeline

        careful, nested apertures (MAG_APER, FLUX_APER and associated errors are droped to pandas.)

        """
        from ztfquery import io
        from astropy.table import Table
        psffilename = io.get_file(self.filename, suffix="sexcat.fits", show_progress=show_progress, **kwargs)
        tbl = Table(fits.open(psffilename)[1].data)#.to_pandas().set_index("sourceid")
        if astable:
            return tbl

        names = [name for name in tbl.colnames if len(tbl[name].shape) <= 1] 
        return tbl[names].to_pandas().set_index("NUMBER")

    def get_daophot_psf(self,  **kwargs):
        """ 
        PSF estimate at science image center as a FITS image generated by the ztf-pipeline
        """
        
        psffile = io.get_file(img.filename, "sciimgdaopsfcent.fits", show_progress=False)
        return fits.getdata(psffile)


    def get_catalog(self, calibrator=["gaia","ps1"], extra=["psfcat"], isolation=20, seplimit=0.5, **kwargs):
        """ **kwargs goes to get_calibrators """
        from .catalog import match_and_merge
        cal  = self.get_calibrators(calibrator, isolation=isolation, seplimit=seplimit, **kwargs)
        if "gaia" in np.atleast_1d(calibrator).tolist():
            onleft = "Source"
        else:
            raise NotImplementedError("calibrator should contain gaia in the current implementation.")
        
        extra = np.atleast_1d(extra).tolist()
        if "psfcat" in extra:
            psfcat = self.get_psfcat()
            return match_and_merge(cal, psfcat, "Source", mergehow="left", suffixes=('', '_psfcat'), seplimit=seplimit)
        
        return cal
        
    def get_calibrators(self, which=["gaia","ps1"],
                            setxy=True, drop_outside=True, drop_namag=True,
                            pixelbuffer=10, isolation=None, mergehow="inner", seplimit=0.5, **kwargs):
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
                                                 isolation=isolation, **kwargs)
            elif which[0] == "ps1":
                return self.get_ps1_calibrators(setxy=setxy, drop_outside=drop_outside, pixelbuffer=pixelbuffer, **kwargs)
            else:
                raise ValueError(f"Only ps1 or gaia calibrator catalog have been implemented, {which} given.")
            
        # Two Catalogs
        if len(which) == 2:
            if which.tolist() in [["gaia","ps1"], ["ps1","gaia"]]:
                from .catalog import match_and_merge
                catps1  = self.get_ps1_calibrators(setxy=setxy,
                                                       drop_outside=drop_outside, pixelbuffer=pixelbuffer, **kwargs)
                catgaia = self.get_gaia_calibrators(setxy=setxy, drop_namag=drop_namag,isolation=isolation,
                                                        drop_outside=drop_outside, pixelbuffer=pixelbuffer, **kwargs)

                return match_and_merge(catgaia.reset_index(),
                                           catps1.reset_index(),
                                           "Source", suffixes=('', '_ps1'), mergehow=mergehow,
                                           seplimit=seplimit)
            else:
                raise ValueError(f"Only ps1 and gaia calibrators catalog have been implemented, {which} given.")
            
            raise ValueError(f"Only single or pair or catalog (ps1 and/or gaia) been implemented, {which} given.")
        
        
    def get_ps1_calibrators(self, setxy=True, drop_outside=True, pixelbuffer=10, **kwargs):
        """ """
        # remark: radec is going to be used only the fieldid is not already downloaded.
        ps1cat = PS1Calibrators.fetch_data(self.rcid, self.fieldid, radec=self.get_center(system="radec"), **kwargs)
        
        # Set mag as the current band magnitude
        ps1cat['mag'] = ps1cat["%smag"%self.filtername[-1]]
        ps1cat['e_mag'] = ps1cat["e_%smag"%self.filtername[-1]]
        if setxy and ("ra" in ps1cat.columns and "x" not in ps1cat.columns):
            ps1cat = self._setxy_to_cat_(ps1cat, drop_outside=drop_outside, pixelbuffer=pixelbuffer)

        return ps1cat

    def get_gaia_calibrators(self, setxy=True, drop_namag=True, drop_outside=True, pixelbuffer=10,
                                 isolation=None, **kwargs):
        """ **kwargs goes to GaiaCalibrators (dl_wait for instance) 

        isolation: [None or positive float] -optional-
            self isolation limit (in arcsec). A True / False flag will be added to the catalog
            
        Returns
        -------
        DataFrame
        """
        cat = GaiaCalibrators.fetch_data(self.rcid, self.fieldid, radec=self.get_center(system="radec"), **kwargs)
        
        if drop_namag:
            cat = cat[~pandas.isna(cat[["gmag","rpmag","bpmag"]]).any(axis=1)]
        cat[["ps1_id","sdssdr13_id"]] = cat[["ps1_id","sdssdr13_id"]].fillna("None")
        
        # Set mag as the current band magnitude
        cat['mag'] = cat["gmag"]
        cat['e_mag'] = cat["e_gmag"]
        if setxy and ("ra" in cat.columns and "x" not in cat.columns):
            cat = self._setxy_to_cat_(cat, drop_outside=drop_outside, pixelbuffer=pixelbuffer)

        
        if isolation is not None:
            from .catalog import get_isolated
            isolation = float(isolation)
            if isolation<=0:
                raise ValueError(f"isolation should be positive ; {isolation} given")

            cat["isolated"] = get_isolated(cat, seplimit=isolation)
            
        return cat.astype({"ps1_id":'string',"sdssdr13_id":'string'})

    def get_data(self, applymask=True, maskvalue=np.NaN,
                       rmbkgd=True, whichbkgd="default", **kwargs):
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
        self._compute_data()
            
        data_ = self.data.copy()
        
        if applymask:
            data_[self.get_mask(**kwargs)] = maskvalue
            
        if rmbkgd:
            data_ -= self.get_background(method=whichbkgd, rmbkgd=False)

        return data_

    def get_mask(self, from_sources=None, **kwargs):
        """ get data mask

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

        #
        # anything else, self.mask is returned #
        #

        Returns
        -------
        2D array (True where should be masked out)
        """
        self._compute_data()

        # Source mask
        if from_sources is not None and from_sources is not False:
            if type(from_sources)== bool and from_sources:
                from_sources = self.sources
            elif type(from_sources) is not pandas.DataFrame:
                raise ValueError("cannot parse the given from_source could be bool, or DataFrame")

            from sep import mask_ellipse
            ellipsemask = np.asarray(np.zeros(self.shape),dtype="bool")
            # -- Apply the mask to falsemask
            mask_ellipse(ellipsemask, *from_sources[["x","y","a","b","theta"]].astype("float").values.T,
                         r=kwargs.get('r',5)
                         )
            return ellipsemask

        return self.mask

    def get_background(self, method=None, rmbkgd=False, backup_default="sep"):
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
        if (method is None or method in ["default"]):
            if not self.has_background():
                if backup_default is None:
                    raise AttributeError("No background set. Use 'method' or run set_background()")
                return self.get_background(backup_default)
            
            return self.background

        if method in ["median"]:
            return np.nanmedian( self.get_data(rmbkgd=rmbkgd, applymask=True, alltrue=True) )

        if method in ["sep","sextractor"]:
            return self.sourcebackground.back()

        raise NotImplementedError(f"method {method} has not been implemented. Use: 'median'")

    def get_noise(self, method="default", rmbkgd=True):
        """ get an estimation of the image's noise

        Parameters
        ----------
        method: [string/None] -optional-
            - None/default: become sep if a sourcebackground has been loaded, nmad otherwise.
            - nmad: get the median absolute deviation of self.data
            - sep: (float) global scatter estimated by sep (python Sextractor), i.e. rms for background subs image
            - std: (float) estimated as half of the counts difference between the 16 and 84 percentiles

        rmbkgd: [bool]
            // ignored if method != std //
            shall the std method be measured on background subtraced image ?

        Return
        ------
        float (see method)
        """
        if method is None or method in ["default"]:
            method = "sep" if hasattr(self,"_sourcebackground") else "nmad"

        if method in ["nmad"]:
            from scipy import stats
            data_ = self.get_data(applymask=False, rmbkgd=False)
            return stats.median_absolute_deviation(data_[~np.isnan(data_)])

        if method in ["std","16-84","84-16"]:
            data_ = self.get_data(rmbkgd=rmbkgd, applymask=True, alltrue=True)
            lowersigma,upsigma = np.percentile(data_[~np.isnan(data_)], [16,84]) # clean nans out
            return 0.5*(upsigma-lowersigma)

        if method in ["sep","sextractor", "globalrms"]:
            return self.sourcebackground.globalrms

        raise NotImplementedError(f"method {method} has not been implemented. Use: 'std'")

    def get_stamps(self, x0, y0, dx, dy=None, data="dataclean", asarray=False):
        """ Get a ztfimg.Stamp object or directly is data array
        """
        from .stamps import stamp_it
        return stamp_it( getattr(self,data), x0, y0, dx, dy=dy, asarray=asarray)
    
    def get_aperture(self, x0, y0, radius, bkgann=None, subpix=0, system="xy",
                         data="dataclean", maskprop={}, noiseprop={},
                         unit="counts", clean_flagout=False, get_flag=False):
        """ Get the Apeture photometry corrected from the background annulus if any.

        # Based on sep.sum_circle() #

        Parameters
        ----------
        x0, y0, radius: [array]
            Center coordinates and radius (radii) of aperture(s).
            (could be x,y, ra,dec or u,v ; see system)

        bkgann: [None/2D array] -optional-
            Length 2 tuple giving the inner and outer radius of a “background annulus”.
            If supplied, the background is estimated by averaging unmasked pixels in this annulus.

        subpix: [int] -optional-
            Subpixel sampling factor. If 0, exact overlap is calculated. 5 is acceptable.

        system: [string] -optional-
            In which system are the input x0, y0:
            - xy (ccd )
            - radec (in deg, sky)
            - uv (focalplane)

        data: [string] -optional-
            the aperture will be applied on self.`data`

        unit: [string] -optional-
            unit of the output | counts, flux and mag are accepted.

        clean_flagout: [bool] -optional-
            remove entries that are masked or partially masked
            (remove sum_circle flag!=0)
            = Careful, this does not affects the returned flags, len(flag) remains len(x0)=len(y0) = 
            
        get_flag: [bool]  -optional-
            shall this also return the sum_circle flags

        maskprop, noiseprop:[dict] -optional-
            options entering self.get_mask() and self.get_noise() for `mask` and `err`
            attribute of the sep.sum_circle function.
            

        Returns
        -------
        2D array (see unit: (counts, dcounts) | (flux, dflux) | (mag, dmag))
           + flag (see get_flag option)
        """
        from sep import sum_circle
        if unit not in ["counts","count", "flux", "mag"]:
            raise ValueError(f"Cannot parse the input unit. counts/flux/mag accepted {unit} given")

        if system == "radec":
            x0, y0 = self.radec_to_xy(x0, y0)
        elif system == "uv":
            x0, y0 = self.uv_to_xy(x0, y0)
        elif system != "xy":
            raise ValueError(f"system must be xy, radec or uv ;  {system} given")

        counts, counterr, flag = sum_circle(getattr(self,data).byteswap().newbyteorder(),
                                                        x0, y0, radius,
                                                        err=self.get_noise(**noiseprop),
                                                        mask=self.get_mask(**maskprop),
                                                        bkgann=bkgann, subpix=subpix)
        if clean_flagout:
            counts, counterr = counts[flag==0], counterr[flag==0]
            
        if unit in ["count","counts"]:
            if not get_flag:
                return counts, counterr
            return counts, counterr, flag
        if unit in ["flux"]:
            if not get_flag:
                return self.counts_to_flux(counts, counterr)
            return self.counts_to_flux(counts, counterr), flag
        if unit in ["mag"]:
            if not get_flag:
                return self.counts_to_mag(counts, counterr)
            return self.counts_to_mag(counts, counterr), flag

    def getcat_aperture(self, catdf, radius, xykeys=["x","y"], join=True, system="xy", **kwargs):
        """ measures the aperture (using get_aperture) using a catalog dataframe as input
        Parameters
        ----------
        catdf: [DataFrame]
            dataframe containing, at minimum the x and y centroid positions
    
        xykeys: [string, string] -optional-
            name of the x and y columns in the input dataframe
        
        join: [bool] -optional-
            shall the returned dataframe be a new dataframe joined 
            to the input one, or simply the aperture dataframe?
        
        **kwargs goes to get_aperture

        Returns
        -------
        DataFrame
        """
        x, y = catdf[xykeys].values.T
        flux, fluxerr, flag = self.get_aperture(x,y, radius[:,None],
                                                unit="counts", get_flag = True, system=system, **kwargs)
        dic = {**{f'f_{k}':f for k,f in enumerate(flux)},\
                   **{f'f_{k}_e':f for k,f in enumerate(fluxerr)},
                   **{f'f_{k}_f':f for k,f in enumerate(flag)}, # for each radius there is a flag
                   }

        fdata = pandas.DataFrame(dic, index=catdf.index) #gaia dataframe
        if join:
            return catdf.join(fdata)
        
        return fdata

    def get_center(self, system="xy"):
        """ x and y or RA, Dec coordinates of the centroid. (shape[::-1]) """
        if system in ["xy","pixel","pixels","pxl"]:
            return (np.asarray(self.shape[::-1])+1)/2

        if system in ["uv","tangent"]:
            return np.squeeze(self.xy_to_uv(*self.get_center(system="xy")) )
        
        if system in ["radec","coords","worlds"]:
            return np.squeeze(self.xy_to_radec(*self.get_center(system="xy")) )

        raise ValueError(f"cannot parse the given system {system}, use xy, radec or uv")

    def get_diagonal(self, inpixel=True):
        """ Get the size of the diagonal [[0,0]->[-1,-1]].
        If inpixel is False, it is given in degrees. """
        from astropy import units
        height, width = self.shape
        diagonal_pixels = np.sqrt(width**2+height**2)
        if inpixel:
            return diagonal_pixels
        
        return diagonal_pixels*self.pixel_scale/3600

    # -------- #
    # CONVERT  #
    # -------- #
    #    
    # WCS
    # pixel->
    def pixels_to_coords(self, x, y):
        """ get sky ra, dec [in deg] coordinates given the (x,y) ccd positions  """
        print("pixels_to_coords is DEPRECATED, use xy_to_radec")
        return self.xy_to_radec(x, y)

    # coords -> 
    def coords_to_pixels(self, ra, dec):
        """ get the (x,y) ccd positions given the sky ra, dec [in deg] corrdinates """
        print("coords_to_pixels is DEPRECATED, use radec_to_xy")
        return self.radec_to_xy(ra,dec)
    
    #
    # Flux - Counts - Mags
    def counts_to_mag(self, counts, dcounts=None):
        """ converts counts into flux [erg/s/cm2/A] """
        return tools.counts_to_mag(counts,dcounts, self.magzp, self.filter_lbda)

    def counts_to_flux(self, counts, dcounts=None):
        """ converts counts into flux [erg/s/cm2/A] """
        from . import tools
        return tools.counts_to_flux(counts,dcounts, self.magzp, self.filter_lbda)

    def flux_to_counts(self, flux, dflux=None):
        """ converts flux [erg/s/cm2/A] into counts """
        from . import tools
        return tools.flux_to_counts(flux, dflux, self.magzp, self.filter_lbda)

    def flux_to_mag(self, flux, dflux=None):
        """ converts flux [erg/s/cm2/A] into counts """
        from . import tools
        return tools.flux_to_mag(flux, dflux, wavelength=self.filter_lbda)

    def mag_to_counts(self, mag, dmag=None):
        """ """
        from . import tools
        return tools.mag_to_counts(mag, dmag, self.magzp, self.filter_lbda)

    def mag_to_flux(self, mag, dmag=None):
        """ """
        from . import tools
        return tools.mag_to_flux(mag, dmag, wavelength=self.filter_lbda)

    # -------- #
    #  MAIN    #
    # -------- #
    def extract_sources(self, thresh=2, err=None, mask=None, data="dataclean",
                              setradec=True, setmag=True,
                              update=True, **kwargs):
        """ uses sep.extract to extract sources 'a la Sextractor' """
        
        from sep import extract

        if err is None:
            err = self.get_noise()

        elif err in ["None"]:
            err = None

        if mask is None:
            mask = self.get_mask()
        elif mask in ["None"]:
            mask = None

        self._compute_data()
        sout = extract(getattr(self, data).byteswap().newbyteorder(),
                        thresh, err=err, mask=mask, **kwargs)

        _sources = pandas.DataFrame(sout)
        if setradec:
            ra, dec= self.pixels_to_coords(*_sources[["x","y"]].values.T)
            _sources["ra"] = ra
            _sources["dec"] = dec
        if setmag:
            _sources["mag"] = self.counts_to_mag(_sources["flux"], None)[0]
            # Errors to be added
        if not update:
            return _sources

        self.set_catalog(_sources, "sources")

    # -------- #
    # PLOTTER  #
    # -------- #
    def show(self, which="data", ax=None, show_ps1cal=False, vmin="1", vmax="99",
                 stretch=None, floorstretch=True, transpose=False,
                 colorbar=False, cax=None, clabel=None, clabelprop={}, **kwargs):
        """ """
        import matplotlib.pyplot as mpl
        if ax is None:
            fig = mpl.figure(figsize=[8,6])
            ax = fig.add_axes([0.1,0.1,0.8,0.8])
        else:
            fig = ax.figure

        # - Data
        self._compute_data()
            
        toshow_ = getattr(self,which)
        if transpose:
            toshow_ = np.transpose(toshow_)
        # - Colorstretching
        if stretch is not None:
            if floorstretch:
                toshow_ -=np.nanmin(toshow_)

            toshow_ = getattr(np,stretch)(toshow_)

        if type(vmin) == str:
            vmin = np.nanpercentile(toshow_,float(vmin))
        if type(vmax) == str:
            vmax = np.nanpercentile(toshow_,float(vmax))

        # - Properties
        defaultprop = dict(origin="lower", cmap="cividis",
                               vmin=vmin,
                               vmax=vmax,
                               )
        # - imshow
        im = ax.imshow(toshow_, **{**defaultprop, **kwargs})
        if colorbar:
            cbar = fig.colorbar(im, ax=ax, cax=cax)
            if clabel is not None:
                cbar.set_label(clabel, **clabelprop)
                
            
        # - overplot
        if show_ps1cal:
            xpos, ypos = self.coords_to_pixels(self.ps1calibrators["ra"],
                                              self.ps1calibrators["dec"])
            if transpose:
                xpos, ypos = ypos, xpos
            ax.scatter(xpos, ypos, marker=".", zorder=5,
                           facecolors="None", edgecolor="k",s=30,
                           vmin=0, vmax=2, lw=0.5)

            ax.set_xlim(0,self.data.shape[1])
            ax.set_ylim(0,self.data.shape[0])
        # - return
        return ax

    # -------- #
    #  DASK    #
    # -------- #
        
        
    # =============== #
    #  Properties     #
    # =============== #
    @property
    def data(self):
        """" Image data """
        return self._data

    @property
    def shape(self):
        """ Shape of the data """
        return self.SHAPE

    @property
    def datamasked(self):
        """" Image data """
        if not hasattr(self,"_datamasked"):
            self._datamasked = self.get_data(applymask=True, maskvalue=np.NaN, rmbkgd=False)

        return self._datamasked

    @property
    def dataclean(self):
        """ data background subtracted with bad pixels masked out (nan) """
        if not hasattr(self, "_dataclean") or self._dataclean is None:
            self._dataclean = self.get_data(applymask=True, maskvalue=np.NaN,
                                            rmbkgd=True, whichbkgd="default")
        return self._dataclean

    @property
    def sourcebackground(self):
        """ SEP (Sextractor in python) Background object.
        reload it using self.load_source_background(options)
        """
        if not hasattr(self,"_sourcebackground"):
            self.load_source_background()
        return self._sourcebackground

    @property
    def mask(self):
        """ Mask data associated to the data """
        if not hasattr(self,"_mask"):
            self._mask = np.asarray(np.zeros(self.shape), dtype='bool')
        return self._mask

    @property
    def background(self):
        """ Default background set by set_background, see also get_background() """
        if not hasattr(self,"_background"):
            return None
        return self._background

    def has_background(self):
        """ """
        return self.background is not None

    @property
    def catalogs(self):
        """ Dictionary containing the loaded catalogs """
        if not hasattr(self,"_catalogs"):
            from .catalog import CatalogCollection
            self._catalogs = CatalogCollection()
        return self._catalogs

    @property
    def ps1calibrators(self):
        """ PS1 calibrators used by IPAC """
        if "ps1cat" not in self.catalogs.labels:
            self.load_ps1_calibrators()
        return self.catalogs.catalogs["ps1cat"]

    @property
    def sources(self):
        """ Sources extracted using sep.extract """
        if "sources" not in self.catalogs.labels:
            return None
        return self.catalogs.catalogs["sources"]

    @property
    def header(self):
        """" """
        return self._header

    @property
    def wcs(self):
        """ Astropy WCS solution loaded from the header """
        if not hasattr(self,"_wcs"):
            self.load_wcs()
        return self._wcs

    def is_data_bad(self):
        """ """
        return self.header.get("STATUS") == 0

    @property
    def filename(self):
        """ """
        if not hasattr(self, "_filename"):
            return None
        return self._filename

    @property
    def basename(self):
        """ """
        filename =  self.filename
        return os.path.basename(self.filename) if filename is not None else None

    # // Header Short cut
    @property
    def filtername(self):
        """ """
        self._compute_header()
        return self.header.get("FILTER", None).replace("_","").replace(" ","").lower()

    @property
    def filter_lbda(self):
        """ effective wavelength of the filter """
        return ZTF_FILTERS[self.filtername]["wave_eff"]

    @property
    def pixel_scale(self):
        """ Pixel scale, in arcsec per pixel """
        self._compute_header()
        return self.header.get("PIXSCALE", self.header.get("PXSCAL", None) )

    @property
    def magzp(self):
        """ """
        self._compute_header()        
        return self.header.get("MAGZP", None)

    @property
    def maglim(self):
        """ 5 sigma magnitude limit """
        self._compute_header()        
        return self.header.get("MAGLIM", None)

    @property
    def saturation(self):
        """ """
        self._compute_header()
        return self.header.get("SATURATE", None)

    # -> IDs
    @property
    def rcid(self):
        """ """
        self._compute_header()        
        return self.header.get("RCID", self.header.get("DBRCID", None))

    @property
    def ccdid(self):
        """ """
        return int((self.rcid-(self.qid-1))/4 + 1)

    @property
    def qid(self):
        """ """
        return ( self.rcid%4 )+1

    @property
    def fieldid(self):
        """ """
        self._compute_header()
        return self.header.get("FIELDID", self.header.get("DBFIELD", None))

    @property
    def filterid(self):
        """ """
        self._compute_header()        
        return self.header.get("FILTERID", self.header.get("DBFID", None)) #science vs. ref

    
    
class ScienceImage( ZTFImage ):

    def __init__(self, imagefile=None, maskfile=None):
        """ """
        warnings.warn("ztfimg.image.ScienceImage is deprecated: use ztfimg.science.ScienceQuadrant ")
        if imagefile is not None:
            self.load_data(imagefile)

        if maskfile is not None:
            self.load_mask(maskfile)

    @classmethod
    def from_filename(cls, filename, filenamemask=None, download=True, **kwargs):
        """ 
        Parameters
        ----------
        download: [bool] -optional-
             Downloads the maskfile if necessary.

        **kwargs goes to ztfquery.io.get_file()
        """
        from ztfquery import io
        sciimgpath = io.get_file(filename, suffix="sciimg.fits", downloadit=download, **kwargs)
        mskimgpath = io.get_file(filename if filenamemask is None else filenamemask,
                                     suffix="mskimg.fits", downloadit=download, **kwargs)
        return cls(sciimgpath, mskimgpath)
        
    # -------- #
    #  LOADER  #
    # -------- #
    def load_source_background(self, bitmask_sources=True, datamasked=None, setit=True, **kwargs):
        """ """
        if datamasked is None and bitmask_sources:
            datamasked = self.get_data(rmbkgd=False, applymask=True, alltrue=True)
        return super().load_source_background(datamasked=datamasked, setit=setit, **kwargs)

    # -------- #
    #  GETTER  #
    # -------- #
    def get_mask(self, from_sources=None,
                       tracks=True, ghosts=True, spillage=True, spikes=True,
                       dead=True, nan=True, saturated=True, brightstarhalo=True,
                       lowresponsivity=True, highresponsivity=True, noisy=True,
                       sexsources=False, psfsources=False,
                       alltrue=False, flip_bits=True,
                       verbose=False, getflags=False,
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
        if from_sources is not None and from_sources is not False:
            return super().get_mask(from_sources=from_sources, **kwargs)

        # // BitMasking
        if alltrue and not getflags:
            return np.asarray(self.mask>0, dtype="bool")

        locals_ = locals()
        if verbose:
            print({k:locals_[k] for k in self.BITMASK_KEY})

        flags = [2**i for i,k in enumerate(self.BITMASK_KEY) if locals_[k] or alltrue]
        if getflags:
            return flags

        return bitmask.bitfield_to_boolean_mask(self.mask, ignore_flags=flags, flip_bits=flip_bits)

    # =============== #
    #  Properties     #
    # =============== #
    @property
    def exptime(self):
        """ """
        return self.header.get("EXPTIME", None)

    @property
    def obsjd(self):
        """ """
        return self.header.get("OBSJD", None)

    @property
    def obsmjd(self):
        """ """
        return self.header.get("OBSMJD", None)

    @property
    def _expid(self):
        """ """
        return self.header.get("EXPID", None)

class ReferenceImage( ZTFImage ):

    def __init__(self, imagefile=None):
        """ """
        if imagefile is not None:
            self.load_data(imagefile)

    # =============== #
    #  Properties     #
    # =============== #
    @property
    def background(self):
        """ Default background set by set_background, see also get_background() """
        if not hasattr(self,"_background"):
            return self.header.get("GLOBMED", 0)
        return self._background

    @property
    def filtername(self):
        """ """
        if not hasattr(self, "_filtername"):
            self._filtername = [k for k,v in ZTF_FILTERS.items() if v['fid']==self.filterid][0]
        return self._filtername

    @property
    def nframe(self):
        """  Number of frames used to build the reference image """
        return self.header.get('NFRAMES', None)
