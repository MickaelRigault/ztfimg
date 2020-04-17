""" """
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import bitmask


ZTF_FILTERS = {"ZTF_g":{"wave_eff":4813.97},
               "ZTF_r":{"wave_eff":6421.81},
               "ZTF_i":{"wave_eff":7883.06}
                }


class ZTFImage( object ):
    """ """
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
    def query_associated_data(self, suffix=None, source="irsa", which="science", verbose=False, **kwargs):
        """ """
        from ztfquery import buildurl
        return getattr(buildurl,f"filename_to_{which}url")(self._filename, source=source, suffix=suffix,
                                                               verbose=False, **kwargs)
    
    # -------- #
    # LOADER   #
    # -------- #
    def load_data(self, imagefile, **kwargs):
        """ """
        self._filename = imagefile
        self._data = fits.getdata(imagefile,**kwargs)
        self._header = fits.getheader(imagefile,**kwargs)
        
    def load_mask(self, maskfile, **kwargs):
        """ """
        self._mask = fits.getdata(maskfile,**kwargs)
        self._maskheader = fits.getheader(maskfile,**kwargs)
        
    def load_wcs(self, header=None):
        """ """
        if header is None:
            header = self.header
        self._wcs = WCS(header)
        
    def load_ps1_calibrators(self, setxy=True):
        """ """
        from . import catalogs
        self._ps1calibrators = catalogs.PS1CalCatalog(self.rcid, self.fieldid)
        if setxy:
            x,y = self.coords_to_pixels(self.ps1calibrators.data["ra"], self.ps1calibrators.data["dec"])
            self._ps1calibrators.data["x"] = x
            self._ps1calibrators.data["y"] = y
            
        self._sources_ps1cat_match = None
        
    def load_sepbackground(self, **kwargs):
        """ """
        from sep import Background
        self._sepbackground = Background(self.get_data(rmbkgd=False, applymask=True,
                                                       alltrue=True).byteswap().newbyteorder(),
                                         **kwargs)

    def match_sources_and_ps1cat(self, ps1mag=None):
        """ """
        from . import matching
        ps1cat = self.ps1calibrators.data
        if ps1mag is None:
            ps1mag = "%smag"%self.filtername.split("_")[-1]
        ps1cat['mag'] = ps1cat[ps1mag]
        
        self._sources_ps1cat_match = matching.CatMatch.from_dataframe(self.extracted_sources, ps1cat)
        self._sources_ps1cat_match.match()

    # -------- #
    # SETTER   #
    # -------- #
    def set_background(self, background):
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
        self._dataclean = None
            
            
    # -------- #
    # GETTER   #
    # -------- #
    def get_data(self, applymask=True, maskvalue=np.NaN,
                       rmbkgd=True, whichbkgd="sep", **kwargs):
        """ """
        data_ = self.data.copy()
        if applymask:
            data_[self.get_mask(**kwargs)] = maskvalue
        if rmbkgd:
            data_ -= self.get_background(method=whichbkgd, rmbkgd=False)
            
        return data_

    def get_mask(self, tracks=True, ghosts=True, spillage=True, spikes=True,
                     dead=True, nan=True, saturated=True, brightstarhalo=True,
                     lowresponsivity=True, highresponsivity=True, noisy=True, 
                     sexsources=False, psfsources=False, 
                     alltrue=False, flip_bits=True, verbose=False, getflags=False):
        """ """
        if alltrue and not getflags:
            return np.asarray(self.mask>0, dtype="bool")
        
        locals_ = locals()
        if verbose:
            print({k:locals_[k] for k in self.BITMASK_KEY})

        flags = [2**i for i,k in enumerate(self.BITMASK_KEY) if locals_[k] or alltrue]
        if getflags:
            return flags
        
        return bitmask.bitfield_to_boolean_mask(self.mask, ignore_flags=flags, flip_bits=flip_bits)

    def get_background(self, method=None, rmbkgd=False):
        """ """
        if (method is None or method in ["default"]):
            if not self.has_background():
                raise AttributeError("No background set. Use 'method' or run set_background()")
            return self.background
        
        if method in ["median"]:
            return np.median( self.get_data(rmbkgd=rmbkgd, applymask=True, alltrue=True) )

        if method in ["sep","sextractor"]:
            return self.sepbackground.back()
        
        raise NotImplementedError(f"method {method} has not been implemented. Use: 'median'")
    
    def get_noise(self, method="std", rmbkgd=False):
        """ 
    
        Parameters
        ----------
        method: [string] -optional-
        std:
        """
        if method in ["std","16-84","84-16"]:
            lowersigma,upsigma = np.percentile(self.get_data(rmbkgd=rmbkgd, applymask=True,
                                                            alltrue=True), [16,84])
            return 0.5*(upsigma-lowersigma)
        
        if method in ["sep","sextractor", "globalrms"]:
            return self.sepbackground.globalrms
        
        raise NotImplementedError(f"method {method} has not been implemented. Use: 'std'")

    def get_sources_ps1cat_matched_entries(self, keys):
        """ keys could be e.g. ["ra","dec","mag"] """
        return self.sources_ps1cat_match.get_matched_entries(keys, "source","ps1")

    # -------- #
    # CONVERT  #
    # -------- #
    def coords_to_pixels(self, ra, dec):
        """ """
        return self.wcs.all_world2pix(np.asarray([np.atleast_1d(ra),
                                                  np.atleast_1d(dec)]).T,
                                      0).T
    def pixels_to_coords(self, x, y):
        """ """
        return self.wcs.all_pix2world(np.asarray([np.atleast_1d(x),
                                                  np.atleast_1d(y)]).T,
                                      0).T

    def count_to_mag(self, counts, dcounts=None):
        """ converts counts into flux """
        from . import tools
        return tools.count_to_mag(counts,dcounts, self.magzp, self.filter_lbda)
    
    def count_to_flux(self, counts, dcounts=None):
        """ converts counts into flux """
        from . import tools
        return tools.count_to_flux(counts,dcounts, self.magzp, self.filter_lbda)
    
    def flux_to_counts(self, flux, dflux=None):
        """ """
        from . import tools
        return tools.flux_to_count(flux, dflux, self.magzp, self.filter_lbda)

    def flux_to_mag(self, flux, dflux=None):
        """ """
        from . import tools
        return tools.flux_to_mag(flux, dflux, wavelength=self.filter_lbda)
    
    def mag_to_counts(self, mag, dmag=None):
        """ """
        from . import tools
        return tools.mag_to_count(mag, dmag, self.magzp, self.filter_lbda)

    def mag_to_flux(self, mag, dmag=None):
        """ """
        from . import tools
        return tools.mag_to_flux(mag, dmag, wavelength=self.filter_lbda)
    
    # -------- #
    #  MAIN    #
    # -------- #
    def extract_sources(self, thresh=5, err=None, mask=None, on="dataclean", setradec=True, setmag=True, **kwargs):
        """ uses sep.extract to extract sources 'a la Sextractor' """
        import pandas
        from sep import extract
        
        if err is None:
            err = self.get_noise("sep")
        elif err in ["None"]:
            err = None
            
        if mask is None:
            mask = self.get_mask()
        elif mask in ["None"]:
            mask = None

        
        sout = extract(getattr(self,on).byteswap().newbyteorder(),
                        thresh, err=err, mask=mask, **kwargs)
        
        _extracted_sources = pandas.DataFrame(sout)
        if setradec:
            ra, dec= self.pixels_to_coords(*_extracted_sources[["x","y"]].values.T)
            _extracted_sources["ra"] = ra
            _extracted_sources["dec"] = dec
        if setmag:
            _extracted_sources["mag"] = self.count_to_mag(_extracted_sources["flux"], None)[0]
            # Errors to be added
            
        self._extracted_sources = _extracted_sources
        self._sources_ps1cat_match = None
    # -------- #
    # PLOTTER  #
    # -------- #
    def show(self, which="data", ax=None, show_ps1cal=True, vmin="1", vmax="99",
                 stretch=None, floorstretch=True, **kwargs):
        """ """
        import matplotlib.pyplot as mpl
        if ax is None:
            fig = mpl.figure(figsize=[8,6])
            ax = fig.add_axes([0.1,0.1,0.8,0.8])
        else:
            fig = ax.figure
            
        # - Data
        toshow_ = getattr(self,which)

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
        ax.imshow(toshow_, **{**defaultprop, **kwargs})
        
        # - overplot        
        if show_ps1cal:
            xpos, ypos = self.coords_to_pixels(self.ps1calibrators.data["ra"],
                                              self.ps1calibrators.data["dec"])
            ax.scatter(xpos, ypos, marker=".", zorder=5, 
                           facecolors="None", edgecolor="k",s=30,
                           vmin=0, vmax=2, lw=0.5)
            
            ax.set_xlim(0,self.data.shape[1])
            ax.set_ylim(0,self.data.shape[0])
        # - return
        return ax
    
    # =============== #
    #  Properties     #
    # =============== #
    @property
    def data(self):
        """" Image data """
        return self._data

    @property
    def datamasked(self):
        """" Image data """
        if not hasattr(self,"_datamasked"):
            self._datamasked = self.get_data(applymask=True, maskvalue=np.NaN)
            
        return self._datamasked

    @property
    def dataclean(self):
        """ data background subtracted with bad pixels masked out (nan) """
        if not hasattr(self, "_dataclean") or self._dataclean is None:
            self._dataclean = self.get_data(applymask=True, maskvalue=np.NaN,
                                            rmbkgd=True, whichbkgd="default")
        return self._dataclean
    @property
    def sepbackground(self):
        """ SEP (Sextractor in python) Background object. 
        reload it using self.load_sepbackground(options) 
        """
        if not hasattr(self,"_sepbackground"):
            self.load_sepbackground()
        return self._sepbackground
    
    @property
    def mask(self):
        """ Mask data associated to the data """
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
    def extracted_sources(self):
        """ Sources extracted using sep.extract """
        if not hasattr(self,"_extracted_sources"):
            return None
        return self._extracted_sources

    @property
    def sources_ps1cat_match(self):
        """ ztfphot.CatMatch between the extracted sources and PS1Cat """
        if not hasattr(self,"_sources_ps1cat_match") or self._sources_ps1cat_match is None:
            self.match_sources_and_ps1cat()
            
        return self._sources_ps1cat_match
    
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
    def ps1calibrators(self):
        """ PS1 calibrators used by IPAC """
        if not hasattr(self, "_ps1calibrators"):
            self.load_ps1_calibrators()
        return self._ps1calibrators



    
    # // Header Short cut
    @property
    def exptime(self):
        """ """
        return self.header.get("EXPTIME", None)
    
    @property
    def filtername(self):
        """ """
        return self.header.get("FILTER", None)
    
    @property
    def filter_lbda(self):
        """ effective wavelength of the filter """
        return ZTF_FILTERS[self.filtername]["wave_eff"]
    
    @property
    def pixel_scale(self):
        """ """
        return self.header.get("PIXSCALE", None)
    
    @property
    def seeing(self):
        """ """
        return self.header.get("SEEING", None)
    
    @property
    def obsjd(self):
        """ """
        return self.header.get("OBSJD", None)
    
    @property
    def obsmjd(self):
        """ """
        return self.header.get("OBSMJD", None)
    
    @property
    def magzp(self):
        """ """
        return self.header.get("MAGZP", None)
    
    @property
    def maglim(self):
        """ 5 sigma magnitude limit """
        return self.header.get("MAGLIM", None)
    
    @property
    def saturation(self):
        """ """
        return self.header.get("SATURATE", None)        
    
    # -> IDs
    @property
    def ccdid(self):
        """ """
        return self.header.get("CCDID", None)
    
    @property
    def qid(self):
        """ """
        return self.header.get("QID", None)
    
    @property
    def rcid(self):
        """ """
        return self.header.get("RCID", None)

    @property
    def fieldid(self):
        """ """
        return self.header.get("FIELDID",None)
    
    @property
    def filterid(self):
        """ """
        return self.header.get("FILTPOS",None)

    @property
    def _expid(self):
        """ """
        return self.header.get("EXPID", None)
    
    @property
    def _framenum(self):
        """ """
        return self.header.get("FRAMENUM", None)
