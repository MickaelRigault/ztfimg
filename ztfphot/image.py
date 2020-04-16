""" """

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

class ZTFImage( object ):
    """ """
    def __init__(self, imagefile=None, maskfile=None):
        """ """
        if imagefile is not None:
            self.load_data(imagefile)

        if maskfile is not None:
            self.load_mask(maskfile)
            
    @classmethod
    def fetch_local(cls):
        """ """
    # =============== #
    #  Methods        #
    # =============== #
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
        
    def load_ps1_calibrators(self):
        """ """
        from . import catalogs
        self._ps1calibrators = catalogs.PS1CalCatalog(self.rcid, self.fieldid)
        
    # -------- #
    # SETTER   #
    # -------- #
    # -------- #
    # GETTER   #
    # -------- #
    def get_data(self, applymask=True, maskvalue=np.NaN):
        """ """
        data_ = self.data.copy()
        data_[np.asarray(self.mask, dtype="bool")] = maskvalue
        return data_


    def get_mask(self,)
    def get_associated_data(self, suffix=None, source="irsa", which="science", verbose=False, **kwargs):
        """ """
        from ztfquery import buildurl
        return getattr(buildurl,f"filename_to_{which}url")(self._filename, source=source, suffix=suffix,
                                                               verbose=False, **kwargs)
    
    # -------- #
    #  WCS     #
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

    # -------- #
    # PLOTTER  #
    # -------- #
    def show(self, which="datamasked", show_ps1cal=True, **kwargs):
        """ """
        import matplotlib.pyplot as mpl
        fig = mpl.figure(figsize=[8,6])
        ax = fig.add_axes([0.1,0.1,0.8,0.8])
    
        defaultprop = dict(origin="lower", cmap="cividis", 
                               vmin=np.percentile(self.data,5),
                               vmax=np.percentile(self.data,95),
                               )
        ax.imshow(getattr(self,which), **{**defaultprop, **kwargs})
        if show_ps1cal:
            xpos, ypos = self.coords_to_pixels(self.ps1calibrators.data["ra"],
                                              self.ps1calibrators.data["dec"])
            ax.scatter(xpos, ypos, marker=".", zorder=5, 
                           facecolors="None", edgecolor="k",s=30,
                           vmin=0, vmax=2, lw=0.5)
            
            ax.set_xlim(0,self.data.shape[1])
            ax.set_ylim(0,self.data.shape[0])
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
    def mask(self):
        """ Mask data associated to the data """
        return self._mask
    
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
