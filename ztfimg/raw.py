""" """

from astropy.io import fits
import numpy as np


from .tools import fit_polynome

class _RawImage_( object ):
    """ """
    
    # -------- #
    #  SETTER  #
    # -------- #        
    def set_header(self, header):
        """ """
        self._header = header
    
    def set_data(self, data):
        """ """
        self._data = data
               
    # -------- #
    #  GETTER  #
    # -------- #    
    def get_headerkey(self, key, default=None):
        """ """
        if self.header is None:
            raise AttributeError("No header set. see the set_header() method")
            
        return self.header.get(key, default)

    # -------- #
    # PLOTTER  #
    # -------- #   
        
    # =============== #
    #  Properties     #
    # =============== #
    @property
    def data(self):
        """ """
        if not hasattr(self, "_data"):
            return None
        return self._data
        
    def has_data(self):
        """ """
        return self.data is not None
    
    @property
    def header(self):
        """ """
        if not hasattr(self, "_header"):
            return None
        return self._header
    
    @property
    def shape(self):
        """ """
        return None if not self.has_data() else np.shape(self.data)
    
    # // header 
    @property
    def filtername(self):
        """ """
        return self.get_headerkey("FILTER", "unknown")
    
    @property
    def exptime(self):
        """ """
        return self.get_headerkey("EXPTIME", np.NaN)
    
    def obsjd(self):
        """ """
        return self.get_headerkey("OBSJD", None)
    
        
class RawQuadrant( _RawImage_ ):
    """ """
    def __init__(self, data=None, header=None, overscan=None):
        """ """
        self.setup(data=data, header=header, overscan=overscan)
        
    @classmethod
    def from_filename(cls, filename, qid, grab_imgkeys=True, **kwargs):
        """ """
        if qid not in [1,2,3,4]:
            raise ValueError(f"qid must be 1,2, 3 or 4 {qid} given")
    
        imgkeys = ["EXPTIME", "IMGTYPE", "PIXSCALE", "THETA_X", "THETA_Y", "INST_ROT", 
                   "FILTER", "OBSJD", "RAD", "DECD", "TELRA","TELDEC", "AZIMUTH","ELVATION",
                   "ILUM_LED", "ILUMWAVE", "ILUMPOWR"]
            
            
        data = fits.getdata(filename, ext=qid)
        header = fits.getheader(filename, ext=qid)
        imgheader = fits.getheader(filename, ext=0)
        if grab_imgkeys:
            for key in imgkeys:
                header.set(key, imgheader.get(key), imgheader.comments[key])
            
        overscan = fits.getdata(filename, ext=qid+4)
        return cls(data, header=header,overscan=overscan, **kwargs)

    # -------- #
    #  SETTER  #
    # -------- #
    def setup(self, data=None, header=None, overscan=None):
        """ """
        if data is not None:
            self.set_data(data)
            
        if header is not None:
            self.set_header(header)
            
        if overscan is not None:
            self.set_overscan(overscan)
            
    def set_overscan(self, overscan):
        """ """
        self._overscan = overscan
                    
    # -------- #
    # GETTER   #
    # -------- #
    def get_data(self, which="data", overscanprop={}):
        """ 
        
        Parameters
        ----------
        which: [string] 
            could be:
            - 'raw': as stored
            - 'data': raw - {overscan model}
            
        overscanprop: [dict] -optional-
            kwargs going to get_overscan()
            (userange=[4,27], stackstat="nanmedian", modeldegree=5)
            
        Returns
        -------
        2d array
        """
        if which == "raw":
            return self.data
        
        if which == "data":
            osmodel = qradrant.get_overscan(**{**dict(which="model"),**overscanprop})
            return self.data - osmodel[:,None]
        
        raise ValueError(f"which can be 'raw' or 'data' ; {which} given")
        
    def get_overscan(self, which="image", userange=[4,27], stackstat="nanmedian", modeldegree=5):
        """ 
        
        Parameters
        ----------
        which: [string] 
            could be:
            - 'raw': as stored
            - 'data': raw within userange
            - 'spec': vertical profile of the overscan
                      see stackstat
            - 'model': polynomial model of spec
            
        stackstat: [string] -optional-
            numpy method to use to converting data into spec
            
        userange: [2d-array] 
            start and end of the raw overscan to be used.
            
        Returns
        -------
        1 or 2d array (see which)
        """
        if which == "raw":
            return self.overscan
        
        if which == "data":
            return self.overscan[:, userange[0]:userange[1]]
        
        if which == "spec":
            return getattr(np, stackstat)( self.get_overscan(which="data", userange=userange), axis=1 )
        
        if which == "model":
            spec = self.get_overscan( which = "spec", userange=userange, stackstat=stackstat)
            return fit_polynome(np.arange(len(spec)), spec, degree=modeldegree)
        
        raise ValueError(f'which should be "raw", "data", "spec", "model", {which} given')    
        
    # -------- #
    # PLOTTER  #
    # -------- #
    def show_data(self, ax=None, colorbar=True, cax=None, **kwargs):
        """ """
        import matplotlib.pyplot as mpl
        
        if ax is None:
            fig = mpl.figure(figsize=[7,7])
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure
        
        prop = dict(origin="lower", cmap="cividis")
        im = ax.imshow(self.data, **prop)
        if colorbar:
            fig.colorbar(im, cax=cax, ax=ax, **{**prop,**kwargs})
            
        return ax
    
    def show_overscan(self, ax=None, axs=None, colorbar=True, cax=None, **kwargs):
        """ """
        import matplotlib.pyplot as mpl
        
        if ax is None:
            fig = mpl.figure(figsize=[3,7])
            ax = fig.add_axes([0.1,0.1,0.65,0.8])
            axs = fig.add_axes([0.78,0.1,0.2,0.8])
        else:
            fig = ax.figure
            
        prop = dict(origin="lower", cmap="cividis", aspect="auto")
        im = ax.imshow(self.get_overscan("raw"), **{**prop,**kwargs})
        
        if axs is not None:
            spec = self.get_overscan("spec")
            model = self.get_overscan("model")
            axs.plot(spec, np.arange(len(spec)))
            axs.plot(model, np.arange(len(spec)))
            axs.set_yticks([])
            axs.set_ylim(*ax.get_ylim())
            
        if colorbar:
            fig.colorbar(im, cax=cax, ax=ax)
            
        return ax
        
    # =============== #
    #  Properties     #
    # =============== #
    @property
    def overscan(self):
        """ """
        if not hasattr(self, "_overscan"):
            return None
        return self._overscan
    
    @property
    def ccdid(self):
        """ """
        return self.get_headerkey("CCD_ID")
        
    @property
    def qid(self):
        """ """
        return self.get_headerkey("AMP_ID")+1
    
    @property
    def rcid(self):
        """ """
        return 4*(self.ccdid - 1) + self.qid - 1
    
    @property
    def gain(self):
        """ """
        return self.get_headerkey("GAIN", np.NaN)

    @property
    def darkcurrent(self):
        """ """
        return self.get_headerkey("DARKCUR", None)
    
    @property
    def readnoise(self):
        """ """
        return self.get_headerkey("READNOI", None)
        
        
class RawImage( _RawImage_ ):

    def __init__(self, filename=None):
        """ """
        self.load_file(filename)
        
    @classmethod
    def from_filename(cls, filename, **kwargs):
        """ """
        return cls(filename, **kwargs)

    # =============== #
    #   Methods       #
    # =============== #
    # -------- #
    #  LOADER  #
    # -------- #
    def load_file(self, filename):
        """ """
        raw = fits.open(filename)
        self.set_header(raw[0].header)
        
    # -------- #
    #  SETTER  #
    # -------- #                
    
    # =============== #
    #  Properties     #
    # =============== #
