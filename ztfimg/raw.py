from astropy.io import fits
import numpy as np
import dask
import dask.array as da
from .tools import fit_polynome, rebin_arr, parse_vmin_vmax


class _RawImage_( object ):
    """ """
    def __init__(self, dasked=False):
        """ """
        self._use_dask = dasked
        
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
    def __init__(self, data=None, header=None, overscan=None, dasked=True):
        """ """
        _ = super().__init__(dasked=dasked)
        self.setup(data=data, header=header, overscan=overscan)
        
    @classmethod
    def from_filename(cls, filename, qid, grab_imgkeys=True, **kwargs):
        """ """
        if qid not in [1,2,3,4]:
            raise ValueError(f"qid must be 1,2, 3 or 4 {qid} given")
    
        imgkeys = ["EXPTIME", "IMGTYPE", "PIXSCALE", "THETA_X", "THETA_Y", "INST_ROT", 
                   "FILTER", "OBSJD", "RAD", "DECD", "TELRA","TELDEC", "AZIMUTH","ELVATION",
                   ]
        
        if "_f.fits" in filename:
            imgkeys += ["ILUM_LED", "ILUMWAVE", "ILUMPOWR"]
            
            
        data = fits.getdata(filename, ext=qid)
        header = fits.getheader(filename, ext=qid)
        imgheader = fits.getheader(filename, ext=0)
        if grab_imgkeys:
            for key in imgkeys:
                header.set(key, imgheader.get(key), imgheader.comments[key])
            
        overscan = fits.getdata(filename, ext=qid+4)
        if qid in [2, 3]:
            print("switching overscan")
            overscan = overscan[:,::-1]
            
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
    def get_data(self, which="data", overscanprop={}, corr_gain=False, corr_nl=False):
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
            data_ = self.data
        
        elif which == "data":
            osmodel = self.get_overscan(**{**dict(which="model"),**overscanprop})
            data_ = self.data - osmodel[:,None]
        else:
            raise ValueError(f"which can be 'raw' or 'data' ; {which} given")
            
        if corr_gain:
            data_ *=self.gain

        #if corr_nl:
            #data_ *=
            
        return data_

        
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
    def show_data(self, ax=None, colorbar=True, cax=None, apply=None, 
                  vmin="1", vmax="99", **kwargs):
        """ """
        import matplotlib.pyplot as mpl
        
        if ax is None:
            fig = mpl.figure(figsize=[5 + (1.5 if colorbar else 0),5])
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure
        
        if apply is not None:
            data = getattr(np,apply)(self.data)
        else:
            data = self.data

            
        vmin, vmax= parse_vmin_vmax(data, vmin, vmax)
        prop = dict(origin="lower", cmap="cividis", vmin=vmin,vmax=vmax)
            
            
        im = ax.imshow(data, **{**prop,**kwargs})
        if colorbar:
            fig.colorbar(im, cax=cax, ax=ax)
            
        return ax
    
    def show_overscan(self, ax=None, axs=None, axm=None, 
                      colorbar=False, cax=None, **kwargs):
        """ """
        import matplotlib.pyplot as mpl
        
        if ax is None:
            fig = mpl.figure(figsize=[4,6])
            ax  = fig.add_axes([0.15, 0.100, 0.58, 0.75])
            axs = fig.add_axes([0.75, 0.100, 0.20, 0.75])
            axm = fig.add_axes([0.15, 0.865, 0.58, 0.10])
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
        
        if axm is not None:
            overraw = self.get_overscan("raw")
            spec_to = np.nanmedian(overraw, axis=0)
            axm.plot(np.arange(len(spec_to)), spec_to)
            axm.set_xticks([])
            axm.set_xlim(*ax.get_xlim())
            
        if colorbar:
            fig.colorbar(im, cax=cax, ax=ax)
            
        return ax
        
    # =============== #
    #  Properties     #
    # =============== #
    @property
    def shape(self):
        return 3080, 3072    
    
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
        
        
class RawCCD( _RawImage_ ):

    def __init__(self, filename=None, dasked=True, **kwargs):
        """ """
        _ = super().__init__(dasked=dasked)
        self.load_file(filename, **kwargs)
        
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
    def load_file(self, filename, **kwargs):
        """ """
        raw = fits.open(filename)
        self.set_header(raw[0].header)
        self.load_quadrants(filename, **kwargs)
        
    def load_quadrants(self, filename, which="*", persist=True):
        """ """
        if which is None or type(which)==str and which in ["*","all"] :
            which = [1,2,3,4]
        else:
            which = np.asarray(np.atleast_1d(which), dtype="int")
        
        for qid in which:
            if self._use_dask:
                qradrant = dask.delayed(RawQuadrant.from_filename)(filename, qid)
                if persist:
                    qradrant = qradrant.persist()
            else:
                qradrant = RawQuadrant.from_filename(filename, qid)
                
            self.set_quadrant(qradrant,  qid=qid)
            
    def load_data(self, **kwargs):
        """  **kwargs goes to self.get_data() """
        self._data = self.get_data(**kwargs)
        
    # -------- #
    #  SETTER  #
    # -------- #                
    def set_quadrant(self, rawquadrant, qid=None, persist=True):
        """ """
        if qid is None:
            qid = rawquadrant.qid
            
        self.quadrants[qid] = rawquadrant
        
    # --------- #
    #  GETTER   #
    # --------- # 
    def get_quadrant(self, qid):
        """ """
        return self.quadrants[qid]
    
    def get_data(self, corr_overscan=True, corr_gain=True, 
                rebin=None, npstat="mean"):
        """ """
        ccd = np.zeros(self.shape)
        which = "data" if corr_gain else "raw"
        prop_qdata = dict(which=which, corr_gain=corr_gain)
        
        if self._use_dask:
            d = [da.from_delayed(self.get_quadrant(i).get_data(**prop_qdata),
                               shape=self.qshape, dtype="float")
                    for i in [1,2,3,4]]
            ccd_up   = da.concatenate([d[3],d[2]], axis=1)
            ccd_down = da.concatenate([d[0],d[1]], axis=1)
            ccd = da.concatenate([ccd_down,ccd_up], axis=0)
            if rebin is not None:
                ccd = getattr(da,npstat)( rebin_arr(ccd, (rebin,rebin), dasked=True), axis=(-2,-1))
        else:
            d = [self.get_quadrant(i).get_data(**prop_qdata)
                        for i in [1,2,3,4]]
            
            ccd_up   = np.concatenate([d[3],d[2]], axis=1)
            ccd_down = np.concatenate([d[0],d[1]], axis=1)
            ccd = np.concatenate([ccd_down,ccd_up], axis=0)
            if rebin is not None:
                ccd = getattr(np,npstat)( rebin_arr(ccd, (rebin,rebin), dasked=False), axis=(-2,-1))

        return ccd
        
    # --------- #
    #  PLOTTER  #
    # --------- # 
    def show_quadrants(self, **kwargs):
        """ """
        fig = mpl.figure(figsize=[6,6])
        ax1 = fig.add_axes([0.1,0.1,0.4,0.4])
        ax2 = fig.add_axes([0.5,0.1,0.4,0.4])
        ax3 = fig.add_axes([0.5,0.5,0.4,0.4])    
        ax4 = fig.add_axes([0.1,0.5,0.4,0.4])    

        prop = {**dict(colorbar=False),**kwargs}
        self.quadrants[1].show_data(ax=ax1, **prop)
        self.quadrants[2].show_data(ax=ax2, **prop)
        self.quadrants[3].show_data(ax=ax3, **prop)
        self.quadrants[4].show_data(ax=ax4, **prop) 

        [ax_.set_xticks([]) for ax_ in [ax3, ax4]]
        [ax_.set_yticks([]) for ax_ in [ax2, ax3]]
        
        
    def show(self, ax=None, vmin="1", vmax="99", colorbar=False, cax=None, 
             rebin=None, dataprop={}, **kwargs):
        """ """
        if ax is None:
            fig = mpl.figure(figsize=[6,6])
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure
        
        data  = self.get_data(rebin=rebin, **dataprop)
        vmin, vmax = parse_vmin_vmax(data, vmin, vmax)

        prop = {**dict(origin="lower", cmap="cividis", vmin=vmin, vmax=vmax),
                **kwargs}
        
        im = ax.imshow(data, **prop)
        
        if colorbar:
            fig.colorbar(im, cax=cax, ax=ax)
            
        return ax

    # =============== #
    #  Properties     #
    # =============== #
    @property
    def data(self):
        """ """
        if not hasattr(self, "_data"):
            if not self.has_quadrants("all"):
                return None
            
            self.load_data()
            
        return self._data
    
    @property
    def quadrants(self):
        """ """
        if not hasattr(self,"_quadrants"):
            self._quadrants = {k:None for k in [1,2,3,4]}
        return self._quadrants
    
    def has_quadrants(self, logic="all"):
        """ """
        is_none = [v is not None for v in self.quadrants.values()]
        return getattr(np, logic)(is_none)
        
    @property
    def shape(self):
        """ shape of ccd image """        
        return 3080*2, 3072*2
    
    @property
    def qshape(self):
        """ shape of an individual ccd quadrant """
        return 3080, 3072
    
    @property
    def ccdid(self):
        """ number of the CCD """
        return self.get_headerkey("CCD_ID", None)
    @property
    def filename(self):
        """ """
        return self.get_headerkey("FILENAME", None)
    
    
class RawFocalPlane( _RawImage_):
    # INFORMATION
    # 15 µm/arcsec  (ie 1 arcsec/pixel) and using 
    # 7.2973 mm = 487 pixel gap along rows (ie between columns) 
    # and 671 pixels along columns.
    @classmethod
    def from_filenames(cls, ccd_filenames, dasked=True, **kwargs):
        """ """
        this = cls(dasked=dasked)
        
        for file_ in ccd_filenames:
            ccd_ = RawCCD.from_filename(file_, dasked=dasked, **kwargs)
            this.set_ccd(ccd_, ccdid=ccd_.ccdid)
            
        return this
            
    # =============== #
    #   Methods       #
    # =============== #
    def set_ccd(self, rawccd, ccdid=None):
        """ """
        if ccdid is None:
            ccdid = rawccd.qid
            
        self.ccds[ccdid] = rawccd
        
    # --------- #
    #  GETTER   #
    # --------- # 
    def get_ccd(self, ccdid):
        """ """
        return self.ccds[ccdid]
    
    def get_quadrant(self, rcid):
        """ """
        ccdid, qid = self.rcid_to_ccdid_qid(rcid)
        return self.get_ccd(ccdid).get_quadrant(qid)
    
    # --------- #
    # CONVERTS  #
    # --------- # 
    @staticmethod
    def ccdid_qid_to_rcid(ccdid, qid):
        """ computes the rcid """
        return 4*(ccdid - 1) + qid - 1
    
    @staticmethod
    def rcid_to_ccdid_qid(rcid):
        """ computes the rcid """
        qid = (rcid%4)+1
        ccdid  = int((rcid-(qid - 1))/4 +1)
        return ccdid,qid
          
    # =============== #
    #  Properties     #
    # =============== #
    @property
    def ccds(self):
        """ """
        if not hasattr(self,"_ccds"):
            self._ccds = {k:None for k in np.arange(1,17)}
            
        return self._ccds
    
    def has_ccds(self, logic="all"):
        """ """
        is_none = [v is not None for v in self.ccds.values()]
        return getattr(np, logic)(is_none)
