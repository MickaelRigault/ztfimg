import numpy as np
import pandas
import dask
import dask.array as da

from dask.array.core import Array as DaskArray
from dask.delayed import Delayed

from .tools import rebin_arr, parse_vmin_vmax

class _Image_( object ):
    SHAPE = None
    # Could be any type (raw, science)

    def __init__(self, use_dask=True):
        """ """
        self._use_dask = use_dask
        
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
    def get_header(self):
        """ returns the header (self.header), see self.header
        """
        return self.header
    
    def get_headerkey(self, key, default=None):
        """ """
        if self.header is None:
            raise AttributeError("No header set. see the set_header() method")
            
        return self.header.get(key, default)

    # -------- #
    # PLOTTER  #
    # -------- #   
    def show(self, ax=None, colorbar=True, cax=None, apply=None, 
                  vmin="1", vmax="99", dataprop={}, savefile=None,
                  dpi=150, **kwargs):
        """ """
        import matplotlib.pyplot as mpl
        
        if ax is None:
            fig = mpl.figure(figsize=[5 + (1.5 if colorbar else 0),5])
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure

        data = self.get_data(**dataprop)
        if apply is not None:
            data = getattr(np,apply)(data)
            
        vmin, vmax= parse_vmin_vmax(data, vmin, vmax)
        prop = dict(origin="lower", cmap="cividis", vmin=vmin,vmax=vmax)
            
            
        im = ax.imshow(data, **{**prop,**kwargs})
        if colorbar:
            fig.colorbar(im, cax=cax, ax=ax)
            
        if savefile is not None:
            fig.savefig(savefile, dpi=dpi)
            
        return ax

    # -------- #
    # PLOTTER  #
    # -------- #   
    def _compute_header(self):
        """ """
        if self._use_dask and type(self._header) == Delayed:
            self._header = self._header.compute()
                
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
    def header(self, compute=True):
        """ """
        if not hasattr(self, "_header"):
            return None
        # Computes the header only if necessary
        self._compute_header()
        return self._header
    
    @property
    def shape(self):
        """ """
        return self.SHAPE
    
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

# -------------- #
#                #
#   QUANDRANT    #
#                #
# -------------- #

class _Quadrant_( _Image_ ):
    SHAPE = 3080, 3072

    def get_data(self, rebin=None, rebin_stat="nanmean"):
        """ 
        
        Parameters
        ----------

        Returns
        -------
        2d array
        """
        data_ = self.data
        
        if rebin is not None:
            data_ = getattr(da if self._use_dask else np, rebin_stat)(
                rebin_arr(data_, (rebin,rebin), use_dask=True), axis=(-2,-1))
            
        return data_

    def get_aperture(self, x0, y0, radius, bkgann=None, subpix=0, system="xy",
                         dataprop={},
                         mask=None, maskprop={},
                         err=None, noiseprop={}, use_dask=True,
                         ):
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
        from .tools import get_aperture
        
        if system == "radec":
            x0, y0 = self.radec_to_xy(x0, y0)
        elif system == "uv":
            x0, y0 = self.uv_to_xy(x0, y0)
        elif system != "xy":
            raise ValueError(f"system must be xy, radec or uv ;  {system} given")

        if err is None:
            err=self.get_noise(**noiseprop)
        if mask is None:
            mask=self.get_mask(**maskprop)
            
        return get_aperture(self.get_data(**dataprop),
                                x0, y0, radius=radius,
                                 err=err, mask=mask, use_dask=use_dask, **kwargs)
        

# -------------- #
#                #
#     CCD        #
#                #
# -------------- #
class _CCD_( _Image_ ):
    # Basically a Quadrant collection
    SHAPE = 3080*2, 3072*2

    def __init__(self, quadrants=None, qids=None, use_dask=True, **kwargs):
        """ """
        _ = super().__init__(use_dask=use_dask)
        if quadrants is not None:
            if qids is None:
                qids = [None]*len(quadrants)
            elif len(quadrants) != len(qids):
                raise ValueError("quadrants and qids must have the same size.")
            
            [self.set_quadrant(quad_, qid=qid, **kwargs)
                 for quad_, qid in zip(quadrants, qids)]

    def load_data(self, **kwargs):
        """  **kwargs goes to self.get_data() """
        self._data = self.get_data(**kwargs)
        
    # --------- #
    #  GETTER   #
    # --------- # 
    def set_quadrant(self, quadrant, qid=None):
        """ """
        if qid is None:
            qid = quadrant.qid
            
        self.quadrants[qid] = quadrant
        self._meta = None
        
    # --------- #
    #  GETTER   #
    # --------- # 
    def get_quadrant(self, qid):
        """ """
        return self.quadrants[qid]

    def get_quadrantheader(self):
        """ returns a DataFrame of the header quadrants """
        qid_range = [1,2,3,4]
        hs = [self.get_quadrant(i).get_header() for i in qid_range]
        df = pandas.concat(hs, axis=1)
        df.columns = qid_range
        return df
        
    def get_data(self, rebin=None, npstat="mean", rebin_ccd=None, persist=False, **kwargs):
        """ ccd data 
        
        rebin, rebin_ccd: [None, int]
            rebinning (based on restride) the data. 
            rebin affect the individual quadrants, while rebin_ccd affect the ccd. 
            then, rebin_ccd applies after rebin.
        """
        d = [self.get_quadrant(i).get_data(rebin=rebin, **kwargs) for i in [1,2,3,4]]

        # numpy or dask.array ?
        npda = da if self._use_dask else np 
        
        ccd_up   = npda.concatenate([d[3],d[2]], axis=1)
        ccd_down = npda.concatenate([d[0],d[1]], axis=1)
        ccd = npda.concatenate([ccd_down,ccd_up], axis=0)
        if rebin_ccd is not None:
            ccd = getattr(npda,npstat)( rebin_arr(ccd, (rebin_ccd, rebin_ccd), use_dask=self._use_dask),
                                              axis=(-2,-1))
        if self._use_dask and persist:
            return ccd.persist()
        
        return ccd
    # ----------- #
    #   PLOTTER   #
    # ----------- #
    def show(self, ax=None, vmin="1", vmax="99", colorbar=False, cax=None, 
             rebin=None, rebin_ccd=None, dataprop={}, savefile=None,
                 dpi=150, **kwargs):
        """ """
        import matplotlib.pyplot as mpl
        if ax is None:
            fig = mpl.figure(figsize=[6,6])
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure
        
        data  = self.get_data(rebin=rebin, rebin_ccd=rebin_ccd, **dataprop).compute()
        vmin, vmax = parse_vmin_vmax(data, vmin, vmax)

        prop = {**dict(origin="lower", cmap="cividis", vmin=vmin, vmax=vmax),
                **kwargs}
        
        im = ax.imshow(data, **prop)
        
        if colorbar:
            fig.colorbar(im, cax=cax, ax=ax)

        if savefile is not None:
            fig.savefig(savefile, dpi=dpi)
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
    def qshape(self):
        """ shape of an individual ccd quadrant """
        return 3080, 3072
    
# -------------- #
#                #
#  Focal Plane   #
#                #
# -------------- #

class _FocalPlane_( _Image_):

    # Basically a CCD collection
    def __init__(self, ccds=None, ccdids=None, use_dask=True, **kwargs):
        """ """
        _ = super().__init__(use_dask=use_dask)
         
        if ccds is not None:
            if ccdids is None:
                ccdids = [None]*len(quadrants)
            elif len(ccds) != len(ccdids):
                raise ValueError("ccds and ccdids must have the same size.")
            
            [self.set_ccd(ccd_, ccdid=ccdid_, **kwargs)
                 for ccd_, ccdid_ in zip(ccds, ccdids)]

    def set_ccd(self, rawccd, ccdid=None):
        """ """
        if ccdid is None:
            ccdid = rawccd.qid
            
        self.ccds[ccdid] = rawccd
        self._meta = None
        
    def get_ccd(self, ccdid):
        """ """
        return self.ccds[ccdid]
    
    def get_quadrant(self, rcid):
        """ """
        ccdid, qid = self.rcid_to_ccdid_qid(rcid)
        return self.get_ccd(ccdid).get_quadrant(qid)

    def get_quadrantheader(self, rcids="all"):
        """ returns a DataFrame of the header quadrants (rcid) """
        if rcids in ["*","all"]:
            rcids = np.arange(64)
            
        hs = [self.get_quadrant(i).get_header() for i in rcids]
        df = pandas.concat(hs, axis=1)
        df.columns = rcid_range
        return df

    @staticmethod
    def get_datagap(which, rebin=None, fillna=np.NaN):
        """ 
        horizontal (or row) = between rows
        """
        if which in ["horizontal", "row", "rows"]:
            hpixels = 672
            vpixels = 3072*2
        else:
            hpixels = 3080*2
            vpixels = 488

        if rebin is not None:
            hpixels /= rebin
            vpixels /= rebin
            
        return hpixels, vpixels

    def get_data(self, rebin=None, rebin_ccd=None, incl_gap=True, persist=False, **kwargs):
        """  """
        # Merge quadrants of the 16 CCDs
        prop = {**dict(rebin=rebin, rebin_ccd=rebin_ccd), **kwargs}

        npda = da if self._use_dask else np

        if not incl_gap:
            line_1 = getattr(npda,"concatenate")(( self.get_ccd(4).get_data(**prop), 
                                                 self.get_ccd(3).get_data(**prop), 
                                                 self.get_ccd(2).get_data(**prop), 
                                                 self.get_ccd(1).get_data(**prop)), axis=1)
            line_2 = getattr(npda,"concatenate")(( self.get_ccd(8).get_data(**prop), 
                                                 self.get_ccd(7).get_data(**prop), 
                                                 self.get_ccd(6).get_data(**prop), 
                                                 self.get_ccd(5).get_data(**prop)), axis=1)
            line_3 = getattr(npda,"concatenate")(( self.get_ccd(12).get_data(**prop), 
                                                self.get_ccd(11).get_data(**prop), 
                                                self.get_ccd(10).get_data(**prop), 
                                                self.get_ccd(9).get_data(**prop)), axis=1)
            line_4 = getattr(npda,"concatenate")(( self.get_ccd(16).get_data(**prop), 
                                                self.get_ccd(15).get_data(**prop), 
                                                self.get_ccd(14).get_data(**prop), 
                                                self.get_ccd(13).get_data(**prop)), axis=1)


            mosaic = getattr(npda,"concatenate")((line_1, line_2, line_3, line_4), axis=0)
        else:
            line_1 = getattr(npda,"concatenate")(( self.get_ccd(4).get_data(**prop), 
                                                  da.ones(self.get_datagap("columns", rebin=rebin))*np.NaN,
                                                 self.get_ccd(3).get_data(**prop), 
                                                  da.ones(self.get_datagap("columns", rebin=rebin))*np.NaN,
                                                 self.get_ccd(2).get_data(**prop), 
                                                  da.ones(self.get_datagap("columns", rebin=rebin))*np.NaN,
                                                 self.get_ccd(1).get_data(**prop)), axis=1)
            line_2 = getattr(npda,"concatenate")(( self.get_ccd(8).get_data(**prop), 
                                                  da.ones(self.get_datagap("columns", rebin=rebin))*np.NaN,
                                                 self.get_ccd(7).get_data(**prop), 
                                                  da.ones(self.get_datagap("columns", rebin=rebin))*np.NaN,
                                                 self.get_ccd(6).get_data(**prop), 
                                                  da.ones(self.get_datagap("columns", rebin=rebin))*np.NaN,
                                                 self.get_ccd(5).get_data(**prop)), axis=1)
            line_3 = getattr(npda,"concatenate")(( self.get_ccd(12).get_data(**prop), 
                                                  da.ones(self.get_datagap("columns", rebin=rebin))*np.NaN,
                                                self.get_ccd(11).get_data(**prop), 
                                                  da.ones(self.get_datagap("columns", rebin=rebin))*np.NaN,
                                                self.get_ccd(10).get_data(**prop), 
                                                  da.ones(self.get_datagap("columns", rebin=rebin))*np.NaN,
                                                self.get_ccd(9).get_data(**prop)), axis=1)
            line_4 = getattr(npda,"concatenate")(( self.get_ccd(16).get_data(**prop), 
                                                  da.ones(self.get_datagap("columns", rebin=rebin))*np.NaN,
                                                self.get_ccd(15).get_data(**prop), 
                                                  da.ones(self.get_datagap("columns", rebin=rebin))*np.NaN,
                                                self.get_ccd(14).get_data(**prop), 
                                                  da.ones(self.get_datagap("columns", rebin=rebin))*np.NaN,
                                                self.get_ccd(13).get_data(**prop)), axis=1)
            size_shape= self.get_datagap("rows", rebin=rebin)[0]

            mosaic = getattr(npda,"concatenate")((line_1, 
                                                  da.ones((size_shape, line_1.shape[1]))*np.NaN,
                                                  line_2, 
                                                  da.ones((size_shape, line_1.shape[1]))*np.NaN,                                              
                                                  line_3, 
                                                  da.ones((size_shape, line_1.shape[1]))*np.NaN,
                                                  line_4), axis=0)
        if self._use_dask and persist:
            return mosaic.persist()

        return mosaic



    def show(self, ax=None, vmin="1", vmax="99", colorbar=False, cax=None, 
             rebin=None, rebin_ccd=None, incl_gap=True, dataprop={},
             savefile=None, dpi=150,
             **kwargs):
        """ """
        import matplotlib.pyplot as mpl
        if ax is None:
            fig = mpl.figure(figsize=[6,6])
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure
        
        data  = self.get_data(rebin=rebin, rebin_ccd=rebin_ccd, incl_gap=incl_gap, **dataprop).compute()
        
        vmin, vmax = parse_vmin_vmax(data, vmin, vmax)

        prop = {**dict(origin="lower", cmap="cividis", vmin=vmin, vmax=vmax),
                **kwargs}
        
        im = ax.imshow(data, **prop)
        
        if colorbar:
            fig.colorbar(im, cax=cax, ax=ax)

        if savefile is not None:
            fig.savefig(savefile, dpi=dpi)

        return ax

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

    @property
    def shape_full(self):
        """ shape with gap"""
        print("gap missing")
        return self.ccdshape*4 
    
    @property
    def shape(self):
        """ shape without gap"""
        return self.ccdshape*4
    
    @property
    def ccdshape(self):
        """ """
        return self.qshape*2

    @property
    def qshape(self):
        """ """
        return np.asarray([3080, 3072])
