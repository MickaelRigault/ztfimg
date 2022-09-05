import os
from astropy.io import fits
import numpy as np
from scipy import stats
import pandas
import warnings
import dask
import dask.array as da

from ztfquery import io
        
from .utils.tools import fit_polynome, rebin_arr, parse_vmin_vmax
from .base import Quadrant, CCD, FocalPlane, Delayed
from .io import get_nonlinearity_table

NONLINEARITY_TABLE= get_nonlinearity_table()

__all__ = ["RawQuadrant", "RawCCD", "RawFocalPlane"]


class RawQuadrant( Quadrant ):

    SHAPE_OVERSCAN = 3080, 30
    def __init__(self, data=None, header=None, overscan=None, use_dask=True):
        """ """
        # Add the overscan to the __init__
        _ = super().__init__(data=data, header=header, use_dask=use_dask)
        if overscan is not None:
            self.set_overscan(overscan)

    @classmethod
    def from_data(cls, data, header=None, overscan=None, **kwargs):
        """ """
        # the super knows overscan thanks to the kwargs passed to __init__
        return super().from_data(data, header=header, overscan=overscan,
                                     **kwargs)
    
    @classmethod
    def from_filename(cls, filename, qid,
                          as_path=False,
                          use_dask=True,
                          persist=False, download=True,
                          format=None, **kwargs):
        """ 

        as_path: [bool] -optional-
            if as_path is False, then rawfile=io.get_file(rawfile) is used.
            the enables to automatically download the missing file but work
            only for IPAC-pipeline based file. It add a (small) overhead.
            If you know the file exists, use as_path=True.

        """
        # - guessing format        
        if format is None:
            if type(filename) == Delayed:
                raise ValueError("cannot guess the input format from delayed filenames. Please provide the format.")
            
            if filename.endswith(".fits") or filename.endswith(".fits.gz") or filename.endswith(".fits.fz"):
                format = "fits"
            else:
                format = filename.split('.')
                
        # - Fits format
        if format in ["fits",".fits"]:
            return cls.read_fits(filename, qid,
                                    use_dask=use_dask,
                                    persist=persist, download=download,
                                    as_path=as_path,
                                     **kwargs)
        
        raise NotImplementedError("Only fits format implemented.")
            

    @classmethod
    def read_fits(cls, filename, qid, as_path=False,
                      use_dask=True, persist=False,
                      download=True, reorder=True, **kwargs):
        """ reads the fits file and load the object 

        as_path: [bool] -optional-
            if as_path is False, then rawfile=io.get_file(rawfile) is used.
            the enables to automatically download the missing file but work
            only for IPAC-pipeline based file. It add a (small) overhead.
            If you know the file exists, use as_path=True.

        """
        if qid not in [1,2,3,4]:
            raise ValueError(f"qid must be 1,2, 3 or 4 {qid} given")
        
        if use_dask:
            if not as_path:
                filename  = dask.delayed(io.get_file)(filename, show_progress=False,
                                                  maxnprocess=1, download=download)
                
            data      = da.from_delayed( dask.delayed( fits.getdata )(filename, ext=qid),
                                            shape=cls.SHAPE, dtype="float")
            overscan  = da.from_delayed(dask.delayed(fits.getdata)(filename, ext=qid+4),
                                            shape=cls.SHAPE_OVERSCAN, dtype="float")
            header    = dask.delayed(fits.getheader)(filename, qid=qid)
            
        else:
            if not as_path:
                filename  = io.get_file(filename, show_progress=False,
                                            maxnprocess=1, download=download)
            data      = fits.getdata(filename, ext=qid)
            overscan  = fits.getdata(filename, ext=qid+4)
            header    = fits.getheader(filename, qid=qid)

        if qid in [2, 3]:
            overscan = overscan[:,::-1]
        
        if persist and use_dask:
            data = data.persist()
            overscan = overscan.persist()
            header = header.persist()
            
        this = cls(data, header=header, overscan=overscan, use_dask=use_dask, **kwargs)
        this._qid = qid
        return this

    @classmethod
    def from_filefracday(cls, filefracday, rcid, use_dask=True, persist=False, **kwargs):
        """ """
        from ztfquery.io import filefracday_to_local_rawdata
        ccdid, qid = RawFocalPlane.rcid_to_ccdid_qid(rcid)
        
        filename = filefracday_to_local_rawdata(filefracday, ccdid=ccdid)
        
        if len(filename)==0:
            raise IOError(f"No local raw data found for filefracday: {filefracday} and ccdid: {ccdid}")
        
        if len(filename)>1:
            raise IOError(f"Very strange: several local raw data found for filefracday: {filefracday} and ccdid: {ccdid}", filename)
        
        return cls.from_filename(filename[0], qid=qid, use_dask=use_dask,
                                     persist=persist, **kwargs)
    
    @staticmethod
    def read_rawfile_header(filename, qid, grab_imgkeys=True):
        """ """
        imgkeys = ["EXPTIME", "IMGTYPE", "PIXSCALE", "THETA_X", "THETA_Y", "INST_ROT", 
                   "FILTER", "OBSJD", "RAD", "DECD", "TELRA","TELDEC", "AZIMUTH","ELVATION",
                   ]
        
        if "_f.fits" in filename:
            imgkeys += ["ILUM_LED", "ILUMWAVE", "ILUMPOWR"]

        header  = fits.getheader(filename, ext=qid)
        if grab_imgkeys:
            imgheader = fits.getheader(filename, ext=0)
            for key in imgkeys:
                header.set(key, imgheader.get(key), imgheader.comments[key])
            
        del imgheader
        # DataFrame to be able to dask it.
        return pandas.DataFrame( pandas.Series(header) )

    # -------- #
    #  SETTER  #
    # -------- #
    def set_overscan(self, overscan):
        """ """
        self._overscan = overscan
                    
    # -------- #
    # GETTER   #
    # -------- #
    def get_data(self, corr_overscan=False, corr_nl=False,
                     rebin=None, rebin_stat="nanmean", reorder=True,
                     overscanprop={},
                     **kwargs):
        """ 
        
        Parameters
        ----------
        corr_overscan, corr_nl: [bool] -optional-
            Should the data be corrected for non-linearity and
            overscan (if both are true, nl is applied first).

        rebin: [int / None] -optional-
            Shall the data be rebinned by square of size `rebin` ?
            None means no rebinning

        rebin_stat: [string] -optional-
            numpy (dask.array) method used for rebinning the data.
        
        reorder: [bool] -optional-
            Should the data be re-order to match the actual north-up.
            (leave to True if not sure)
            
        overscanprop: [dict] -optional-
            kwargs going to get_overscan()
            - > e.g. userange=[10,20], stackstat="nanmedian", modeldegree=5,
            
        Returns
        -------
        2d array
        """
        # rebin is made later on.
        data_ = super().get_data(rebin=None, **kwargs)
        
        if corr_nl:
            a, b = self.get_nonlinearity_corr()
            data_ /= (a*data_**2 + b*data_ + 1)
        
        if corr_overscan:
            osmodel = self.get_overscan(**{**dict(which="model"),**overscanprop})
            data_ -= osmodel[:,None]            

        if rebin is not None:
            data_ = getattr(da if self._use_dask else np, rebin_stat)(
                rebin_arr(data_, (rebin,rebin), use_dask=True), axis=(-2,-1) )

        if reorder:
            data_ = data_[::-1,::-1]
            
        return data_

    def get_nonlinearity_corr(self):
        """ looks in the raw.NONLINEARITY_TABLE the the entry corresponding to the quadrant's rcid
        and returns the a and b parameters. 
        
        raw data should be corrected as such:
        ```
        data_corr = data/(a*data**2 + b*data +1)
        ```
        Return
        ------
        data
        """
        return NONLINEARITY_TABLE.loc[self.rcid][["a","b"]].astype("float").values
        
    def get_overscan(self, which="data", clipping=True,
                         userange=[5,25], stackstat="nanmedian",
                         modeldegree=3, specaxis=1):
        """ 
        
        Parameters
        ----------
        which: [string] 
            could be:
            - 'raw': as stored
            - 'data': raw within userange 
            - 'spec': vertical or horizontal profile of the overscan
                      see stackstat
                      (see specaxis)
                      clipping is applied at that time (if clipping=True)

            - 'model': polynomial model of spec
            
        clipping: [bool]:
            Should clipping be applied to remove the obvious flux excess.
            This clipping is made using median statistics (median and 3*nmad)
            
        specaxis : [int] -optional-
            axis along which you are doing the median 
            = Careful: after userange applyed = 
            - axis: 1 (default) -> horizontal overscan data spectrum (~3000 pixels)
            - axis: 0 -> vertical stack of the overscan (~30 pixels)
            (see stackstat for stacking statistic (mean, median etc)
            
            
        stackstat: [string] -optional-
            numpy method to use to converting data into spec
            
        userange: [2d-array] 
            start and end of the raw overscan to be used.
                        
        Returns
        -------
        1 or 2d array (see which)

        Examples
        --------
        To get the raw overscan vertically stacked spectra, using mean statistic do:
        get_overscan('spec', userange=None, specaxis=0, stackstat='nanmean')


        """
        # = Raw as given
        if which == "raw" or (which=="data" and userange is None):
            return self.overscan
        
        # = Raw but cleaned
        if which == "data":
            data = self.overscan[:, userange[0]:userange[1]]                
            return data

        if which == "spec":
            data = self.get_overscan(which="data", userange=userange)
            func_to_apply = getattr(np if not self._use_dask else da, stackstat)
            spec = func_to_apply(data , axis=specaxis)
            if clipping:
                med_ = np.median( spec )
                mad_  = stats.median_absolute_deviation( spec )
                # Symetric to avoid bias, even though only positive outlier are expected.
                flag_out = (spec>(med_+3*mad_)) +(spec<(med_-3*mad_))
                spec[flag_out] = np.NaN
                #warnings.warn("overscan clipping is not implemented")
                
            return spec
        
        if which == "model":
            spec = self.get_overscan( which = "spec", userange=userange, stackstat=stackstat,
                                          clipping=clipping)
            # dask
            if self._use_dask:                
                d_ = dask.delayed(fit_polynome)(np.arange( len(spec) ), spec, degree=modeldegree)
                return da.from_delayed(d_, shape=spec.shape, dtype="float")
            # numpy
            return fit_polynome(np.arange(len(spec)), spec, degree=modeldegree)
        
        raise ValueError(f'which should be "raw", "data", "spec", "model", {which} given')    
        
    # -------- #
    # PLOTTER  #
    # -------- #
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
            spec_to = self.get_overscan("spec", userange=None, specaxis=0)
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
    def shape_overscan():
        """ shape of the raw overscan data """
        return self.SHAPE_OVERSCANE
    
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
        return self._qid if hasattr(self, "_qid") else (self.get_headerkey("AMP_ID")+1)
    
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
        
        
class RawCCD( CCD ):
    _QUADRANTCLASS = RawQuadrant
    
    @classmethod
    def from_filefracday(cls, filefracday, ccdid, use_dask=True, **kwargs):
        """ """
        from ztfquery.io import filefracday_to_local_rawdata
        filename = filefracday_to_local_rawdata(filefracday, ccdid=ccdid)
        if len(filename)==0:
            raise IOError(f"No local raw data found for filefracday: {filefracday} and ccdid: {ccdid}")
        if len(filename)>1:
            raise IOError(f"Very strange: several local raw data found for filefracday: {filefracday} and ccdid: {ccdid}", filename)
        
        return cls.from_filename(filename[0], use_dask=use_dask, **kwargs)

    
class RawFocalPlane( FocalPlane ):
    # INFORMATION || Numbers to be fine tuned from actual observations
    # 15 µm/arcsec  (ie 1 arcsec/pixel) and using 
    # 7.2973 mm = 487 pixel gap along rows (ie between columns) 
    # and 671 pixels along columns.
    _CCDCLASS = RawCCD
    
    @classmethod
    def from_filenames(cls, ccd_filenames, use_dask=True, **kwargs):
        """ """
        this = cls(use_dask=use_dask)
        for file_ in ccd_filenames:
            ccd_ = cls._CCDCLASS.from_filename(file_, use_dask=use_dask, **kwargs)
            this.set_ccd(ccd_, ccdid=ccd_.ccdid)

        this._filenames = ccd_filenames
        return this

    @classmethod
    def from_filefracday(cls, filefracday, use_dask=True, **kwargs):
        """ """
        from ztfquery.io import filefracday_to_local_rawdata
        filenames = filefracday_to_local_rawdata(filefracday, ccdid="*")
        if len(filenames)==0:
            raise IOError(f"No local raw data found for filefracday: {filefracday}")
        
        if len(filenames)>16:
            raise IOError(f"Very strange: more than 16 local raw data found for filefracday: {filefracday}", filename)
        
        if len(filenames)<16:
            warnings.warn(f"Less than 16 local raw data found for filefracday: {filefracday}")
        
        return cls.from_filenames(filenames, use_dask=use_dask, **kwargs)
            
    # =============== #
    #   Methods       #
    # =============== #
    # --------- #
    #  GETTER   #
    # --------- # 
        



