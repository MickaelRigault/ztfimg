import os
from astropy.io import fits
import numpy as np
import pandas
import warnings
import dask
import dask.array as da

from ztfquery import io
        
from .tools import fit_polynome, rebin_arr, parse_vmin_vmax
from .base import _Quadrant_, _CCD_, _FocalPlane_, Delayed
from .io import get_nonlinearity_table
from .collection import CCDCollection

NONLINEARITY_TABLE= get_nonlinearity_table()

class RawQuadrant( _Quadrant_ ):

    SHAPE_OVERSCAN = 3080, 30
    def __init__(self, data=None, header=None, overscan=None, use_dask=True):
        """ """
        _ = super().__init__(use_dask=use_dask)
        self.setup(data=data, header=header, overscan=overscan)
        
    @classmethod
    def from_filename(cls, filename, qid, use_dask=True,
                          persist=True, download=True, format=None, **kwargs):
        """ """
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
            return cls.read_fits(filename, qid, use_dask=use_dask, persist=persist, download=download, **kwargs)
        
        raise NotImplementedError("Only fits format implemented.")
            

    @classmethod
    def read_fits(cls, filename, qid, use_dask=True, persist=True, download=True, **kwargs):
        """ reads the fits file and load the object """
        if qid not in [1,2,3,4]:
            raise ValueError(f"qid must be 1,2, 3 or 4 {qid} given")
        
        if use_dask:
            filename  = dask.delayed(io.get_file)(filename, show_progress=False,
                                                  maxnprocess=1, download=download)
            data      = da.from_delayed( dask.delayed( fits.getdata )(filename, ext=qid),
                                            shape=cls.SHAPE, dtype="float")
            overscan  = da.from_delayed(dask.delayed(fits.getdata)(filename, ext=qid+4),
                                            shape=cls.SHAPE_OVERSCAN, dtype="float")
            header    = dask.delayed(fits.getheader)(filename, qid=qid)
            
        else:
            filename  = io.get_file(filename, show_progress=False, maxnprocess=1,
                                    download=download)
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
    def from_filefracday(cls, filefracday, rcid, use_dask=True, persist=True, **kwargs):
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
    def get_data(self, corr_overscan=False, corr_gain=False, corr_nl=False, rebin=None,
                     overscanprop={},
                     rebin_stat="nanmean", data="data", **kwargs):
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
        data_ = super().get_data(rebin=False, data=data, **kwargs)
        
        if corr_overscan:
            osmodel = self.get_overscan(**{**dict(which="model"),**overscanprop})
            data_ -= osmodel[:,None]
            
        if corr_gain:
            data_ *=self.gain

        if corr_nl:
            a, b = self.get_nonlinearity_corr()
            data_ /= (a*data_**2 + b*data_ + 1)

        if rebin is not None:
            data_ = getattr(da if self._use_dask else np, rebin_stat)(
                rebin_arr(data_, (rebin,rebin), use_dask=True), axis=(-2,-1) )
            
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
        
    def get_overscan(self, which="data", userange=[10,20], stackstat="nanmedian",
                         modeldegree=5, specaxis=1):
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
            - 'model': polynomial model of spec
            

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
        if which == "raw" or (which=="data" and userange is None):
            return self.overscan
        
        if which == "data":
            return self.overscan[:, userange[0]:userange[1]]
        
        if which == "spec":
            return getattr(np if not self._use_dask else da, stackstat
                          )( self.get_overscan(which="data", userange=userange), axis=specaxis )
        
        if which == "model":
            spec = self.get_overscan( which = "spec", userange=userange, stackstat=stackstat)
            if self._use_dask:
                d_ = dask.delayed(fit_polynome)(np.arange( len(spec) ), spec, degree=modeldegree)
                return da.from_delayed(d_, shape=spec.shape, dtype="float")
                
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
        
        
class RawCCD( _CCD_ ):

    def __init__(self, filename=None, use_dask=True, **kwargs):
        """ """
        _ = super().__init__(use_dask=use_dask)
        self.load_file(filename, **kwargs)
        
    @classmethod
    def from_filename(cls, filename, use_dask=True, **kwargs):
        """ """
        return cls(filename, use_dask=use_dask, **kwargs)

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
        
    # =============== #
    #   Methods       #
    # =============== #
    # -------- #
    #  LOADER  #
    # -------- #
    def load_file(self, filename, **kwargs):
        """ """
        if self._use_dask:
            filename = dask.delayed(io.get_file)(filename, show_progress=False, maxnprocess=1)
            header = dask.delayed(fits.open)(filename)[0].header
        else:
            filename = io.get_file(filename, show_progress=False, maxnprocess=1)
            header = fits.open(filename)[0].header
            
        self.set_header(header)
        self.load_quadrants(filename, format="fits", **kwargs)
        
    def load_quadrants(self, filename, which="*", persist=True, **kwargs):
        """ """
        if which is None or type(which)==str and which in ["*","all"] :
            which = [1,2,3,4]
        else:
            which = np.asarray(np.atleast_1d(which), dtype="int")
        
        for qid in which:
            qradrant = RawQuadrant.from_filename(filename, qid, use_dask=self._use_dask,
                                                     persist=persist, **kwargs)
            self.set_quadrant(qradrant,  qid=qid)
            
    # -------- #
    #  SETTER  #
    # -------- #                

    # --------- #
    #  GETTER   #
    # --------- # 
           
    # --------- #
    #  PLOTTER  #
    # --------- # 
        
    # =============== #
    #  Properties     #
    # =============== #
    @property
    def ccdid(self):
        """ number of the CCD """
        return self.get_headerkey("CCD_ID", None)
    
    @property
    def filename(self):
        """ """
        return self.get_headerkey("FILENAME", None)
    
    
class RawFocalPlane( _FocalPlane_ ):
    # INFORMATION || Numbers to be fine tuned from actual observations
    # 15 µm/arcsec  (ie 1 arcsec/pixel) and using 
    # 7.2973 mm = 487 pixel gap along rows (ie between columns) 
    # and 671 pixels along columns.
    @classmethod
    def from_filenames(cls, ccd_filenames, use_dask=True, **kwargs):
        """ """
        this = cls(use_dask=use_dask)
        for file_ in ccd_filenames:
            ccd_ = RawCCD.from_filename(file_, use_dask=use_dask, **kwargs)
            this.set_ccd(ccd_, ccdid=ccd_.ccdid)
            
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
        



class RawCCDCollection( CCDCollection ):
    
    @staticmethod
    def bulk_getfile_daterange(start, end, ccdid, imgtype=None, filtercode=None, dateformat=None, persist=False):
        """ """
        from astropy.time import Time
        from ztfquery import query
        if format != "jd":
            start, end = Time([start, end], format=dateformat).jd


        sql_query = f"ccdid = {ccdid} and obsjd between {start} and {end}"
        # IMGTYPE        
        if imgtype is not None:
            sql_query += f" and imgtype='{imgtype}'"
            
        # CCDID            
        if filtercode is not None:
            filtercode = tuple(np.atleast_1d(filtercode))
            sql_query += f" and filtercode IN {filtercode}"

        zquery = query.ZTFQuery.from_metaquery(kind="raw",   sql_query=sql_query)
        return io.bulk_get_file(zquery.get_data(exist=False), as_dask="persist" if persist else "delayed")
    
    # ------------ #
    #  INITIALISE  #
    # ------------ #
    @classmethod
    def from_filenames(cls, filenames, use_dask=True, imgkwargs={}, persist=True, **kwargs):
        """ """
        filenames = np.atleast_1d(filenames).tolist()
        if use_dask:
            images = [dask.delayed(RawCCD.from_filename)(filename, use_dask=False,
                                                                **imgkwargs)
                     for filename in filenames]
            if persist:
                images = [i.persist() for i in images]
        else:
            images = [RawCCD.from_filename(filename, use_dask=False, **imgkwargs)
                          for filename in filenames]
        
        return cls(images, use_dask=use_dask, **kwargs)
    
    @classmethod
    def from_date(cls, date, ccdid, dateformat=None, **kwargs):
        """ """
        start = Time(date, format=dateformat).jd
        return cls.from_daterange(start, start+1, dateformat="jd", ccdid=ccdid, **kwargs)
        
    @classmethod
    def from_daterange(cls, start, end, ccdid, filtercode=None, dateformat=None, persist=True, queryprop={}, **kwargs):
        """ """
        files = cls.bulk_getfile_daterange(start, end, ccdid, filtercode=filtercode, dateformat=dateformat, 
                                           persist=persist, **queryprop)
        return cls.from_filenames(files, use_dask=True, persist=persist, **kwargs)
    
    # ============ #
    #  Methods     #
    # ============ #
    # --------- #
    #  GETTER   #
    # --------- #    
    def get_data(self, corr_overscan=False, corr_nl=False, **kwargs):
        """ """
        return super().get_data(corr_overscan=corr_overscan, corr_nl=corr_nl, **kwargs)
    
class RawFlatCCDCollection( RawCCDCollection ):
    
    @classmethod
    def bulk_getfile_daterange(cls, start, end, ccdid, filtercode=["zr","zg","zi"],
                                   dateformat=None, persist=False,
                                   **kwargs):
        """ """
        return super().bulk_getfile_daterange(start, end, ccdid, filtercode=filtercode,
                                              dateformat=None, persist=False, imgtype="flat", 
                                              **kwargs)    
    
    # ============ #
    #  Methods     #
    # ============ #
    def split_by_led(self):
        """ splits the images by ILUM_LED number (from header) 
        and returns a dictionary containing {ledid: self.__class__}
        
        Returns
        -------
        dict
        """
        data = self.headerdf.compute().T
        leds = data.groupby("ILUM_LED").groups
        return {ledid:self.__class__.from_images( list(np.asarray(self.images)[index]) )
                for ledid, index in leds.items()}
