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
from .base import Quadrant, CCD, FocalPlane
from .io import get_nonlinearity_table

NONLINEARITY_TABLE= get_nonlinearity_table()

__all__ = ["RawQuadrant", "RawCCD", "RawFocalPlane"]


class RawQuadrant( Quadrant ):

    SHAPE_OVERSCAN = 3080, 30
    def __init__(self, data=None, header=None, overscan=None):
        """ 
        See also
        --------
        from_filename: load the instance given a filename 
        from_data: load the instance given its data (and header)
        """
        # Add the overscan to the __init__
        _ = super().__init__(data=data, header=header)
        if overscan is not None:
            self.set_overscan(overscan)


    @classmethod
    def _read_overscan(cls, filename, ext, use_dask=True, persist=False):
        """ assuming fits format. """
        from astropy.io.fits import getdata
        
        if use_dask:
            overscan = da.from_delayed(dask.delayed(getdata)(filename, ext=ext+4),
                                            shape=cls.SHAPE_OVERSCAN, dtype="float32")
            if persist:
                overscan = overscan.persist()
        else:
            overscan = getdata(filename, ext=ext+4)
            
        return overscan            

    @classmethod
    def from_data(cls, data, header=None, overscan=None, **kwargs):
        """ Instanciate this class given data. 
        
        Parameters
        ----------
        data: numpy.array or dask.array]
            Data of the Image.
            this will automatically detect if the data are dasked.

        header: fits.Header or dask.delayed
            Header of the image.

        overscan: 2d-array
            overscan image.

        **kwargs goes to __init__

        Returns
        -------
        class instance
        """
        # the super knows overscan thanks to the kwargs passed to __init__
        use_dask = "dask" in str( type(data))
        return super().from_data(data, header=header, overscan=overscan,
                                 **kwargs)
    
    @classmethod
    def from_filename(cls, filename, qid,
                          as_path=True,
                          use_dask=False, persist=False,
                          dask_header=False,
                          **kwargs):
        """ classmethod load an instance given an input file.

        Parameters
        ----------
        filename: str
            fullpath or filename or the file to load. This must be a raw ccd file.
            If a filename is given, set as_path=False, then ztfquery.get_file() 
            will be called to grab the file for you (and download it if necessary)
            
        qid: int
            quadrant id. Which quadrant to load from the input raw image ?
            
        as_path: bool
            Set this to true if the input file is not the fullpath but you need
            ztfquery.get_file() to look for it for you.
        
        use_dask: bool
            Should dask be used ? The data will not be loaded but delayed 
            (dask.array)

        persist: bool
            = only applied if use_dask=True =
            should we use dask's persist() on data ?

        dask_header: bool, optional
            should the header be dasked too (slows down a lot)

        **kwargs: goes to __init__()

        Returns
        -------
        class instance     
        """
        # - guessing format        
        if qid not in [1,2,3,4]:
            raise ValueError(f"qid must be 1,2, 3 or 4 {qid} given")

        if not use_dask:
            dask_header = False

        meta = io.parse_filename(filename)
        filename = cls._get_filename(filename, as_path=as_path, use_dask=use_dask)
        # data
        data = cls._read_data(filename, ext=qid, use_dask=use_dask, persist=persist)
        header = cls._read_header(filename, ext=qid, use_dask=dask_header, persist=persist)
        # and overscan
        overscan = cls._read_overscan(filename, ext=qid, use_dask=use_dask, persist=persist)
        
        this = cls(data, header=header, overscan=overscan, **kwargs)
        this._qid = qid
        this._filename = filename
        this._meta = meta
        return this

    @classmethod
    def from_filefracday(cls, filefracday, rcid, use_dask=True, persist=False, **kwargs):
        """ load the instance given a filefracday and the rcid (ztf ID)

        Parameters
        ----------
        filefracday: str
            ztf ID of the exposure (YYYYMMDDFFFFFF) like 20220704387176
            ztfquery will fetch for the corresponding data.

        rcid: int
            rcid of the given quadrant

        use_dask: bool
            Should dask be used ? The data will not be loaded but delayed 
            (dask.array)

        persist: bool
            = only applied if use_dask=True =
            should we use dask's persist() on data ?

        **kwargs goes to from_filename -> __init__

        Returns
        -------
        class instance
        
        """
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
        """ reads the filename's header and returns it as a pandas.DataFrame

        Parameters
        ----------
        filename: str
            path of the data file.

        qid: int
            quadrant id for the header you want.

        grab_imgkeys: bool
            should the gobal image header data also be included
            (i.e. header from both ext=0 and ext=qid

        Returns
        -------
        `pandas.DataFrame`
            the header
        """
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
        """ set the overscan image.
        
        = It is unlikely you need to use that directly. =
        
        Parameters
        ----------
        overscan: 2d-array
            overscan 2d-image

        Returns
        -------
        None
        """
        self._overscan = overscan
                    
    # -------- #
    # GETTER   #
    # -------- #
    def get_data(self, corr_overscan=False, corr_nl=False,
                     rebin=None, rebin_stat="nanmean", reorder=True,
                     overscan_prop={}, **kwargs):
        """ get the image data. 

        returned data can be affected by different effects.
        
        Parameters
        ----------
        corr_overscan: bool
            Should the data be corrected for overscan
            (if both corr_overscan and corr_nl are true, 
            nl is applied first)

        corr_nl: bool
            Should data be corrected for non-linearity
            
        rebin: int, None
            Shall the data be rebinned by square of size `rebin` ?
            None means no rebinning.
            (see details in rebin_stat)
            rebin must be a multiple of the image shape.
            for instance if the input shape is (6160, 6144)
            rebin could be 2,4,8 or 16

        rebin_stat: str
            = applies only if rebin is not None =
            numpy (dask.array) method used for rebinning the data.
            For instance, if rebin=4 and rebin_stat = median
            the median of a 4x4 pixel will be used to form a new pixel.
            The dimension of the final image will depend on rebin.
        
        reorder: bool
            Should the data be re-order to match the actual north-up.
            (leave to True if not sure)
            
        overscan_prop: [dict] -optional-
            kwargs going to get_overscan()
            - > e.g. userange=[10,20], stackstat="nanmedian", modeldegree=5,
            
        Returns
        -------
        2d-array
            numpy or dask array
        """
        # rebin is made later on.
        data_ = super().get_data(rebin=None, reorder=reorder, **kwargs)
        
        if corr_nl:
            a, b = self.get_nonlinearity_corr()
            data_ /= (a*data_**2 + b*data_ + 1)
        
        if corr_overscan:
            os_model = self.get_overscan(**{**dict(which="model"),**overscan_prop})
            data_ -= os_model[:,None]            

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
              see stackstat (see specaxis). Clipping is applied at that time (if clipping=True)
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
        try:
            from scipy.stats import median_abs_deviation as nmad # scipy>1.9
        except:
            from scipy.stats import median_absolute_deviation as nmad # scipy<1.9
            
        
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
                mad_  = nmad( spec )
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
                return da.from_delayed(d_, shape=spec.shape, dtype="float32")
            # numpy
            return fit_polynome(np.arange(len(spec)), spec, degree=modeldegree)
        
        raise ValueError(f'which should be "raw", "data", "spec", "model", {which} given')    


    def get_lastdata_firstoverscan(self, **kwargs):
        """ get the last data and the first overscan columns
        
        Parameters
        ----------
        **kwargs goes to get_data

        Returns
        -------
        list 
            (2, n-row) data (last_data, first_overscan)
        """
        data = self.get_data(**kwargs)
        overscan = self.get_overscan("raw")
        if self.qid in [1,4]:
            last_data = data[:,-1]
            first_overscan = overscan[:,0]
        else:
            last_data = data[:,0]
            first_overscan = overscan[:,-1]
            
        return last_data, first_overscan

    
    def get_sciimage(self, use_dask=None, **kwargs):
        """ get the Science image corresponding to this raw image
        
        This uses ztfquery to parse the filename and set up the correct 
        science image filename path.

        Parameters
        ----------
        use_dask: bool or None
            if None, this will use self.use_dask.

        **kwargs goes to ScienceQuadrant.from_filename

        Returns
        -------
        ScienceQuadrant
        """
        if use_dask is None:
            use_dask = self.use_dask
            
        from .science import ScienceQuadrant
        from ztfquery.buildurl import get_scifile_of_filename
        # 
        filename = get_scifile_of_filename(self.filename, qid=self.qid, source="local")
        return ScienceQuadrant.from_filename(filename, use_dask=use_dask, **kwargs)
    
    # -------- #
    # PLOTTER  #
    # -------- #
    def show_overscan(self, ax=None, axs=None, axm=None, 
                      colorbar=False, cax=None, **kwargs):
        """ display the overscan image.

        Parameters
        ----------

        Returns
        -------
        fig
        """
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
            
        return fig
        
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
        return self.get_value("CCD_ID", attr_ok=False) # avoid loop
        
    @property
    def qid(self):
        """ quadrant (amplifier of the ccd) id (1->4) """
        return self._qid if hasattr(self, "_qid") else (self.get_value("AMP_ID")+1)
    
    @property
    def rcid(self):
        """ quadrant (within the focal plane) id (0->63) """
        return 4*(self.ccdid - 1) + self.qid - 1
    
    @property
    def gain(self):
        """ gain [adu/e-] """
        return self.get_value("GAIN", np.NaN, attr_ok=False) # avoid loop

    @property
    def darkcurrent(self):
        """ Dark current [e-/s]"""
        return self.get_value("DARKCUR", None, attr_ok=False) # avoid loop
    
    @property
    def readnoise(self):
        """ read-out noise [e-] """
        return self.get_value("READNOI", None, attr_ok=False) # avoid loop
        
        
class RawCCD( CCD ):
    _QUADRANTCLASS = RawQuadrant
    
    @classmethod
    def from_filename(cls, filename, as_path=True, use_dask=False, persist=False, **kwargs):
        """ load the instance from the raw filename.

        Parameters
        ----------
        filename: str
            fullpath or filename or the file to load.
            If a filename is given, set as_path=False,  then ztfquery.get_file() 
            will be called to grab the file for you (and download it if necessary)
            
        as_path: bool
            Set this to true if the input file is not the fullpath but you need
            ztfquery.get_file() to look for it for you.
        
        use_dask: bool, optional
            Should dask be used ? The data will not be loaded but delayed 
            (dask.array)

        persist: bool, optional
            = only applied if use_dask=True =
            should we use dask's persist() on data ?

        **kwargs: goes to _QUADRANTCLASS.from_filename

        
        Returns
        -------
        class instance 
        
        Examples
        --------
        Load a ztf image you know the name of but not the full path.
        
        >>> rawccd = ztfimg.RawCCD.from_filename("ztf_20220704387176_000695_zr_c11_o.fits.fz", as_path=False)
        """
        qids = (1,2,3,4)
        
        quadrant_from_filename = cls._QUADRANTCLASS.from_filename
        if use_dask:
            quadrant_from_filename = dask.delayed(quadrant_from_filename)
            
        quadrants = [quadrant_from_filename(filename, qid=qid,
                                                as_path=as_path,
                                                use_dask=False,
                                                persist=False, **kwargs)
                         for qid in qids]
        if persist and use_dask:
            quadrants = [q.persist() for q in quadrants]
            
        this = cls.from_quadrants(quadrants, qids=qids)
        this._filename = filename
        this._meta = io.parse_filename(filename)
        return this

    @classmethod
    def from_single_filename(cls, *args, **kwargs):
        """ rawccd data have a single file. 

        See also
        --------
        from_filename: load the instance given the raw filename
        """
        return cls.from_filename(*args, **kwargs)
    
    @classmethod
    def from_filenames(cls, *args, **kwargs):
        """ rawccd data have a single file. 

        See also
        --------
        from_filename: load the instance given the raw filename
        """
        raise NotImplementedError("from_filenames does not exists. See from_filename")
    
    @classmethod
    def from_filefracday(cls, filefracday, ccdid, use_dask=True, **kwargs):
        """ load the instance given a filefracday and the ccidid (ztf ID)

        Parameters
        ----------
        filefracday: str
            ztf ID of the exposure (YYYYMMDDFFFFFF) like 20220704387176
            ztfquery will fetch for the corresponding data.

        ccidid: int
            ccidid of the given ccd

        use_dask: bool
            Should dask be used ? The data will not be loaded but delayed 
            (dask.array)

        persist: bool
            = only applied if use_dask=True =
            should we use dask's persist() on data ?

        **kwargs goes to from_filename -> __init__

        Returns
        -------
        class instance
        
        """
        from ztfquery.io import filefracday_to_local_rawdata
        filename = filefracday_to_local_rawdata(filefracday, ccdid=ccdid)
        if len(filename)==0:
            raise IOError(f"No local raw data found for filefracday: {filefracday} and ccdid: {ccdid}")
        if len(filename)>1:
            raise IOError(f"Very strange: several local raw data found for filefracday: {filefracday} and ccdid: {ccdid}", filename)
        
        return cls.from_filename(filename[0], use_dask=use_dask, **kwargs)



    def get_sciimage(self, use_dask=None, qid=None, as_ccd=True, **kwargs):
        """ get the Science image corresponding to this raw image
        
        This uses ztfquery to parse the filename and set up the correct 
        science image filename path.

        Parameters
        ----------
        use_dask: bool or None
            if None, this will use self.use_dask.
            
        qid: int or None
            do you want a specific quadrant ?
            
        as_ccd: bool
            = ignored if qid is not None =
            should this return a list of science quadrant (False)
            or a ScienceCCD (True) ?

        **kwargs goes to ScienceQuadrant.from_filename

        Returns
        -------
        ScienceQuadrant
        """
        if use_dask is None:
            use_dask = self.use_dask

        from .science import ScienceQuadrant            
        from ztfquery.buildurl import get_scifile_of_filename
        # Specific quadrant
        if qid is not None:
            filename = get_scifile_of_filename(self.filename, qid=qid, source="local")
            return ScienceQuadrant.from_filename(filename, use_dask=use_dask, **kwargs)

        # no quadrant given -> 4 filenames (qid = 1,2,3,4)
        filenames = get_scifile_of_filename(self.filename, source="local")
        quadrants = [ScienceQuadrant.from_filename(filename, use_dask=use_dask, **kwargs)
                    for filename in filenames]
        
        if as_ccd:
            from .science import ScienceCCD
            return ScienceCCD.from_quadrants(quadrants, qids=[1,2,3,4], **kwargs)
        
        # If not, then list of science quadrants
        return quadrants

    
class RawFocalPlane( FocalPlane ):
    # INFORMATION || Numbers to be fine tuned from actual observations
    # 15 µm/arcsec  (ie 1 arcsec/pixel) and using 
    # 7.2973 mm = 487 pixel gap along rows (ie between columns) 
    # and 671 pixels along columns.
    _CCDCLASS = RawCCD
    
    @classmethod
    def from_filenames(cls, filenames, as_path=True,
                           use_dask=False, persist=False,
                           **kwargs):
        """ load the instance from the raw filename.

        Parameters
        ----------
        filenames: list of str
            list of fullpath or filename or the ccd file to load.
            If a filename is given, set as_path=False,  then ztfquery.get_file() 
            will be called to grab the file for you (and download it if necessary)
            
        as_path: bool
            Set this to true if the input file is not the fullpath but you need
            ztfquery.get_file() to look for it for you.
        
        use_dask: bool, optional
            Should dask be used ? The data will not be loaded but delayed 
            (dask.array)

        persist: bool, optional
            = only applied if use_dask=True =
            should we use dask's persist() on data ?

        **kwargs: goes to _CCDCLASS.from_filename
        
        Returns
        -------
        class instance 
        
        """
        this = cls()
        for file_ in ccd_filenames:
            ccd_ = cls._CCDCLASS.from_filename(file_, as_path=as_path,
                                                   use_dask=use_dask, persist=persist,
                                                   **kwargs)
            this.set_ccd(ccd_, ccdid=ccd_.ccdid)

        this._filenames = ccd_filenames
        return this

    @classmethod
    def from_filefracday(cls, filefracday, use_dask=True, **kwargs):
        """ load the instance given a filefracday and the ccidid (ztf ID)

        Parameters
        ----------
        filefracday: str
            ztf ID of the exposure (YYYYMMDDFFFFFF) like 20220704387176
            ztfquery will fetch for the corresponding data.

        use_dask: bool
            Should dask be used ? The data will not be loaded but delayed 
            (dask.array)

        persist: bool
            = only applied if use_dask=True =
            should we use dask's persist() on data ?

        **kwargs goes to from_filenames -> _CCDCLASS.from_filename

        Returns
        -------
        class instance
        
        """
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
        



