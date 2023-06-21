import numpy as np
import pandas
import warnings
import dask
import dask.array as da
import dask.dataframe as dd

from .utils.tools import rebin_arr, parse_vmin_vmax, ccdid_qid_to_rcid, rcid_to_ccdid_qid
from .utils.decorators import classproperty

__all__ = ["Image", "Quadrant", "CCD", "FocalPlane"]


def read_header(filepath, ext=None, use_dask=False, persist=False):
    """ reads the file header while handling serialization issues.

    Parameters
    ----------
    filepath: str
        file path.

    ext: int
        extension fo the header. see fits.getheader

    use_dask: bool
        should this use dask ?
        
    persist: bool
        = ignored if use_dask=False = 
        should the returned delayed object be persisted ?

    Returns
    -------
    Header or delayed
        if delayed, it is a pandas.Series that is delayed for serialization issues
    """
    from astropy.io import fits
    if use_dask:
        h_ = dask.delayed(fits.getheader)(filepath, ext)
        header = dask.delayed(pandas.Series)(h_)
        if persist:
            header = header.persist()
    else:
        header = fits.getheader(filepath, ext)
    return header

    
class Image( object ):
    SHAPE = None
    # Could be any type (raw, science)

    def __init__(self, data=None, header=None):
        """  
        See also
        --------
        from_filename: load the instance given a filename 
        from_data: load the instance given its data (and header)
        
        """
        if data is not None:
            # this updates use_dask
            self.set_data(data)
            
        if header is not None:
            self.set_header(header)

    @classmethod
    def _read_data(cls, filepath, use_dask=False, persist=False, ext=None):
        """ assuming fits format."""
        from astropy.io.fits import getdata

        if "dask" in str( type(filepath) ):
            use_dask = True
    
        if use_dask:
            # - Data
            data = da.from_delayed( dask.delayed( getdata) (filepath, ext=ext),
                                   shape=cls.SHAPE, dtype="float32")
            if persist:
                data = data.persist()

        else:
            data = getdata(filepath, ext=ext)

        return data

    @staticmethod
    def _read_header(filepath, use_dask=False, persist=False, ext=None):
        """ assuming fits format. """
        if "dask" in str( type(filepath) ):
            use_dask = True

        header = read_header(filepath, use_dask=use_dask, persist=persist, ext=ext)            
        return header

    @staticmethod
    def _to_fits(fileout, data, header=None,  overwrite=True,
                     **kwargs):
        """ """
        import os        
        from astropy.io import fits
        dirout = os.path.dirname(fileout)
        if dirout not in ["","."] and not os.path.isdir(dirout):
            os.makedirs(dirout, exist_ok=True)

        fits.writeto(fileout, data, header=header,
                         overwrite=overwrite, **kwargs)
        return fileout
    
    @staticmethod
    def _get_filepath(filename, as_path=True, use_dask=False, **kwargs):
        """ internal function that can call ztfquery.get_file() to download files you don't have yet.
        
        Parameters
        ----------
        as_path: bool
            is the into filename the full path (so nothing to do)
        
        use_dask: bool
            = only applies if as_path is False =
            this using ztfquery.get_file(). Should this be delayed ?

        **kwargs: options of ztfquery.get_file()

        Returns
        -------
        filepath or delayed
        """
        if as_path:
            return filename
        
        from ztfquery import io
        # Look for it
        prop = dict(show_progress=False, maxnprocess=1)
        if use_dask:
            filename  = dask.delayed(io.get_file)(filename, **prop)
        else:
            filename  = io.get_file(filename, **prop)

        return filename
    
    @classmethod
    def from_filename(cls, filename,
                          as_path=True,
                          use_dask=False, persist=False,
                          dask_header=False, **kwargs):
        """ classmethod load an instance given an input file.

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

        dask_header: bool, optional
            should the header be dasked too (slows down a lot)

        **kwargs: goes to __init__()

        
        Returns
        -------
        class instance 
        
        Examples
        --------
        Load a ztf image you know the name of but not the full path.
        
        >>> img = Image.from_filename("ztf_20220704387176_000695_zr_c11_o_q3_sciimg.fits", as_path=False, use_dask=False)
        
        """
        if not use_dask:
            dask_header = False
            
        filepath = cls._get_filepath(filename, as_path=as_path, use_dask=use_dask)
        data = cls._read_data(filepath, use_dask=use_dask, persist=persist)
        header = cls._read_header(filepath, use_dask=dask_header, persist=persist)
        
        # self
        this = cls.from_data(data=data, header=header, **kwargs)
        this._filename = filename
        this._filepath = filepath
        return this

    @classmethod
    def from_data(cls, data, header=None, **kwargs):
        """ Instanciate this class given data. 
        

        Parameters
        ----------
        data: numpy.array or dask.array]
            Data of the Image.
            this will automatically detect if the data are dasked.

        header: fits.Header or dask.delayed
            Header of the image.

        **kwargs goes to __init__

        Returns
        -------
        class instance
        
        """
        return cls(data=data, header=header, **kwargs)

    def to_fits(self, fileout, overwrite=True, **kwargs):
        """ writes the image (data and header) into a fits file.

        Parameters
        ----------
        fileout: str
            path of where the data (.fits format) should be stored

        overwrite: bool
            if fileout already exists, shall this overwrite it ?

        **kwargs goes to astropy.fits.writeto()
        
        Returns
        -------
        str
            for convinience, the fileout is returned
        """
        return self._to_fits(fileout, data=self.data, header=self.header,
                                overwrite=overwrite, **kwargs)
    
    # -------- #
    #  SETTER  #
    # -------- #
    def set_header(self, header):
        """ set self.header with the given header. """
        self._header = header

    def set_data(self, data):
        """ Set the data to the instance

        = It is unlikely you need to use that directly. =
        
        Parameters
        ----------
        data: numpy.array, dask.array
            numpy or dask array with the same shape as cls.shape.
        
        Returns
        -------
        None

        Raises
        ------
        ValueError
            This error is returned if the shape is not matching 
        
        See also
        --------
        from_filename: loads the instance (data, header) given a filepath
        from_data: loads the instance given data and header.
        
        """
        if self.shape is not None and ((dshape:=data.shape) != self.shape).any():
            raise ValueError(f"shape of the input CCD data must be {self.shape} ; {dshape} given")

        # Check dask compatibility
        used_dask = "dask" in str(type(data))
        if self.use_dask is not None and used_dask != self.use_dask:
            warnings.warn(f"Input data and self.use_dask are not compatible. Now: use_dask={used_dask}")
            
        self._use_dask = used_dask
        self._data = data
        
    # -------- #
    #  GETTER  #
    # -------- #
    def get_data(self, rebin=None, rebin_stat="nanmean", data=None):
        """ get image data 

        Parameters
        ----------
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

        data: str
            If None, data="data" assumed.
            Internal option to modify the data.
            This could be any attribure value of format (int/float)
            Leave to 'data' if you are not sure.

        Returns
        -------
        2d array
            dask or numpy array, which shape will depend on rebin
        """
        npda = da if self.use_dask else np
        
        if data is None:
            data = "data"
            
        if rebin is not None and rebin <= 1:
            warnings.warn("rebin <=1 does not make sense. rebin set to None")
            rebin = None

        # This is the normal case
        if self.has_data() and rebin is None and data=="data":
            data_ = self.data#.copy()
        
        else:
            if type(data) == str:
                if data == "data":
                    data_ = self.data#.copy()
                elif hasattr(self, data):
                    data_ = npda.ones(self.shape) * getattr(self, data)
                else:
                    raise ValueError(
                        f"value as string can only be 'data' or a known attribute ; {data} given")
                
            elif type(data) in [int, float]:
                data_ = npda.ones(self.shape) * data
            else:
                data_ = data#.copy()

        if rebin is not None:
            data_ = getattr(npda, rebin_stat)(
                rebin_arr(data_, (rebin, rebin), use_dask=self._use_dask), axis=(-2, -1))

        return data_

    def get_header(self, compute=True):
        """ get the current header

        this is a shortcut to self.header

        Returns
        -------
        fits.Header or pandas.Series
            if compute is needed, this returns a pandas.Series
            for serialization issues
        """
        if "dask" in str( type( self.header) ) and compute:
            return self.header.compute()
        
        return self.header

    def get_value(self, key, default=None, attr_ok=True):
        """ quick access to an image value.
        
        This method looks for this key in:
        1. image's attributes
        2. image's meta
        3. image's header (using upper case).
        
        Parameters
        ----------
        key: str
            entry of the header. 

        default: None, float, str
            what is returned if the entry cannot be found in the header.

        attr_ok: bool
            allows to look for self.key.

        Returns
        -------
        str, float, int, etc.
            whatever is in the header key you queried.

        Raises
        ------
        AttributeError 
            If no header is set this returns is returned
        """
        key_ = key.lower()
        # Is that a local attribute ?
        if attr_ok and hasattr(self, key_):
            return getattr(self, key_)
        
        # Look for meta then
        key_ = key.lower()
        _CONVERTION = {"fieldid":"field",
                       "filter": "filtercode",
                       "filtername": "filtercode",
                        }
        key_ = _CONVERTION.get(key_,key_)
        
        meta = self.meta
        if meta is not None and key_ in meta:
            return meta[key_]

        # Look for header next
        if self.header is None:
            return default

        # header so upper case:
        key_ = key.upper()
        
        # keep all in upper case
        _HEADER_KEY = {"CCDID": "CCD_ID",
                        "FILTERCODE": "FILTER"}
        key_ = _HEADER_KEY.get(key_, key_) # update
        
        return self.header.get(key_, default)

    def get_aperture(self, x, y, radius,
                     data=None,
                     bkgann=None, subpix=0,
                     err=None, mask=None,
                     as_dataframe=True,
                     **kwargs):
        """ get the apeture photometry, base on `sep.sum_circle()`

        Parameters
        ----------
        x, y: array
            coordinates of the centroid for the apertures.
            x and y are image pixel coordinates.
            numpy or dask array.

        radius: float, list
            size (radius) of the aperture. This could be a list of radius.
            
        data: 2d-array, None
            if you want to apply the aperture photometry on this specific image, provide it.
            otherwhise, ``data = self.get_data()`` is used
            
        bkgann: 2d-array, None
            inner and outer radius of a “background annulus”.
            If supplied, the background is estimated by averaging 
            unmasked pixels in this annulus.

        subpix: int
            Subpixel sampling factor. 0 is the exact overlap calculation ; 5 is acceptable.
            
        err: 2d-array, None
            error image if you have it.

        mask: 2d-array, None
            mask image if you have it. Pixels within this mask will be ignored.

        as_dataframe: bool
            return format.
            If As DataFrame, this will be a dataframe with
            3xn-radius columns (f_0...f_i, f_0_e..f_i_e, f_0_f...f_i_f)
            standing for fluxes, errors, flags.
            
        Returns
        -------
        2d-array or dataframe
           flux, error and flag for each coordinates and radius.

        Examples
        --------
        get the aperture photometry of random location in the image.

        >>> import ztfimg
        >>> import numpy as np
        >>> img = ztfimg.Quadrant.from_filename("ztf_20220704387176_000695_zr_c11_o_q3_sciimg.fits", as_path=False, 
                                    use_dask=False)
        >>> x = np.random.uniform(0, ztfimg.Quadrant.shape[1], size=400)
        >>> y = np.random.uniform(0, ztfimg.Quadrant.shape[0], size=400)
        >>> radius = np.linspace(1,5,10)
        >>> df = img.get_aperture(x,y, radius=radius[:,None], as_dataframe=True)
        >>> df.shape
        (400, 30) # 30 because 10 radius, so 10 flux, 10 errors, 10 flags
         
        """
        if data is None:
            data = self.get_data()

        radius = np.atleast_1d(radius) # float->array for dasked images.
        return self._get_aperture(data, x, y, radius,
                                  bkgann=bkgann, subpix=subpix,
                                  err=err, mask=mask,
                                  as_dataframe=as_dataframe,
                                  **kwargs)

    @staticmethod
    def _get_aperture(data,
                     x, y, radius,
                     bkgann=None, subpix=0,
                     use_dask=None,
                     err=None, mask=None,
                     as_dataframe=False,
                     **kwargs):
        """  get the apeture photometry, base on `sep.sum_circle()`
        = Internal method = 

        Parameters
        ----------
        data: 2d-array, None
            data onto which the aperture will be applied.

        x, y: array
            coordinates of the centroid for the apertures.
            x and y are image pixel coordinates.

        radius: float, list
            size (radius) of the aperture. This could be a list of radius.
            
        bkgann: 2d-array, None
            inner and outer radius of a “background annulus”.
            If supplied, the background is estimated by averaging 
            unmasked pixels in this annulus. If supplied, the inner
            and outer radii obey numpy broadcasting rules along with ``x``,
            ``y`` and ``r``.

        subpix: int
            Subpixel sampling factor. 0 is the exact overlap calculation ; 5 is acceptable.
            
        err: 2d-array, None
            error image if you have it.

        mask: 2d-array, None
            mask image if you have it. Pixels within this mask will be ignored.

        as_dataframe: bool
            return format.
            If As DataFrame, this will be a dataframe with
            3xn-radius columns (f_0...f_i, f_0_e..f_i_e, f_0_f...f_i_f)
            standing for fluxes, errors, flags.

        Returns
        -------
        (3, n) array or `pandas.DataFrame`
            array: with n the number of radius.
        """

        from .utils.tools import get_aperture

        if use_dask is None:
            use_dask = "dask" in str( type(data) )

        apdata = get_aperture(data,
                              x, y, radius=radius,
                              err=err, mask=mask, bkgann=bkgann,
                              use_dask=use_dask, **kwargs)
        
        if not as_dataframe:
            return apdata

        # Generic form works for dask and np arrays
        nradius = len(radius)
        dic = {**{f'f_{k}': apdata[0, k] for k in range(nradius)},
               **{f'f_{k}_e': apdata[1, k] for k in range(nradius)},
               # for each radius there is a flag
               **{f'f_{k}_f': apdata[2, k] for k in range(nradius)},
               }

        if "dask" in str(type(apdata)):
            return dd.from_dask_array(da.stack(dic.values(), allow_unknown_chunksizes=True).T,
                                      columns=dic.keys())

        return pandas.DataFrame(dic)
        
    # -------- #
    # PLOTTER  #
    # -------- #
    def show(self, ax=None, colorbar=True, cax=None, apply=None,
                 data=None,
                 vmin="1", vmax="99",
                 rebin=None,
                 savefile=None,
                 dpi=150,  **kwargs):
        """ Show the image data (imshow)

        Parameters
        ----------
        ax: matplotlib.Axes, None
            provide the axes where the image should be drawn

        colobar: bool
            should be colobar be added ?

        cax: matplotlib.Axes, None
            = ignored if colobar=False =
            axes where the colorbar should be drawn

        apply: str, None
            provide a numpy method that should be applied to the data
            prior been shown. 
            For instance, apply="log10", then np.log10(data) will be shown.
            
        data: 2d-array, None
            if you want to plot this specific image, provide it.
            otherwhise, ``data = self.get_data(rebin=rebin)`` is shown.

        vmin, vmax: str, float
            minimum and maximum value for the colormap.
            string are interpreted as 'percent of data'. 
            float or int are understood as 'use as such'

        rebin: int, None
            by how much should the data be rebinned when accessed ?
            (see details in get_data())
            
        savefile: str, None
            if you want to save the plot, provide here the path for that.
        
        dpi: int
            = ignored if savefile is None =
            dpi of the stored image

        Returns
        -------
        matplotlib.Figure


        See also
        --------
        get_data: acess the image data
        """
        if ax is None:
            import matplotlib.pyplot as plt            
            fig = plt.figure(figsize=[5 + (1.5 if colorbar else 0), 5])
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure

        if data is None:
            data = self.get_data(rebin=rebin)
            
        if "dask" in str( type(data) ):
            data = data.compute()

        if apply is not None:
            data = getattr(np, apply)(data)

        vmin, vmax = parse_vmin_vmax(data, vmin, vmax)
        prop = dict(origin="lower", cmap="cividis", vmin=vmin, vmax=vmax)

        im = ax.imshow(data, **{**prop, **kwargs})
        if colorbar:
            fig.colorbar(im, cax=cax, ax=ax)

        if savefile is not None:
            fig.savefig(savefile, dpi=dpi)

        return fig

    # =============== #
    #  Properties     #
    # =============== #
    @property
    def use_dask(self):
        """ are the data dasked (did you set use_dask=True) """
        if not hasattr(self, "_use_dask"): # means set_data never called
            return None
        
        return self._use_dask
    
    @property
    def data(self):
        """ data of the image ; numpy.array or dask.array"""
        if not hasattr(self, "_data"):
            return None
        
        return self._data

    def has_data(self):
        """ are data set ? (True means yes)"""
        return hasattr(self, "_data") and self._data is not None

    @property
    def header(self):
        """ header of the data."""
        if not hasattr(self, "_header"):
            return None

        return self._header

    @classproperty
    def shape(cls):
        """ shape of the images """
        return np.asarray(cls.SHAPE)

    # // header
    @property
    def filename(self):
        """ If this method was loaded from a file, this is it's filename. None otherwise """
        if not hasattr(self, "_filename"):
            return self.filepath # could be None 
        return self._filename

    @property
    def filepath(self):
        """ If this method was loaded from a file, this is it's filename. None otherwise """
        if not hasattr(self, "_filepath"):
            return None
        return self._filepath
    

    @property
    def filtername(self):
        """ Name of the image's filter (from header) """
        return self.get_value("filtercode", "unknown", attr_ok=False) # avoid loop

    @property
    def exptime(self):
        """ Exposure time of the image (from header) """
        return self.get_value("EXPTIME", np.NaN, attr_ok=False) # avoid loop

    @property
    def obsjd(self):
        """ Observation Julian date of the image (from header) """
        return self.get_value("OBSJD", None, attr_ok=False) # avoid loop

    @property
    def meta(self):
        """ meta data for the instance, from the filename. """
        if not hasattr(self, "_meta"):
            return None
        return self._meta

    
# -------------- #
#                #
#   Virtual      #
#                #
# -------------- #
class _Collection_( object ):
    _COLLECTION_OF = Image # Set this to None to let it guess

    @classmethod
    def _read_filenames(cls, filenames, use_dask=False, 
                           as_path=False, persist=False,
                           **kwargs):
        """ """
        filenames = np.atleast_1d(filenames).tolist()
        
        prop = {**dict(as_path=as_path, use_dask=use_dask), **kwargs}

        # Collection of what ?
        if cls._COLLECTION_OF is not None: # already specified
            COLLECTION_OF = cls._COLLECTION_OF
        else:
            COLLECTION_OF = cls._guess_filenames_imgtype_(filenames)
        
        # dask is called inside this.
        images = [COLLECTION_OF.from_filename(filename, **prop) for filename in filenames]
        if use_dask and persist:
            images = [i.persist() for i in images]
            
        return images, filenames

    @staticmethod
    def _guess_filenames_imgtype_(filenames):
        """ guess the Image Class given the filenames
        
        Only sci->ScienceQuadrant and raw->RawCCD implemented.

        This will raise a NotImplementedError if multiple kinds of filenames 
        are given.
        
        
        """
        from ztfquery import io
        kind = io.filename_to_kind(filenames)
        if len( np.unique(kind) ) > 1:
            raise NotImplementedError("Only single filename image kind inplemented.")
        else:
            kind = kind[0] # sci, raw or cal
            
        # Works
        if kind == "sci":
            from .science import ScienceQuadrant
            imgtype = ScienceQuadrant
        elif kind == "raw":
            from .raw import RawCCD
            imgtype = RawCCD
        else:
            raise NotImplementedError(f"Only sci and raw image kind implemented ; {kind} given")
        
        return imgtype
            
    def _get_subdata(self, calling="get_data", **kwargs):
        """ get a stack of the data from get_data collection of """
        datas = self._call_down(calling, **kwargs)
        # This rest assumes all datas types match.
        use_dask = "dask" in str( type(datas[0]) )
        if use_dask:
            if type( datas[0] ) != da.Array: # Dask delayed
                datas = [da.from_delayed(f_, shape=self._COLLECTION_OF.shape,
                                                 dtype="float32")
                                     for f_ in datas]
            # dask.array stack
            datas = da.stack(datas)
        else:
            # numpy.array stack
            datas = np.stack(datas)
            
        return datas

    # -------- #
    # INTERNAL #
    # -------- #
    def _map_down(self, what, margs, *args, **kwargs):
        """ """
        return [getattr(img, what)(marg, *args, **kwargs)
                for img, marg in zip(self._images, margs)]
    
    def _call_down(self, what, *args, **kwargs):
        """ call an attribute or a method to each image.
        
        Parameters
        ----------
        what: str
            attribute or method of individual images.
            
        args, kwargs: 
            = ignored if what is an attribute = 
            method options
            
        See also
        --------
        _map_down: map a list to the list of image
        """
        import inspect
        
        if inspect.ismethod( getattr(self._images[0], what) ): # is func ?
            return [getattr(img, what)(*args, **kwargs) for img in self._images]
        
        return [getattr(img, what) for img in self._images]


    def compute(self, **kwargs):
        """ compute all dasked images
        = careful this may load a lot of data in memory =
        
        Ignored if not use_dask.
            
        **kwargs goes to individual's image compte().
        """
        warnings.warn("compute of a collection is not optimal yet. Loops over images to call their compute.")
        _ = self._call_down("compute", **kwargs)
        self._use_dask = False # if needed be
        
    def persist(self, **kwargs):
        """ persist all dasked images
        = this loads data in your cluster's memory =
        
        **kwargs goes to individual's image compte().
        """
        _ = self._call_down("persist", **kwargs)

    # ============== #
    #   Properties   #
    # ============== #
    @property
    def collection_of(self):
        """ name of the collection elements. """
        if not hasattr(self, "_collection_of") or self._collection_of is None:
            self._collection_of = self._COLLECTION_OF
            
        return self._collection_of
# -------------- #
#                #
#   QUADRANT     #
#                #
# -------------- #
class Quadrant(Image):
    SHAPE = (3080, 3072)
    # "family"
    _CCDCLASS = "CCD"
    _FocalPlaneCLASS = "FocalPlane"

    # ============== #
    #   Methods      #
    # ============== #
    # -------- #
    #  GETTER  #
    # -------- #
    def get_ccd(self, use_dask=None, as_path=False, **kwargs):
        """ get the ccd object containing this quadrant. 
        (see self._CCDCLASS)

        Parameters
        ----------
        use_dask: bool, None
            should the ccd object be build using dask ? 
            if None, the current instance use_dask is used.
        
        as_path: bool
            should this assume that instance self.filename can 
            be directly transformed ? False should be favored 
            as it downloads the missing images if need.

        **kwargs goes the self._CCDCLASS.from_single_filename()

        Returns
        -------
        CCD
            instance of self._CCDCLASS.
        """
        if use_dask is None:
            use_dask = self.use_dask
        
        return self._ccdclass.from_single_filename(self.filename,
                                                   use_dask=use_dask,
                                                   as_path=as_path,
                                                   **kwargs)

    def get_focalplane(self, use_dask=None, as_path=False, **kwargs):
        """ get the full focal plane (64 quadrants making 16 CCDs) 
        containing this quadrant

        (see self._FocalPlaneCLASS)

        Parameters
        ----------
        use_dask: bool, None
            should the ccd object be build using dask ? 
            if None, the current instance use_dask is used.
        
        as_path: bool
            should this assume that instance self.filename can 
            be directly transformed ? False should be favored 
            as it downloads the missing images if need.

        **kwargs goes the self._FocalPlaneCLASS.from_single_filename()

        Returns
        -------
        FocalPlane
            instance of self._FocalPlaneCLASS.
        """
        if use_dask is None:
            use_dask = self.use_dask
        
        return self._focalplaneclass.from_single_filename(self.filename,
                                                          use_dask=use_dask,
                                                          as_path=as_path,
                                                          **kwargs)
    
    def get_data(self, rebin=None, reorder=True, **kwargs):
        """ get image data 
        
        Parameters
        ----------
        rebin: int, None
            Shall the data be rebinned by square of size `rebin` ?
            None means no rebinning.
            (see details in rebin_stat)
            rebin must be a multiple of the image shape.
            for instance if the input shape is (6160, 6144)
            rebin could be 2,4,8 or 16

        reorder: bool
            Should the data be re-order to match the actual north-up.
            (leave to True if not sure)
            
        **kwargs goes to super.get_data() e.g. rebin_stat

        Returns
        -------
        data
            numpy.array or dask.array
        """
        data = super().get_data(rebin=rebin, **kwargs)
        if reorder:
            data = data[::-1,::-1]
            
        return data

    def get_catalog(self, name, fieldcat=False, radius=0.7, 
                        reorder=True, in_fov=False,
                        use_dask=None, **kwargs):
        """ get a catalog for the image
        
        Parameters
        ----------
        name: str
            name of a the catalog.
            - ps1 # available for fieldcat

        fieldcat: bool
            is the catalog you are looking for a "field catalog" ?
            (see catalog.py). See list of available names in 
            name parameter doc.

        radius: float
            = ignored if fieldcat is True =
            radius [in degree] of the cone search 
            centered on the quadrant position. 
            radius=0.7 is slightly larger that half a diagonal.

        reorder: bool
            when creating the x and y columns given the 
            catalog ra, dec, should this assume x and y reordered
            position. 
            reorder=True matches data from get_data() but not self.data. 
            (leave to True if unsure).
        
        in_fov: bool
            should entries outside the image footprint be droped ?
            (ignored if x and y column setting fails).

        use_dask: bool
            should this return a dask.dataframe for the catalog ?
            If None, this will use self.use_dask.
            
        **kwargs goes to get_field_catalog or download_vizier_catalog.
        
        Returns
        -------
        DataFrame
            catalog dataframe. The image x, y coordinates columns will be added
            using the radec_to_xy method. If not available, NaN will be set.
            
        """
        if use_dask is None:
            use_dask = self.use_dask
        
        if fieldcat:
            from .catalog import get_field_catalog
            cat = get_field_catalog(name,
                                    fieldid=self.get_value("fieldid"),
                                    rcid=self.get_value("rcid", None), 
                                    ccdid=self.get_value("ccdid", None), # ignored if rcid is not None
                                    use_dask=use_dask,
                                    **kwargs)
        else:
            from .catalog import download_vizier_catalog
            cat = download_vizier_catalog(name, self.get_center("radec"),
                                            radius=radius, r_unit='deg',
                                            use_dask=use_dask,
                                            **kwargs)
        # This is inplace.
        cat = self.add_xy_to_catalog(cat, ra="ra", dec="dec",
                                       reorder=reorder, in_fov=in_fov)
        return cat

    def add_xy_to_catalog(self, cat, ra="ra", dec="dec", reorder=True, in_fov=False):
        """ add the quadrant xy coordinates to a given catalog if possible.
        
        This assume that radec_to_xy is implemented for this instance.

        Parameters
        ----------
        cat: pandas.DataFrame or dask.dataframe
            catalog with at least the ra and dec keys.

        ra: str
            R.A. entry of the input catalog

        dec: str
            Dec entry of the input catalog

        reorder: bool
            when creating the x and y columns given the 
            catalog ra, dec, should this assume x and y reordered
            position. 
            reorder=True matches data from get_data() but not self.data. 
            (leave to True if unsure).

        in_fov: bool
            should entries outside the image footprint be droped ?
            (ignored if x and y column setting fails).

        Returns
        -------
        DataFrame
            pandas or dask. 

        See also
        --------
        get_catalog: get a catalog for this instance.
        """
        # is catalog dasked ?
        dasked_cat = "dask" in str( type(cat) )
        
        # Adding x, y coordinates
        # convertion available ?
        if hasattr(self, "radec_to_xy") and ra in cat and dec in cat:
            if dasked_cat: # dasked catalog
                x_y = dask.delayed(self.radec_to_xy)(*cat[[ra, dec]].values.T, reorder=reorder)
                x = da.from_delayed(x_y[0], shape=(cat[ra].values.size,), dtype="float32" )
                y = da.from_delayed(x_y[1], shape=(cat[ra].values.size,), dtype="float32" )
            else:
                x, y = self.radec_to_xy(*cat[[ra, dec]].values.T, reorder=reorder)
        else:
            if dasked_cat:
                x, y = da.NaN, da.NaN
            else:
                x, y = np.NaN, np.NaN
                
            in_fov = False
            
        cat["x"] = x
        cat["y"] = y
        if in_fov:
            cat = cat[cat["x"].between(0, self.shape[-1]) &
                      cat["y"].between(0, self.shape[0]) ]
            
        return cat
        
        
    def get_center(self, system="xy", reorder=True):
        """ get the center of the image

        Parameters
        ----------
        system: string
            coordinate system.
            - xy: image pixel coordinates
            - ij: ccd coordinates
            - radec: sky coordinates (in deg)
            - uv: camera coordinates (in arcsec)

        reorder: bool
            should this provide the coordinates assuming
            normal ordering (+ra right, +dec up) (True) ?
            = leave True if unsure = 

        Returns
        -------
        2d-array
            coordinates (see system)
        """
        center_pixel = (self.shape[::-1]+1)/2
        if system in ["xy","pixel","pixels","pxl"]:
            return center_pixel

        if system in ["ccd", "ij"]:
            return np.squeeze(self.xy_to_ij(*center_pixel) )
        
        if system in ["uv","tangent"]:
            return np.squeeze(self.xy_to_uv(*center_pixel, reorder=reorder) )
        
        if system in ["radec","coords","worlds"]:
            return np.squeeze(self.xy_to_radec(*center_pixel, reorder=reorder) )

        raise ValueError(f"{system} system not recognized. 'xy', 'ij', 'radec' or 'uv'")            

    def get_corners(self, system="xy", reorder=True):
        """ get the corners of the image.

        Parameters
        ----------
        system: str
            coordinate system.
            - xy: image pixel coordinates
            - ij: ccd coordinates
            - radec: sky coordinates (in deg)
            - uv: camera coordinates (in arcsec)

        reorder: bool
            = ignored if system='xy' =    
            should this provide the coordinates assuming
            normal ordering (+ra right, +dec up) (True) ?
            Leave default if unsure. 

        Returns
        -------
        2d-array
            lower-left, lower-right, upper-right, upper-left
        """
        # 
        corners = np.stack([[0,0], 
                            [self.shape[1],0],
                            [self.shape[1],self.shape[0]],
                            [0,self.shape[0]]])

        if system in ['xy', "pixels"]:
            return corners
        
        if system in ["ccd", "ij"]:
            return self.xy_to_ij(*corners.T).T
        
        if system == "radec":
            return self.xy_to_radec(*corners.T, reorder=True).T
        
        if system == "uv":
            return self.xy_to_uv(*corners.T, reorder=True).T

        raise ValueError(f"{system} system not recognized. 'xy', 'ij', 'radec' or 'uv'")

    # ======== #
    #  Dask    #
    # ======== #
    def compute(self, **kwargs):
        """ computes all delayed attribute.


        Parameters
        ----------
        attrnames: list
            list of attribute name this should be applied to.
            If None, all dasked attributes will be used.

        **kwargs goes to delayed.compute(**kwargs)

        Return
        ------
        None

        See also
        --------
        persist: persists (some) delayed attributes
        """
        
        if not self.use_dask:
            warnings.warn("dask not used ; nothing to do.")
            return

        attrnames = self._get_dasked_attributes_()
        # compute all at once;
        newattr = dask.delayed(list)([getattr(self, a) for a in attrnames]
                                    ).compute(**kwargs)
        _ = [setattr(self, name_, attr_) for name_, attr_ in zip(attrnames, newattr)]

        # not dasked anymore
        self._use_dask = False

    def persist(self, attrnames=None, **kwargs):
        """ persist delayed attributes

        Parameters
        ----------
        attrnames: list
            list of attribute name this should be applied to.
            If None, all dasked attributes will be used.

        **kwargs goes to delayed.persist(**kwargs)
        
        Return
        ------
        None

        See also
        --------
        compute: computes delayed attributes
        """
        
        if not self.use_dask:
            warnings.warn("dask not used ; nothing to do.")
            return

        if attrnames is None:
            attrnames = self._get_dasked_attributes_()

        # persist all individually
        newattr = [getattr(self, a).persist(**kwargs) for a in attrnames]
        _ = [setattr(self, name_, attr_) for name_, attr_ in zip(attrnames, newattr)]

        # still dasked
        self._use_dask = True


    def _get_dasked_attributes_(self):
        """ returns the name of all dasked attributes """
        to_be_check = ["_data", "_mask", "_filepath", "_header"]
        return [attr for attr in to_be_check
                    if hasattr(self, attr) and  "dask" in str( type( getattr(self, attr) ) )]
        
    # =============== #
    #  Properties     #
    # =============== #    
    @property
    def qid(self):
        """ quadrant id (from header) """
        return self.get_value("QID", None, attr_ok=False) # avoid loop
        
    @property
    def ccdid(self):
        """ ccd id (from header) """
        return self.get_value("CCDID", None, attr_ok=False) # avoid loop

    @property
    def rcid(self):
        """ rcid (from header) """
        return self.get_value("RCID", None, attr_ok=False) # avoid loop

    # family
    @classproperty
    def _ccdclass(cls):
        """ """
        if type(cls._CCDCLASS) is str:
            try: # from this .py
                return eval( cls._CCDCLASS )
            except: # from the library 
                exec(f"from .__init__ import {cls._CCDCLASS}")
                return eval( cls._CCDCLASS )
        
        return cls._CCDCLASS

    @classproperty
    def _focalplaneclass(cls):
        """ """
        if type(cls._FocalPlaneCLASS) is str:
            try: # from this .py
                return eval( cls._FocalPlaneCLASS )
            except: # from the library 
                exec(f"from .__init__ import {cls._FocalPlaneCLASS}")
                return eval( cls._FocalPlaneCLASS )

        
        return cls._FocalPlaneCLASS
    
# -------------- #
#                #
#     CCD        #
#                #
# -------------- #

class CCD( Image, _Collection_):
    # Basically a Quadrant collection

    SHAPE = (3080*2, 3072*2)
    _COLLECTION_OF = Quadrant
    # "family"
    _QUADRANTCLASS = "Quadrant"
    _FocalPlaneCLASS = "FocalPlane"

    def __init__(self, quadrants=None, qids=None,
                     data=None, header=None,
                     pos_inverted=None, **kwargs):
        """ CCD are collections for quadrants except if loaded from whole CCD data.
        

        See also
        --------
        from_filename: load the instance given a filename 
        from_filenames: load the image given the list of its four quadrant filenames
        from_single_filename: build and load the instance given the filename of a single quadrant.
        """
        if data is not None and quadrants is not None:
            raise ValueError("both quadrants and data given. This is confusing. Use only one.")
        
        # - Ok good to go
        _ = super().__init__()

        if data is not None:
            # this updates use_dask
            self.set_data(data)
            
        if header is not None:
            self.set_header(header)
        
        if quadrants is not None:
            if qids is None:
                qids = [None]*len(quadrants)
                
            elif len(quadrants) != len(qids):
                raise ValueError("quadrants and qids must have the same size.")
            
            # this updates use_dask
            _ = [self.set_quadrant(quad_, qid=qid, **kwargs)
                     for quad_, qid in zip(quadrants, qids)]
                
    # =============== #
    #  I/O            #
    # =============== #        
    @classmethod
    def from_single_filename(cls, filename, as_path=True, use_dask=False, persist=False, **kwargs):
        """ given a single quadrant file, this fetchs the missing ones and loads the instance.

        Parameters
        ----------
        filename: str
            filename of a single *quadrant* filename.
            This will look for the 3 missing to build the ccd.
            
        as_path: bool -optional-
            Set this to true if the input file is not the fullpath but you need
            ztfquery.get_file() to look for it for you.
        
        use_dask: bool, optional
            Should dask be used ? The data will not be loaded but delayed 
            (dask.array)

        persist: bool, optional
            = only applied if use_dask=True =
            should we use dask's persist() on data ?

        **kwargs goes to _QUADRANTCLASS.from_filename

        Returns
        -------
        class instance

        Examples
        --------
        >>> ccdimg = CCD.from_single_filename("ztf_20220704387176_000695_zr_c11_o_q3_sciimg.fits", as_path=False, use_dask=False)

        See also
        --------
        from_quadrants: loads the instance given a list of the four quadrants
        from_filenames: loads the instance given the filename of the four quadrants
        from_filename: loads the intance given a filename (assuming data are full-ccd shape)
        """
        import re
        qids = range(1, 5)
        filenames = [re.sub(r"_o_q[1-5]*", f"_o_q{i}", filename) for i in qids]
        return cls.from_filenames(filenames, qids=qids, as_path=as_path,
                                      use_dask=use_dask, persist=persist, **kwargs)

    @classmethod
    def from_quadrants(cls, quadrants, qids=None, **kwargs):
        """ loads the instance given a list of four quadrants.

        Parameters
        ----------
        quadrants: list of ztfimg.Quadrants
            the for quadrants 

        qids: None, list of int
            the quadrants idea (otherise taken from the quadrants)
            
        Returns
        -------
        class instance

        Raises
        ------
        ValueError
            this error is returned if you do not provide exactly 4 quadrants.
        """
        if (nquad:=len(quadrants)) !=4:
            raise ValueError(f"You must provide exactly 4 quadrants, {nquad} given")

        # use_dask will be determined by the quadrants data (np or da ?)
        return cls(quadrants=quadrants, qids=qids, **kwargs)

    @classmethod
    def from_filenames(cls, filenames, as_path=True,
                           qids=[1, 2, 3, 4], use_dask=False,
                           persist=False, **kwargs):
        """ loads the instance from a list of the quadrant files.

        Parameters
        ----------
        filanames: list of four str
            filenames for the four quadrants 
        
        as_path: bool
            Set this to true if the input file is not the fullpath but you need
            ztfquery.get_file() to look for it for you.

        qids: list of int
            list of the qid for the input filenames

        use_dask: bool
            Should dask be used ? The data will not be loaded but delayed 
            (dask.array)
 
        persist: bool
            = only applied if use_dask=True =
            should we use dask's persist() on data ?

        **kwargs goes to _QUADRANTCLASS.from_filename

        Returns
        -------
        class instance
        """
        # _read_filenames is a _Collection_ internal method. handle dask
        quadrants, filenames = cls._read_filenames(filenames,
                                                       as_path=as_path, use_dask=use_dask,
                                                       persist=persist, **kwargs)
        return cls(quadrants=quadrants, qids=qids)

    def to_fits(self, fileout, as_quadrants=False, overwrite=True,
                    **kwargs):
        """ dump the data (and header if any) into the given fitsfile

        Parameters
        ----------
        fileout: str
            path to where the data should be stored. (fits file)
            
        as_quadrant: bool
            should the data stored as 4 quadrant hdu images (True) 
            or one larger image (as_quadrant=False)
            
        overwrite: bool
            if fileout already exists, shall this overwrite it ?

        **kwargs goes to astropy.io.fits' writeto.

        Returns
        -------
        output of astropy.io.fits.writeto

        See also
        --------
        to_quadrant_fits: store the data as quadrant fitsfile
        """
        if not as_quadrants:
            out = self._to_fits(fileout, data=self.data, header=self.header,
                              overwrite=overwrite, **kwargs)
            
        else:
            quadrant_datas = self.get_quadrantdata()
            primary_hdu = [fits.PrimaryHDU(data=None, header=self.header)]
            quadrant_hdu = [fits.ImageHDU(qdata_) for qdata_ in quadrant_datas]
            hdulist = fits.HDUList(primary_hdu +quadrant_hdu) # list + list
            out = hdulist.writeto(fileout, overwrite=overwrite, **kwargs)
            
        return out

    def to_quadrant_fits(self, quadrantfiles, overwrite=True, from_data=True,
                    **kwargs):
        """ dump the current image into 4 different files: one for each quadrants 

        Parameters
        ----------
        quadrantfiles: list of string
            list of the filenames for each of the quadrants

        overwrite: bool
            if fileout already exists, shall this overwrite it ?

        from_data: bool
            option of get_quadrantdata(). If both self.data exists and self.qudrants, 
            should the data be taken from self.data (from_data=True) or from the 
            individual quadrants (from_data=False) using 
            ``self.quadrants[i].get_data()``
        
        Returns
        -------
        None

        See also
        --------
        to_fits: store the data as a unique fitsfile
        get_quadrantdata: get a list of 2d-array (on per quadrant)
        """
        if (nfiles:=len(quadrantfiles)) != 4:
            raise ValueError(f"you need exactly 4 files ; {nfiles} given.")

        quadrant_datas = self.get_quadrantdata(from_data=from_data)
        return self._to_quadrant_fits(quadrantfiles, quadrant_datas,
                                       header=self.header, overwrite=True,
                                       **kwargs)

    @classmethod
    def _to_quadrant_fits(cls, fileouts, datas, header, overwrite=True, **kwargs):
        """ dump the current image into 4 different files: one for each quadrants 

        = Internal method =

        Parameters
        ----------
        fileouts: list of string
            list of the filenames for each of the quadrants

        datas: list of 2d-array
            the list of quadrant data

        header: fits.Header of list of
            (list of) quadrant header. If not a list, the same header
            will be used for all the different images.

        overwrite: bool
            if fileout already exists, shall this overwrite it ?

        from_data: bool
            option of get_quadrantdata(). If both self.data exists and self.qudrants, 
            should the data be taken from self.data (from_data=True) or from the 
            individual quadrants (from_data=False) using 
            ``self.quadrants[i].get_data()``
        
        Returns
        -------
        None

        """
        if type(header) == fits.Header:
            header = [header]*4
        if len(header) !=4 or len(fileouts)!=4 or len(datas) !=4:
            raise ValueError("You need exactly 4 of each: data, header, fileouts")

        for file_, data_, header_ in zip(fileouts, datas, header):
            cls._quadrantclass._to_fits(file_, data=data_, header=header_,
                                            overwrite=overwrite, **kwargs)
    
    # --------- #
    #  LOADER   #
    # --------- #
    def load_data(self, **kwargs):
        """  get the data from the quadrants and set it to data. """
        data = self._quadrantdata_to_ccddata(**kwargs)
        self.set_data(data)

    # --------- #
    #  SETTER   #
    # --------- #
    def set_quadrant(self, quadrant, qid=None):
        """ set the quadrants to the instance.
        
        = It is unlikely you need to use that directly. =

        Parameters
        ----------
        quadrant: ztfimg.Quadrant
            attach a quadrant to the instance.
            will be added to ``self.quadrants[qid]``

        qid: int, None
            quadrant id. If not provided, it will be taken from 
            quadrant.qid.

        Returns
        -------
        None

        See also
        --------
        from_filenames: load the image given the list of its four quadrant filenames
        from_single_filename: build and load the instance given the filename of a single quadrant.
 
        """
        self._meta = None
        # dasked quadrants
        if "dask" in str( type(quadrant) ):
            self._use_dask = True
            self.quadrants[qid] = quadrant
            
        # quadrants
        else:
            if qid is None:
                qid = quadrant.qid

            self.quadrants[qid] = quadrant
            self._use_dask = quadrant.use_dask # updates

    # --------- #
    #  Call     #
    # --------- #
    def call_quadrants(self, what, *args, **kwargs):
        """ run the given input on quadrants 

        Parameters
        ----------
        what: str
            method or attribute of quadrants or anything 
            accessing through quadrant.get_value()

        *args goes to each quadrant.what() if method
        **kwargs goes to each quadrant.what() if method

        Returns
        -------
        list
            results of what called on each quadrant (1,2,3,4)
        """
        q0 = self.get_quadrant(1)
        if not hasattr(q0, what):
            kwargs["key"] = what
            what = "get_value"
        
        return self._call_down(what, *args, **kwargs)
    
    # --------- #
    #  GETTER   #
    # --------- #
    def get_quadrant(self, qid):
        """ get a quadrant (``self.quadrants[qid]``) """
        return self.quadrants[qid]

    def get_focalplane(self, use_dask=None, **kwargs):
        """ get the full focal plane (64 quadrants making 16 CCDs) 
        containing this quadrant

        (see self._FocalPlaneCLASS)

        Parameters
        ----------
        use_dask: bool, None
            should the ccd object be build using dask ? 
            if None, the current instance use_dask is used.
        
        as_path: bool
            should this assume that instance self.filename can 
            be directly transformed ? False should be favored 
            as it downloads the missing images if need.

        **kwargs goes the self._FocalPlaneCLASS.from_single_filename()

        Returns
        -------
        FocalPlane
            instance of self._FocalPlaneCLASS.
        """
        if use_dask is None:
            use_dask = self.use_dask
            
        if self.filename is None:
            filename = self.filenames[0]
        else:
            filename = self.filename
            
        return self._focalplaneclass.from_single_filename(filename,
                                                          use_dask=use_dask,
                                                          **kwargs)
    
    def get_quadrantheader(self):
        """ get a DataFrame gathering the quadrants's header 
        
        Returns
        -------
        DataFrame
            on column per quadrant
        """
        if not self.has_quadrants():
            return None
        
        qid_range = [1, 2, 3, 4]
        hs = [quadrant.get_header() for i in qid_range
                  if (quadrant:=self.get_quadrant(i)) is not None]
        df = pandas.concat(hs, axis=1)
        df.columns = qid_range
        return df

    def get_quadrantdata(self, from_data=False, rebin=None, rebin_stat="mean",
                             reorder=True, **kwargs):
        """ get the quadrant's data.
        
        Parameters
        ----------
        from_data: bool
            if self.data exists, should the quadrant be taken from it 
            (spliting it into 4 quadrants) ?
            if not, it will be taken from self.get_quadrant(i).get_data()
        
        
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

        **kwargs goes to each quadrant's get_data() (only if not from_data)

        Returns
        -------
        3d-array
            list of 2d-array.

        See also
        --------
        get_data: access the image data.

        """
        if not self.has_quadrants() and not self.has_data():
            warnings.warn("no quadrant and no data. None returned")
            return None

        if self.has_quadrants() and not from_data:
            qdata = self._get_subdata(rebin=rebin, reorder=reorder, **kwargs) # internal method of _Collection_
            
        else:
            q4 = self.data[:self.qshape[0],self.qshape[1]:]
            q1 = self.data[self.qshape[0]:,self.qshape[1]:]
            q3 = self.data[:self.qshape[0],:self.qshape[1]] 
            q2 = self.data[self.qshape[0]:,:self.qshape[1]]
            
            qdata = [q1,q2,q3,q4]
            if not reorder: # self.data is expected to be reodered
                qdata = [q[::-1,::-1] for q in qdata]
            
            if rebin is not None:
                npda = da if self.use_dask else np
                qdata = getattr(npda, rebin_stat)(rebin_arr(qdata, (rebin, rebin),
                                                        use_dask=self.use_dask),
                                              axis=(-2, -1))
        return qdata
        
    def get_data(self, rebin=None, rebin_quadrant=None,
                     rebin_stat="mean",
                     rebuild=False, persist=False,
                     **kwargs):
        """ get (a copy of) the data

        Parameters
        ----------
        rebin: int, None
            Shall the data be rebinned by square of size `rebin` ?
            None means no rebinning.
            (see details in rebin_stat)
            rebin must be a multiple of the image shape.
            for instance if the input shape is (6160, 6144)
            rebin could be 2,4,8 or 16
            = rebin is applied to the whole ccd image see 
            rebin_quadrant for rebinning at the quadrant level =

        rebin_quadrant: int, None
            rebinning (like rebin) but applied at the quadrant level
            (prior merging then as data)

        rebin_stat: str
            = applies only if rebin is not None =
            numpy (dask.array) method used for rebinning the data.
            For instance, if rebin=4 and rebin_stat = median
            the median of a 4x4 pixel will be used to form a new pixel.
            The dimension of the final image will depend on rebin.

        rebuild: bool
            if self.data exist and rebin_quadrant is None, then 
            ``self.data.copy()`` will be used. If rebuild=True, 
            then this will be re-build the data and ignore self.data

        persist: bool
            = only applied if self.use_dask is True =
            should we use dask's persist() on data ?

        Returns
        -------
        2d-array
            dask or numpy array
        
        See also
        --------
        get_quadrantdata: get a list of all the individual quadrant data.
    
        Examples
        --------
        get the ccd image and rebin it by 4
        >>> ccdimg.get_data(rebin=4).shape
        
        
        """
        # the normal case
        if self.has_data() and not rebuild and rebin_quadrant is None:
            data_ = self.data#.copy()
        else:
            data_ = self._quadrantdata_to_ccddata(rebin_quadrant=rebin_quadrant, **kwargs)
           
        if self.use_dask and persist:
            data_ = data_.persist()
            
        return data_


    def get_catalog(self, name, fieldcat=False, in_fov=False, drop_duplicate=True,
                        sourcekey="Source", use_dask=None, **kwargs):
        """ get a catalog for the image.

        This method calls down to the individual quadrant's get_catalog
        and merge them while updating their x,y position to make x,y 
        ccd pixels and not quadrant pixels.

        """
        cats = self._call_down("get_catalog",
                                name, fieldcat=fieldcat,
                                in_fov=in_fov, use_dask=use_dask,
                                **kwargs)
        
        # updates the quadrants. recall:
        # q2 | q1
        # --------
        # q3 | q4
        cats[0]["x"] += self.qshape[1]
        cats[3]["x"] += self.qshape[1]
        cats[0]["y"] += self.qshape[0]
        cats[1]["y"] += self.qshape[0]
        # -> merging
        cat = pandas.concat(cats)
        
        if drop_duplicate:
            if sourcekey not in cat:
                warnings.warn(f"cannot drop duplicated based on {sourcekey}: not in catalog")
            else:
                cat = cat.groupby(sourcekey).first().reset_index()
        elif not in_fov:
            warnings("no duplicate drop and in_fov=False, you maye have multiple entries of the same source.")

        return cat

    def get_center(self, system="xy", reorder=True):
        """ get the center of the image
        Note: this uses the 1st quadrant wcs solution if one is needed.

        Parameters
        ----------
        system: string
            coordinate system.
            # for ccds xy and ij are the same
            - xy / ij: image pixel coordinates 
            - radec: sky coordinates (in deg)
            - uv: camera coordinates (in arcsec)

        reorder: bool
            should this provide the coordinates assuming
            normal ordering (+ra right, +dec up) (True) ?
            = leave True if unsure = 

        Returns
        -------
        2d-array
            coordinates (see system)
        """
        # reminder | center is -0.5,-0.5 of q1
        # q2 | q1
        # --------
        # q3 | q4s
        if system in ["xy", "ij"]:
            return self.qshape
        
        if system == "radec":
            return self.get_quadrant(1).xy_to_radec(0,0).squeeze()
        
        if system == "uv":
            return self.get_quadrant(1).xy_to_uv(0,0).squeeze()
        
        raise ValueError(f"{system} system not recognized. 'xy', 'radec' or 'uv'")

    def get_corners(self, system="xy", reorder=True):
        """ get the corners of the image.
        Note: this uses the 1st quadrant wcs solution if one is needed.

        Parameters
        ----------
        system: str
            coordinate system.
            - xy/ij: image pixel coordinates
            - radec: sky coordinates (in deg)
            - uv: camera coordinates (in arcsec)

        reorder: bool
            = ignored if system='xy' =    
            should this provide the coordinates assuming
            normal ordering (+ra right, +dec up) (True) ?
            Leave default if unsure. 

        Returns
        -------
        2d-array
            lower-left, lower-right, upper-right, upper-left
        """
        # 
        corners = np.stack([[0,0], 
                            [self.shape[1],0],
                            [self.shape[1],self.shape[0]],
                            [0,self.shape[0]]])
    
        if system in ['xy', "ij"]:
            return corners
                
        if system == "radec":
            return self.get_quadrant(1).ij_to_radec(*corners.T, reorder=True).T
        
        if system == "uv":
            return self.get_quadrant(1).ij_to_uv(*corners.T, reorder=True).T

        raise ValueError(f"{system} system not recognized. 'xy', 'ij', 'radec' or 'uv'")


    
    # - internal get
    def _quadrantdata_to_ccddata(self, rebin_quadrant=None, qdata=None, **kwargs):
        """ combine the ccd data into the quadrants"""
        # numpy or dask.array ?
        npda = da if self.use_dask else np
        if qdata is None:
            qdata = self.get_quadrantdata(rebin=rebin_quadrant, **kwargs)
        
        # ccd structure
        # q2 | q1
        # q3 | q4
        ccd_up = npda.concatenate([qdata[1], qdata[0]], axis=1)
        ccd_down = npda.concatenate([qdata[2], qdata[3]], axis=1)
        ccd = npda.concatenate([ccd_down, ccd_up], axis=0)
        return ccd

    def show_footprint(self, values="qid", ax=None, 
                        system="ij", cmap="coolwarm",
                        vmin=None, vmax=None, 
                        quadrant_id="qid", #in_deg=True, 
                        **kwargs):
        """ illustrate the image footprint

        Parameters
        ----------
        values: str, None, array, dict, pandas.Series
            values to be displaid as facecolor in the image.
            - str: understood as a quadrant properties 
                   (using call_quadrants)
            - None: (or 'None') no facecolor (just edge)
            - array: value to be displaid. size of 4.
            - dict: {id_: value} # empty id_ will be set to None
            - pandas.Series: empty id_ will be set to None

        ax: matplotlib.Axes
            axes where to draw the plot

        system: str
            coordinates system: 
            - xy: quadrant 
            - ij: ccd (favored)
            - uv: focalplane

        cmap: str or matplotlib's cmap
            colormap 

        vmin, vmax: 
            boundaries for the colormap

        quadrant_id: str or None
            value indicated in the quadrants. 
            None means no text written.

        **kwargs goes to matplotlib.patches.Polygon

        Returns
        -------
        fig
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        from matplotlib.colors import Normalize

        # --------
        # Figures
        if ax is None:
            fig = plt.figure(figsize=[6,6])
            ax = fig.add_axes([0.15,0.15,0.8,0.8])
        else:
            fig = ax.figure
            
        prop = dict(lw=0, alpha=0.8, zorder=3)

        # ---------
        # colormap
        if cmap is None:
            cmap = plt.get_cmap("tab10")

        if type(cmap) is str:
            cmap =plt.get_cmap(cmap)

        # Values
        if type(values) is str:
            if values == "None": # like in matplotlib
                values = None
            else:
                values = self.call_quadrants(values) 
        elif values is not None:
            # reindex
            values = pandas.Series(values).reindex( np.arange(4) ).values
        
        if values is not None:
            if vmin is None: 
                vmin = np.nanmin(values)
            if vmax is None: 
                vmax = np.nanmax(values)
            norm = Normalize(vmin=vmin, vmax=vmax)


        # using center and corner as debug, center could be derived from corners. 
        centers = np.stack(self.call_quadrants("get_center", system=system))
        corners = np.stack(self.call_quadrants("get_corners", system=system))

        ids = self.call_quadrants(quadrant_id) # for labels
        # loop over the values
        if values is None:
            values = [None] * len(ids)

        for value_, ids_, corners_, centers_ in zip(values, ids, corners, centers):
            if value_ is None or np.isnan(value_):
                facecolor = "None"
                prop["lw"] = 1
                prop["edgecolor"] = "0.7"
            else:
                facecolor = cmap(norm(value_))

            p = Polygon(corners_, facecolor=facecolor, **{**prop, **kwargs})
            ax.add_patch(p)
            if quadrant_id is not None:
                ax.text(*centers_, ids_, va="center", ha="center", color="w", zorder=9)

        if system in ["xy","ij"]:
            ax.set_xlim(-10, self.shape[1]+10)
            ax.set_ylim(-10, self.shape[0]+10)
        elif system in ["uv"]:
            # requires limits for polygon only plots
            ax.set_xlim(-(3.8*3600),(3.8*3600))
            ax.set_ylim(-(3.8*3600),(3.8*3600))

        # labels
    #    ax.set_ylabel(f"v {label_unit}", fontsize="large")
    #    ax.set_xlabel(f"u {label_unit}", fontsize="large")
        # Fancy
        clearwhich = ["left","right","top","bottom"]
        [ax.spines[which].set_visible(False) for which in clearwhich]
        ax.tick_params(labelsize="small", labelcolor="0.7", color="0.7")

        return fig
    # =============== #
    #  Properties     #
    # =============== #
    @property
    def data(self):
        """ the image data. """
        if not self.has_data():
            if self.has_quadrants("all"):
                warnings.warn("use get_data() to fetch data from the quadrants")
            return None
            # do not load by default too dangerous for they are options in get_data
            #self.load_data()

        return self._data
        
    @property
    def quadrants(self):
        """ dictionnary of the quadrants, keys are the quadrant id"""
        if not hasattr(self, "_quadrants"):
            self._quadrants = {k: None for k in [1, 2, 3, 4]}
        return self._quadrants

    @property
    def _images(self):
        """ Internal property for _Collection_ methods """
        return list( self.quadrants.values() )
    
    def has_quadrants(self, logic="all"):
        """ are (all/any, see option) quadrant loaded ? """
        is_none = [v is not None for v in self.quadrants.values()]
        return getattr(np, logic)(is_none)

    @classproperty
    def qshape(cls):
        """ shape of an individual ccd quadrant """
        return cls._quadrantclass.shape

    @property
    def ccdid(self):
        """ ccd id (from header) """
        return self.get_quadrant(1).ccdid

    @property
    def filenames(self):
        """ list of the filename of the different quadrants constituing the data. """
        return [q.filename for qid, q in self.quadrants.items()]

    @property
    def filepaths(self):
        """ list of the filepath of the different quadrants constituing the data. """
        return [q.filepath for qid, q in self.quadrants.items()]

    
    # family
    @classproperty
    def _quadrantclass(cls):
        """ """
        if type( cls._QUADRANTCLASS ) is str:            
            try: # from this .py
                return eval( cls._QUADRANTCLASS )
            except: # from the library 
                exec(f"from .__init__ import {cls._QUADRANTCLASS}")
                return eval( cls._QUADRANTCLASS )

        return cls._QUADRANTCLASS

    @classproperty
    def _focalplaneclass(cls):
        """ """
        if type(cls._FocalPlaneCLASS) is str:
            try: # from this .py
                return eval( cls._FocalPlaneCLASS )
            except: # from the library 
                exec(f"from .__init__ import {cls._FocalPlaneCLASS}")
                return eval( cls._FocalPlaneCLASS )

        
        return cls._FocalPlaneCLASS
    
# -------------- #
#                #
#  Focal Plane   #
#                #
# -------------- #
class FocalPlane(Image, _Collection_):
    
    _COLLECTION_OF = CCD
    # Family
    _CCDCLASS = "CCD"
    
    # Basically a CCD collection
    def __init__(self, ccds=None, ccdids=None,
                     data=None, header=None, **kwargs):
        """ 

        See also
        --------
        from_filename: load the instance given a filename 
        from_filenames: load the image given the list of its quadrant filenames
        from_single_filename: build and load the instance given the filename of a single quadrant.
        """
        _ = super().__init__(data=data, header=header)
        
        if ccds is not None:
            if ccdids is None:
                ccdids = [None]*len(ccds)
                
            elif len(ccds) != len(ccdids):
                raise ValueError("ccds and ccdids must have the same size.")

            _ = [self.set_ccd(ccd_, ccdid=ccdid_, **kwargs)
                     for ccd_, ccdid_ in zip(ccds, ccdids)]

    # =============== #
    #  I/O            #
    # =============== #
    @classmethod
    def from_filenames(cls, filenames, as_path=True,
                           rcids=None, use_dask=False,
                           persist=False, **kwargs):
        """ loads the instance from a list of the quadrant files.

        Parameters
        ----------
        filanames: list of four str
            filenames for the four quadrants 
        
        as_path: bool
            Set this to true if the input file is not the fullpath but you need
            ztfquery.get_file() to look for it for you.

        rcid: list of int
            list of the rcid (0->63) for the input filenames

        use_dask: bool
            Should dask be used ? The data will not be loaded but delayed 
            (dask.array)
 
        persist: bool
            = only applied if use_dask=True =
            should we use dask's persist() on data ?

        **kwargs goes to _CCDCLASS.from_filenames

        Returns
        -------
        class instance
        """
        if rcids is None:
            rcids = np.arange(0, 64)

        data = pandas.DataFrame({"path": filenames, "rcid": rcids})
        #Get the qid and ccdid associated to the rcid
        data = data.merge(pandas.DataFrame(data["rcid"].apply(rcid_to_ccdid_qid).tolist(),
                                           columns=["ccdid", "qid"]),
                          left_index=True, right_index=True)
        # Get the ccdid list sorted by qid 1->4
        ccdidlist = data.sort_values("qid").groupby("ccdid")[
                                     "path"].apply(list)

        ccds, filenames = cls._read_filenames(ccdidlist.values, as_path=as_path,
                                                 use_dask=use_dask, persist=persist, **kwargs)
        ccds = [cls._ccdclass.from_filenames(qfiles, qids=[1, 2, 3, 4], as_path=as_path,
                                                 use_dask=use_dask, persist=persist, **kwargs)
                for qfiles in ccdidlist.values]
            
        return cls(ccds, np.asarray(ccdidlist.index, dtype="int"),
                   use_dask=use_dask)

    @classmethod
    def from_single_filename(cls, filename, as_path=True, use_dask=False,
                                 persist=False, **kwargs):
        """ given a single quadrant file, this fetchs the missing ones and loads the instance.

        Parameters
        ----------
        filename: str
            filename of a single *quadrant* filename.
            This will look for the 3 missing to build the ccd.
            
        as_path: bool -optional-
            Set this to true if the input file is not the fullpath but you need
            ztfquery.get_file() to look for it for you.
        
        use_dask: bool, optional
            Should dask be used ? The data will not be loaded but delayed 
            (dask.array)

        persist: bool, optional
            = only applied if use_dask=True =
            should we use dask's persist() on data ?

        **kwargs goes to _CCDCLASS.from_single_filename -> _QUADRANTCLASS.from_filename()

        Returns
        -------
        class instance

        Examples
        --------
        >>> ccdimg = CCD.from_single_filename("ztf_20220704387176_000695_zr_c11_o_q3_sciimg.fits", as_path=False, use_dask=False)
        """
        import re
        ccdids = range(1, 17)
        ccds = [cls._ccdclass.from_single_filename(re.sub("_c(\d\d)_*", f"_c{i:02d}_", filename),
                                                   as_path=as_path, use_dask=use_dask,
                                                   persist=persist, **kwargs)
                for i in ccdids]
        return cls(ccds, ccdids)

    # =============== #
    #   Methods       #
    # =============== #
    def set_ccd(self, ccd, ccdid=None):
        """ attach ccd images to the instance. 

        Parameters
        ----------
        ccd: CCD
            CCD (or child of) object to be attached.
            Could be a delayed
           
        ccdid: int or None
            id of the ccd.
        
        Returns
        -------
        None
            sets self.ccds
        """

        self._meta = None
        if "dask" in str( type(ccd) ):
            self._use_dask = True
            self.ccds[ccdid] = ccd
            
        else:
            if ccdid is None:
                ccdid = ccd.ccdid
                
            self.ccds[ccdid] = ccd
            self._use_dask  = ccd.use_dask

    # -------- #
    #  CALL    #
    # -------- #
    def call_ccds(self, what, *args, **kwargs):
        """ run the given input on ccds

        Parameters
        ----------
        what: str
            method or attribute of quadrants

        *args goes to each quadrant.what() if method
        **kwargs goes to each quadrant.what() if method

        Returns
        -------
        list
            results of what called on each quadrant (1->16)
        """        
        return self._call_down(what, *args, **kwargs)
    
    def call_quadrants(self, what, *args, **kwargs):
        """ run the given input on quadrants 

        Parameters
        ----------
        what: str
            method or attribute of quadrants or anything 
            accessing through quadrant.get_value()

        *args goes to each quadrant.what() if method
        **kwargs goes to each quadrant.what() if method

        Returns
        -------
        list
            results of what called on each quadrant (0->63)
        """
        import inspect
        q0 = self.get_quadrant(0)

        if hasattr(q0, what):
            is_func = inspect.ismethod( getattr(q0, what) )
        else: # use the get_value method
            kwargs["key"] = what
            what = "get_value"
            is_func = True
            
        if is_func: # is func ?
            return [getattr(q, what)(*args, **kwargs)
                        for ccdid, ccd in self.ccds.items()
                        for qid, q in ccd.quadrants.items()]
        
        return [getattr(q, what)
                    for ccdid, ccd in self.ccds.items()
                    for qid, q in ccd.quadrants.items()]
    
    # -------- #
    #  GETTER  #
    # -------- #
    def get_ccd(self, ccdid):
        """ get the ccd (self.ccds[ccdid])"""
        return self.ccds[ccdid]

    def get_quadrant(self, rcid):
        """ get the quadrant (get the ccd and then get its quadrant """
        ccdid, qid = rcid_to_ccdid_qid(rcid)
        return self.get_ccd(ccdid).get_quadrant(qid)

    def get_quadrantheader(self, rcids="all"):
        """ returns a DataFrame of the header quadrants (rcid) """
        if rcids in ["*", "all"]:
            rcids = np.arange(64)

        hs = [self.get_quadrant(i).get_header() for i in rcids]
        df = pandas.concat(hs, axis=1)
        df.columns = rcids
        return df

    @staticmethod
    def _get_datagap(which, rebin=None, fillna=np.NaN):
        """ horizontal (or row) = between rows """
        # recall: CCD.SHAPE 3080*2, 3072*2
        if which in ["horizontal", "row", "rows"]:
            hpixels = 672
            vpixels = CCD.SHAPE[1]
        else:
            hpixels = CCD.SHAPE[0]
            vpixels = 488

        if rebin is not None:
            hpixels /= rebin
            vpixels /= rebin

        return hpixels, vpixels

    def get_data(self, rebin=None, incl_gap=True, persist=False, ccd_coef=None, **kwargs):
        """ get data. """

        if ccd_coef is None:
            ccd_coef = np.ones(16)
            
        elif (size:= len(ccd_coef)) != 16:
            raise ValueError(f"size of ccd_coef must be 16. {size} given.")
        
        # Merge quadrants of the 16 CCDs
        prop = {**dict(rebin=rebin), **kwargs}

        npda = da if self.use_dask else np

        if not incl_gap:
            line_1 = getattr(npda, "concatenate")((self.get_ccd(4).get_data(**prop)*ccd_coef[3],
                                                   self.get_ccd(3).get_data(**prop)*ccd_coef[2],
                                                   self.get_ccd(2).get_data(**prop)*ccd_coef[1],
                                                   self.get_ccd(1).get_data(**prop)*ccd_coef[0]
                                                  ),
                                                    axis=1)
            
            line_2 = getattr(npda, "concatenate")((self.get_ccd(8).get_data(**prop)*ccd_coef[7],
                                                   self.get_ccd(7).get_data(**prop)*ccd_coef[6],
                                                   self.get_ccd(6).get_data(**prop)*ccd_coef[5],
                                                   self.get_ccd(5).get_data(**prop)*ccd_coef[4]
                                                  ),
                                                    axis=1)
            
            line_3 = getattr(npda, "concatenate")((self.get_ccd(12).get_data(**prop)*ccd_coef[11],
                                                   self.get_ccd(11).get_data(**prop)*ccd_coef[10],
                                                   self.get_ccd(10).get_data(**prop)*ccd_coef[9],
                                                   self.get_ccd( 9).get_data(**prop)*ccd_coef[8]
                                                   ),
                                                    axis=1)
            
            line_4 = getattr(npda, "concatenate")((self.get_ccd(16).get_data(**prop)*ccd_coef[15],
                                                   self.get_ccd(15).get_data(**prop)*ccd_coef[14],
                                                   self.get_ccd(14).get_data(**prop)*ccd_coef[13],
                                                   self.get_ccd(13).get_data(**prop)*ccd_coef[12]
                                                    ),
                                                    axis=1)

            mosaic = getattr(npda, "concatenate")( (line_1, line_2, line_3, line_4),
                                                    axis=0)
            
        else:
            line_1 = getattr(npda, "concatenate")((self.get_ccd(4).get_data(**prop)*ccd_coef[3],
                                                    da.ones(self._get_datagap("columns", rebin=rebin))*np.NaN,
                                                   self.get_ccd(3).get_data(**prop)*ccd_coef[2],
                                                    da.ones(self._get_datagap("columns", rebin=rebin))*np.NaN,
                                                   self.get_ccd(2).get_data(**prop)*ccd_coef[1],
                                                    da.ones(self._get_datagap("columns", rebin=rebin))*np.NaN,
                                                   self.get_ccd(1).get_data(**prop)*ccd_coef[0],
                                                       ),
                                                    axis=1)
            
            line_2 = getattr(npda, "concatenate")((self.get_ccd(8).get_data(**prop)*ccd_coef[7],
                                                    da.ones(self._get_datagap("columns", rebin=rebin))*np.NaN,
                                                   self.get_ccd(7).get_data(**prop)*ccd_coef[6],
                                                    da.ones(self._get_datagap("columns", rebin=rebin))*np.NaN,
                                                   self.get_ccd(6).get_data(**prop)*ccd_coef[5],
                                                    da.ones(self._get_datagap("columns", rebin=rebin))*np.NaN,
                                                   self.get_ccd(5).get_data(**prop)*ccd_coef[4]
                                                    ),
                                                    axis=1)
            
            line_3 = getattr(npda, "concatenate")((self.get_ccd(12).get_data(**prop)*ccd_coef[11],
                                                    da.ones(self._get_datagap("columns", rebin=rebin))*np.NaN,
                                                   self.get_ccd(11).get_data(**prop)*ccd_coef[10],
                                                    da.ones(self._get_datagap("columns", rebin=rebin))*np.NaN,
                                                   self.get_ccd(10).get_data(**prop)*ccd_coef[9],
                                                    da.ones(self._get_datagap("columns", rebin=rebin))*np.NaN,
                                                   self.get_ccd(9).get_data(**prop)*ccd_coef[8]
                                                       ),
                                                    axis=1)
            
            line_4 = getattr(npda, "concatenate")((self.get_ccd(16).get_data(**prop)*ccd_coef[15],
                                                   da.ones(self._get_datagap("columns", rebin=rebin))*np.NaN,
                                                   self.get_ccd(15).get_data(**prop)*ccd_coef[14],
                                                   da.ones(self._get_datagap("columns", rebin=rebin))*np.NaN,
                                                   self.get_ccd(14).get_data(**prop)*ccd_coef[13],
                                                   da.ones(self._get_datagap("columns", rebin=rebin))*np.NaN,
                                                   self.get_ccd(13).get_data(**prop)*ccd_coef[12]
                                                    ),
                                                    axis=1)
            
            size_shape = self._get_datagap("rows", rebin=rebin)[0]
            mosaic = getattr(npda, "concatenate")((line_1,
                                                   da.ones(
                                                      (size_shape, line_1.shape[1]))*np.NaN,
                                                   line_2,
                                                   da.ones(
                                                      (size_shape, line_1.shape[1]))*np.NaN,
                                                   line_3,
                                                   da.ones(
                                                      (size_shape, line_1.shape[1]))*np.NaN,
                                                   line_4),
                                                    axis=0)
        if self.use_dask and persist:
            return mosaic.persist()

        return mosaic

    # -------- #
    #  PLOTS   #
    # -------- #
    def show_footprint(self, values="id", ax=None, 
                           level="quadrant", cmap="coolwarm",
                           vmin=None, vmax=None, 
                           incl_ids=True, in_deg=True, 
                           **kwargs):
        """ illustrate the image footprint

        Parameters
        ----------
        values: str, None, array, dict, pandas.Series
            values to be displaid as facecolor in the image.
            - str: understood as a `level` properties 
                   (using call_quadrants or call_ccds)
                   Special case: 'id' means:
                   - ccdid if level='ccd' 
                   - rcid if level='quadrant
            - None: (or 'None') no facecolor (just edge)
            - array: value to be displaid. Size must match that
                of the level.
            - dict: {id_: value} # empty id_ will be set to None
            - pandas.Series: empty id_ will be set to None


        ax: matplotlib.Axes
            axes where to draw the plot

        level: str
            'ccd' (16) or 'quadrant' (64) 

        cmap: str or matplotlib's cmap
            colormap 

        vmin, vmax: 
            boundaries for the colormap

        incl_ids: bool
            should the name of the (ccd or quadrant) id
            be shown ?

        in_deg: bool
            should the footprint be shown in deg (True) or
            in arcsec (False) ? 
            note: 1 pixel ~ 1 arcsec
        
        **kwargs goes to matplotlib.patches.Polygon

        Returns
        -------
        fig
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        from matplotlib.colors import Normalize
        
        # -------
        # level
        if level == "ccd":
            call_func = self.call_ccds
            nimages = 16
            id_ = "ccdid"
        elif level == "quadrant":
            call_func = self.call_quadrants
            nimages = 64
            id_ = "rcid"
        else:
            raise ValueError(f"level can be 'ccd' or 'quadrant', {level} given")

        # --------
        # Figures
        if ax is None:
            fig = plt.figure(figsize=[6,6])
            ax = fig.add_axes([0.15,0.15,0.8,0.8])
        else:
            fig = ax.figure
        prop = dict(lw=0, alpha=0.8, zorder=3)

        # ---------
        # colormap
        if cmap is None:
            cmap = plt.get_cmap("tab10")

        if type(cmap) is str:
            cmap =plt.get_cmap(cmap)

        # degree or arcsec
        if in_deg:
            coef = 1/3600
            label_unit = "[in deg]"
        else:
            coef = 1
            label_unit = "[in arcsec]"

        # Values
        if type(values) is str:
            if values == 'id':
                values = id_

            if values == "None": # like in matplotlib
                values = None
            else:
                values = call_func(values)
        elif values is not None:
            # reindex
            values = pandas.Series(values).reindex( np.arange(nimages) ).values
        
        if values is not None:
            if vmin is None: 
                vmin = np.nanmin(values)
            if vmax is None: 
                vmax = np.nanmax(values)
            norm = Normalize(vmin=vmin, vmax=vmax)


        # using center and corner as debug, center could be derived from corners. 
        centers = np.stack(call_func("get_center", system="uv")) * coef
        corners = np.stack(call_func("get_corners", system="uv")) * coef
        ids = call_func(id_) # for labels
        # loop over the values
        if values is None:
            values = [None] * len(ids)

        for value_, ids_, corners_, centers_ in zip(values, ids, corners, centers):
            if value_ is None or np.isnan(value_):
                facecolor = "None"
                prop["lw"] = 1
                prop["edgecolor"] = "0.7"
            else:
                facecolor = cmap(norm(value_))                

            p = Polygon(corners_, facecolor=facecolor, **{**prop, **kwargs})
            ax.add_patch(p)
            if incl_ids:
                ax.text(*centers_, ids_, va="center", ha="center", color="w", zorder=9)

        # requires limits for polygon only plots
        ax.set_xlim(-(3.8*3600) * coef, (3.8*3600) * coef)
        ax.set_ylim(-(3.8*3600) * coef, (3.8*3600) * coef)
        # labels
        ax.set_ylabel(f"v {label_unit}", fontsize="large")
        ax.set_xlabel(f"u {label_unit}", fontsize="large")
        # Fancy
        clearwhich = ["left", "right","top","bottom"]
        [ax.spines[which].set_visible(False) for which in clearwhich]
        ax.tick_params(labelsize="small", labelcolor="0.7", color="0.7")

        return fig
    
    # =============== #
    #  Properties     #
    # =============== #    
    @property
    def ccds(self):
        """ dictionary of the ccds {ccdid:CCD, ...}"""
        if not hasattr(self, "_ccds"):
            self._ccds = {k: None for k in np.arange(1, 17)}

        return self._ccds

    def has_ccds(self, logic="all"):
        """ test if (any/all see option) ccds are loaded. """
        is_none = [v is not None for v in self.ccds.values()]
        return getattr(np, logic)(is_none)

    @property
    def _images(self):
        """ Internal property for _Collection_ methods """
        return list( self.ccds.values() )
    
    @property
    def filenames(self):
        """ list of the filename of the different quadrants constituing the data. """
        return self.call_quadrants("filename")
    
    @property
    def filepaths(self):
        """ list of the filename of the different quadrants constituing the data. """
        return self.call_quadrants("filepath")

    
    @classproperty
    def shape_full(cls):
        """ shape with gap"""
        print("gap missing")
        return cls.shape

    @classproperty
    def shape(cls):
        """ shape without gap"""
        return cls.ccdshape*4

    @classproperty
    def ccdshape(cls):
        """ """
        return cls.qshape*2

    @classproperty
    def qshape(cls):
        """ """
        return cls._ccdclass.qshape
    
    # family
    @classproperty
    def _ccdclass(cls):
        """ """
        if type( cls._CCDCLASS ) is str:
            try: # from this .py
                return eval( cls._CCDCLASS )
            except: # from the library 
                exec(f"from .__init__ import {cls._CCDCLASS}")
                return eval( cls._CCDCLASS )

        return cls._CCDCLASS
