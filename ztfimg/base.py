import numpy as np
import pandas
import warnings
import dask
import dask.array as da
import dask.dataframe as dd

from .utils.tools import rebin_arr, parse_vmin_vmax, ccdid_qid_to_rcid, rcid_to_ccdid_qid
from .utils.decorators import classproperty

__all__ = ["Image", "Quadrant", "CCD", "FocalPlane"]

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
    def _read_data(cls, filename, use_dask=True, persist=False, ext=None):
        """ assuming fits format."""
        from astropy.io.fits import getdata
        if use_dask:
            # - Data
            data = da.from_delayed( dask.delayed( getdata) (filename, ext=ext),
                                   shape=cls.SHAPE, dtype="float32")
            if persist:
                data = data.persist()

        else:
            data = getdata(filename, ext=ext)

        return data

    @staticmethod
    def _read_header(filename, use_dask=True, persist=False, ext=None):
        """ assuming fits format. """
        from astropy.io.fits import getheader
        
        if use_dask:
            header = dask.delayed(getheader)(filename, ext=ext)
            if persist:
                header = header.persist()
        else:
            header = getheader(filename, ext=ext)
            
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
    def _get_filename(filename, as_path=True, use_dask=False, **kwargs):
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
            
        filename = cls._get_filename(filename, as_path=as_path, use_dask=use_dask)
        data = cls._read_data(filename, use_dask=use_dask, persist=persist)
        header = cls._read_header(filename, use_dask=dask_header, persist=persist)
        
        # self
        this = cls.from_data(data=data, header=header, **kwargs)
        this._filename = filename
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

    def get_header(self):
        """ get the current header

        this is a shortcut to self.header

        Returns
        -------
        fits.header
            whatever is in self.header 
        """
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
                     data=None, dataprop={},
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

        radius: float, list
            size (radius) of the aperture. This could be a list of radius.
            
        data: 2d-array, None
            if you want to apply the aperture photometry on this specific image, provide it.
            otherwhise, ``data = self.get_data(**dataprop)`` is used
            
        dataprop: dict
            = ignored if data is given =
            kwargs used to get the data. 
            ``data = self.get_data(**dataprop)``

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
            data = self.get_data(**dataprop)
            
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
                     dataprop={},
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
            
        dataprop: dict
            = ignored if data is given =
            kwargs used to get the data. 
            ``data = self.get_data(**dataprop)``

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
        """

        from .utils.tools import get_aperture

        if use_dask is None:
            use_dask = "dask" in str(type(data))

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
                
    def getcat_aperture(self, catdf, radius,
                            data=None, dataprop={},
                            xykeys=["x", "y"], join=True,
                            as_dataframe=True,
                            **kwargs):
        """ measures the aperture (using get_aperture) using a catalog dataframe as input

        # Careful, the indexing is reset (becomes index column) when joined. #

        Parameters
        ----------
        catdf: pandas.DataFrame
            dataframe containing, at minimum the x and y centroid positions

        radius: float, list
            size (radius) of the aperture. This could be a list of radius.

        data: 2d-array, None
            if you want to apply the aperture photometry on this specific image, provide it.
            otherwhise, ``data = self.get_data(**dataprop)`` is used

        dataprop: dict
            = ignored if data is given =
            kwargs used to get the data. 
            ``data = self.get_data(**dataprop)``

        xykeys: list of two str
            name of the x and y columns in the input dataframe

        join: bool, optional
            shall the returned dataframe be a new dataframe joined
            to the input one, or simply the aperture dataframe?

        as_dataframe: bool
            return format.
            If As DataFrame, this will be a dataframe with
            3xn-radius columns (f_0...f_i, f_0_e..f_i_e, f_0_f...f_i_f)
            standing for fluxes, errors, flags.

        **kwargs goes to self.get_aperture

        Returns
        -------
        2d-array or dataframe
           flux, error and flag for each coordinates and radius.


        See also
        --------
        get_aperture: run the aperture photometry given list of pixel coordinates

        
        """
        if data is None:
            data = self.get_data(**dataprop)
            
        return self._getcat_aperture(catdf, data, radius,
                                     xykeys=["x", "y"],
                                     join=join, **kwargs)

    @classmethod
    def _getcat_aperture(cls, catdf, data, radius,
                         xykeys=["x", "y"],
                         join=True, **kwargs):
        """ """
        if join:
            kwargs["as_dataframe"] = True
            
        x, y = catdf[xykeys].values.T
        fdata = cls._get_aperture(data, x, y, radius,
                                  **kwargs)
        if join:
            # the index and drop is because dask.DataFrame do not behave as pandas.DataFrame
            return catdf.reset_index().join(fdata)

        return fdata

    def get_catalog(self, which="ps1", fieldcat=False, **kwargs):
        """ get a catalog for the image
        
        Parameters
        ----------
        which: str
            name of a the catalog.

        fieldcat: bool
            is the catalog you are looking for a "field catalog" ?
            (see catalog.py)

        Returns
        -------
        DataFrame
            catalog dataframe. The image x, y coordinates columns will be added
            using the radec_to_xy method. If not available, NaN will be set.
            
        """
        if fieldcat:
            from .catalog import get_field_catalog
            cat = get_field_catalog(which,
                                    fieldid=self.get_value("fieldid"),
                                    rcid=self.get_value("rcid", None), 
                                    ccdid=self.get_value("ccdid", None), # ignored if rcid is not None
                                    **kwargs)
        else:
            raise NotImplementedError("Only fieldcat have been implemented.")

        # Adding x, y coordinates
        # convertion available ?
        if hasattr(self, "radec_to_xy"):
            x, y = self.radec_to_xy(*cat[["ra","dec"]].values.T)
        else:
            x, y = np.NaN, np.NaN

        cat["x"] = x
        cat["y"] = y
        
        return cat
        
    # -------- #
    # PLOTTER  #
    # -------- #
    def show(self, ax=None, colorbar=True, cax=None, apply=None,
                 data=None,
                 vmin="1", vmax="99",
                 rebin=None, dataprop={},
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
            otherwhise, ``data = self.get_data(rebin=rebin, **dataprop)`` is shown.

        vmin, vmax: str, float
            minimum and maximum value for the colormap.
            string are interpreted as 'percent of data'. 
            float or int are understood as 'use as such'

        rebin: int, None
            by how much should the data be rebinned when accessed ?
            (see details in get_data())
            
        dataprop: dict
            used as kwargs of get_data() when accessing the data 
            to be shown.

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
            data = self.get_data(rebin=rebin, **dataprop)
            
        if "dask" in str(type(data)):
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

    # -------- #
    # PLOTTER  #
    # -------- #
    def _compute_header(self):
        """ """
        if self._use_dask and "dask" in str(type(self._header)):
            self._header = self._header.compute()

    def _compute_data(self):
        """ """
        if self._use_dask and "dask." in str(type(self._data)):
            self._data = self._data.compute()

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
    def header(self, compute=True):
        """ header of the data."""
        if not hasattr(self, "_header"):
            return None
        # Computes the header only if necessary
        self._compute_header()
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
            return None
        return self._filename

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
    COLLECTION_OF = Image

    @classmethod
    def _read_filenames(cls, filenames, use_dask=True, 
                           as_path=False, persist=False, **kwargs):
        """ """
        filenames = np.atleast_1d(filenames).tolist()
        # it's the whole COLLECTION_OF that is dask, not inside it.
        prop = {**dict(as_path=as_path, use_dask=use_dask), **kwargs}

        # dask is called inside this.
        images = [cls.COLLECTION_OF.from_filename(filename, **prop) for filename in filenames]
        if persist and use_dask:
            images = [i.persist() for i in images]
            
        return images, filenames

    def _get_subdata(self, **kwargs):
        """ get a stack of the data from get_data collection of """
        datas = self._call_down("get_data",True, **kwargs)
        if self.use_dask:
            return da.stack([da.from_delayed(f_, shape=self.COLLECTION_OF.shape,
                                                 dtype="float32")
                                     for f_ in datas])
        return np.stack(datas)

    # -------- #
    # INTERNAL #
    # -------- #
    def _map_down(self, what, margs, *args, **kwargs):
        """ """
        return [getattr(img, what)(marg, *args, **kwargs)
                for img, marg in zip(self._images, margs)]
    
    def _call_down(self, what, isfunc, *args, **kwargs):
        """ """
        if isfunc:
            return [getattr(img, what)(*args, **kwargs) for img in self._images]
        return [getattr(img, what) for img in self._images]
    
# -------------- #
#                #
#   QUADRANT     #
#                #
# -------------- #
class Quadrant(Image):
    SHAPE = (3080, 3072)

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
        
# -------------- #
#                #
#     CCD        #
#                #
# -------------- #

class CCD( Image, _Collection_):
    # Basically a Quadrant collection

    SHAPE = (3080*2, 3072*2)
    COLLECTION_OF = Quadrant    
    _QUADRANTCLASS = Quadrant

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
    def from_single_filename(cls, filename, as_path=True, use_dask=True, persist=False, **kwargs):
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
                           qids=[1, 2, 3, 4], use_dask=True,
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
            cls._QUADRANTCLASS._to_fits(file_, data=data_, header=header_,
                                            overwrite=overwrite, **kwargs)
    
    # --------- #
    #  LOADER   #
    # --------- #
    def load_data(self, **kwargs):
        """  get the data from the quadrants and set it to data. """
        data = self._quadrants_to_ccd(**kwargs)
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
    #  GETTER   #
    # --------- #
    def get_quadrant(self, qid):
        """ get a quadrant (``self.quadrants[qid]``) """
        return self.quadrants[qid]

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

    def get_quadrantdata(self, from_data=False, rebin=None, rebin_stat="mean", **kwargs):
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
            qdata = self._get_subdata(rebin=rebin, **kwargs) # internal method of _Collection_
        else:
            q4 = self.data[:self.qshape[0],self.qshape[1]:]
            q1 = self.data[self.qshape[0]:,self.qshape[1]:]
            q3 = self.data[:self.qshape[0],:self.qshape[1]] 
            q2 = self.data[self.qshape[0]:,:self.qshape[1]]
            qdata = [q1,q2,q3,q4]
            
            if rebin is not None:
                npda = da if self.use_dask else np
                qdata = getattr(npda, rebin_stat)(rebin_arr(qdata, (rebin, rebin),
                                                        use_dask=self.use_dask),
                                              axis=(-2, -1))
        return qdata
        
    def get_data(self, rebin=None, 
                     rebin_quadrant=None,
                     rebin_stat="mean",
                     rebuild=False, persist=False, **kwargs):
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
            data_ = self._quadrants_to_ccd(rebin=rebin_quadrant, **kwargs)
           
        if rebin is not None:
            npda = np if not self.use_dask else da
            data_ = getattr(npda, rebin_stat)(rebin_arr(data_, (rebin, rebin),
                                                        use_dask=self.use_dask),
                                              axis=(-2, -1))
        if self.use_dask and persist:
            data_ = data_.persist()
            
        return data_

    # - internal get
    def _quadrants_to_ccd(self, rebin=None, **kwargs):
        """ combine the ccd data into the quadrants"""
        # numpy or dask.array ?
        npda = da if self.use_dask else np
        
        d = self.get_quadrantdata(rebin=rebin, **kwargs)
        # ccd structure
        # q2 | q1
        # q3 | q4
        ccd_up = npda.concatenate([d[1], d[0]], axis=1)
        ccd_down = npda.concatenate([d[2], d[3]], axis=1)
        ccd = npda.concatenate([ccd_down, ccd_up], axis=0)
        return ccd
    
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
        return cls._QUADRANTCLASS.shape

    @property
    def ccdid(self):
        """ ccd id (from header) """
        return self.get_value("CCDID", None, attr_ok=False) # avoid loop
        
# -------------- #
#                #
#  Focal Plane   #
#                #
# -------------- #
class FocalPlane(Image, _Collection_):
    _CCDCLASS = CCD
    COLLECTION_OF = CCD

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
                           rcids=None, use_dask=True,
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
        ccds = [cls._CCDCLASS.from_filenames(qfiles, qids=[1, 2, 3, 4], as_path=as_path,
                                                 use_dask=use_dask, persist=persist, **kwargs)
                for qfiles in ccdidlist.values]
            
        return cls(ccds, np.asarray(ccdidlist.index, dtype="int"),
                   use_dask=use_dask)

    @classmethod
    def from_single_filename(cls, filename, as_path=True, use_dask=True,
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
        ccds = [cls._CCDCLASS.from_single_filename(re.sub("_c(\d\d)_*", f"_c{i:02d}_", filename),
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
                ccdid = ccd.qid
                
            self.ccds[ccdid] = ccd
            self._use_dask  = ccd.use_dask
        
    def get_ccd(self, ccdid):
        """ get the ccd (self.ccds[ccdid])"""
        return self.ccds[ccdid]

    def get_quadrant(self, rcid):
        """ get the quadrant (get the ccd and then get its quadrant """
        ccdid, qid = self.rcid_to_ccdid_qid(rcid)
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

    def get_data(self, rebin=None, incl_gap=True, persist=False, **kwargs):
        """ get data. """
        # Merge quadrants of the 16 CCDs
        prop = {**dict(rebin=rebin), **kwargs}

        npda = da if self.use_dask else np

        if not incl_gap:
            line_1 = getattr(npda, "concatenate")((self.get_ccd(4).get_data(**prop),
                                                   self.get_ccd(
                                                     3).get_data(**prop),
                                                   self.get_ccd(
                                                     2).get_data(**prop),
                                                   self.get_ccd(1).get_data(**prop)), axis=1)
            line_2 = getattr(npda, "concatenate")((self.get_ccd(8).get_data(**prop),
                                                   self.get_ccd(
                                                     7).get_data(**prop),
                                                   self.get_ccd(
                                                     6).get_data(**prop),
                                                   self.get_ccd(5).get_data(**prop)), axis=1)
            line_3 = getattr(npda, "concatenate")((self.get_ccd(12).get_data(**prop),
                                                   self.get_ccd(
                                                       11).get_data(**prop),
                                                   self.get_ccd(
                                                       10).get_data(**prop),
                                                   self.get_ccd(9).get_data(**prop)), axis=1)
            line_4 = getattr(npda, "concatenate")((self.get_ccd(16).get_data(**prop),
                                                   self.get_ccd(
                                                       15).get_data(**prop),
                                                   self.get_ccd(
                                                       14).get_data(**prop),
                                                   self.get_ccd(13).get_data(**prop)), axis=1)

            mosaic = getattr(npda, "concatenate")(
                (line_1, line_2, line_3, line_4), axis=0)
        else:
            line_1 = getattr(npda, "concatenate")((self.get_ccd(4).get_data(**prop),
                                                   da.ones(self._get_datagap(
                                                       "columns", rebin=rebin))*np.NaN,
                                                   self.get_ccd(
                                                     3).get_data(**prop),
                                                   da.ones(self._get_datagap(
                                                       "columns", rebin=rebin))*np.NaN,
                                                   self.get_ccd(
                                                     2).get_data(**prop),
                                                   da.ones(self._get_datagap(
                                                       "columns", rebin=rebin))*np.NaN,
                                                   self.get_ccd(1).get_data(**prop)), axis=1)
            line_2 = getattr(npda, "concatenate")((self.get_ccd(8).get_data(**prop),
                                                   da.ones(self._get_datagap(
                                                       "columns", rebin=rebin))*np.NaN,
                                                   self.get_ccd(
                                                     7).get_data(**prop),
                                                   da.ones(self._get_datagap(
                                                       "columns", rebin=rebin))*np.NaN,
                                                   self.get_ccd(
                                                     6).get_data(**prop),
                                                   da.ones(self._get_datagap(
                                                       "columns", rebin=rebin))*np.NaN,
                                                   self.get_ccd(5).get_data(**prop)), axis=1)
            line_3 = getattr(npda, "concatenate")((self.get_ccd(12).get_data(**prop),
                                                   da.ones(self._get_datagap(
                                                       "columns", rebin=rebin))*np.NaN,
                                                   self.get_ccd(
                                                       11).get_data(**prop),
                                                   da.ones(self._get_datagap(
                                                       "columns", rebin=rebin))*np.NaN,
                                                   self.get_ccd(
                                                       10).get_data(**prop),
                                                   da.ones(self._get_datagap(
                                                       "columns", rebin=rebin))*np.NaN,
                                                   self.get_ccd(9).get_data(**prop)), axis=1)
            line_4 = getattr(npda, "concatenate")((self.get_ccd(16).get_data(**prop),
                                                   da.ones(self._get_datagap(
                                                       "columns", rebin=rebin))*np.NaN,
                                                   self.get_ccd(
                                                       15).get_data(**prop),
                                                   da.ones(self._get_datagap(
                                                       "columns", rebin=rebin))*np.NaN,
                                                   self.get_ccd(
                                                       14).get_data(**prop),
                                                   da.ones(self._get_datagap(
                                                       "columns", rebin=rebin))*np.NaN,
                                                   self.get_ccd(13).get_data(**prop)), axis=1)
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
                                                  line_4), axis=0)
        if self.use_dask and persist:
            return mosaic.persist()

        return mosaic

    # --------- #
    # CONVERTS  #
    # --------- #
    @staticmethod
    def ccdid_qid_to_rcid(ccdid, qid):
        """ computes the rcid """
        return ccdid_qid_to_rcid(ccdid, qid)

    @staticmethod
    def rcid_to_ccdid_qid(rcid):
        """ computes the rcid """
        return rcid_to_ccdid_qid(rcid)

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
        return cls._CCDCLASS.qshape
