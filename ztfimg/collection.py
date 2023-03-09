

import dask
import dask.array as da
import dask.dataframe as dd
import pandas
import warnings

import numpy as np
from .science import ScienceQuadrant
from .base import Image, Quadrant, CCD, _Collection_
from .raw import RawCCD
from astropy.utils.decorators import classproperty

# -------------- #
#
# -------------- #
def build_headerdf(files, as_path=False, persist=False):
    """ build a dask.dataframe files's header

    Paramters
    ---------
    files: list of str
        list of filenames. Could be path or ztf filename

    as_path: bool
        Set to True if the filename [filename_mask] are path and not just ztf filename, 
        hence bypassing ``files = ztfquery.io.bulk_get_file(files)``
        
    persist: bool
        shall this run .persist() on data ?

    Returns
    -------
    dask.DataFrame
        column as file-number.
    """
    if not as_path:
        files = io.bulk_get_file(files)
        
    headers = [dask.delayed(fits.getheader)(f_) for f_ in files]
    dask_df = _headers_to_headerdf_(headers, persist=persist)
    return dask_df

def _headers_to_headerdf_(headers, persist=False):
    """ """
    headersdf = [dask.delayed(pandas.DataFrame)(h_.items(), 
                                                 columns=("keys", "values")).set_index("keys").sort_index()
               for h_ in headers]
    
    meta = pandas.DataFrame([], columns=("keys", "values")).set_index("keys")
    das = [dd.from_delayed(header, meta=meta) for header in headersdf]
    daskdf = dd.concat(das, axis=1, ignore_unknown_divisions=True)
    daskdf.columns = range(len(headers))
    return daskdf.persist() if persist else daskdf


class ImageCollection( _Collection_ ):
    
    def __init__(self, images, **kwargs):
        """  Image collection handles multi-images

        Parameters
        ----------
        images: list
            list of images. Images could be numpy or dask arrays 
            as well as dask.delayed object. 
            self.use_dask will be set automatically accordingly.

        **kwargs goes to self.set_images()
        
        Returns
        -------
        instance
        """
        self.set_images(images, **kwargs)
        
    @classmethod
    def from_images(cls, images, **kwargs):
        """ load the instance given a list of images
        
        Parameters
        ----------
        images: list
            list of images. Images could be numpy or dask arrays 
            as well as dask.delayed object. 
            self.use_dask will be set automatically accordingly.


        **kwargs goes to __init__() -> set_images()

        Returns
        -------
        instance

        See also
        --------
        from_filenames: loads the instance given a list of files.
        """
        images = np.atleast_1d(images)
        return cls(images, **kwargs)

    @classmethod
    def from_filenames(cls, filenames, as_path=False,
                           use_dask=True, persist=False, **kwargs):
        """ loads the instance from a list of filenames

        Parameters
        ----------
        filenames: list
            list of filenames. Could be pathes or ztf filenames
            (see as_path)

        as_path: bool
            Set to True if the filename [filename_mask] are path and not just ztf filename, 
            hence bypassing ``files = ztfquery.io.get_file(files)``

        use_dask: bool
            Should dask be used ?
            The data will not be loaded but delayed  (dask.array).

        persist: bool
            = only applied if use_dask=True =
            should we use dask's persist() on data ?
            
        **kwargs will be to individual from_filename()

        Returns
        -------
        instance
        
        See also
        --------
        from_images: load the instance given a list of images.
        """
        images, filenames = cls._read_filenames(filenames, use_dask=use_dask, 
                                                as_path=as_path, persist=persist,
                                                **kwargs)
        
        this = cls.from_images(images)
        this._filenames = filenames
        return this
    
    # =============== #
    #  Methods        #
    # =============== #
    # ------- #
    # GETTER  #
    # ------- #
    def get_filenames(self, computed=False):
        """ get the file of filenames for this collection 

        Paramters
        ---------
        computed: bool
            if dask is used, do you want to compute 
            to get the filename (True) or the list of delayed (False)

        Returns
        -------
        list
            list of filename (or delayed)
        """
        filenames = self._call_down("filename")
        if self.use_dask and computed:
            # Tricks to avoid compute and gather
            return dask.delayed(list)(filenames).compute()
        
        return filenames

    def get_singleheader(self, index, as_serie=True, use_dask=False):
        """ call the fits.getheader function from the filenames[index]. 
        
        Parameters
        ----------
        index: [int] 
            index of the filename to pick (self.filenames[index]

        as_serie: [bool] -optional-
            shall the returned header be a pandas/dask Serie ?
            If not, it will be a normal fits.Header()
            Remark that under dask environment, fits.header() may have
            serialization issues.

        Returns
        -------
        Serie (pandas or dask) or header 
            (see as_serie)
        """
        from astropy.io import fits
        if use_dask is None:
            use_dask = self.use_dask
            
        if use_dask:
            header_  = dask.delayed(fits.getheader)(self.filenames[index])
        else:
            header_  = fits.getheader(self.filenames[index])
            
        if not as_serie:
            return header_
        
        if self.use_dask:
            dh = dask.delayed(dict)(header_.items())
            return dd.from_delayed(dask.delayed(pandas.Series)(dh))
        
        return pandas.Series( dict(header_) )
            
    def get_data(self, **kwargs):
        """ get a stack of the collection_of data 
        
        Parameters
        ----------
        **kwargs goes to the collection_of.get_data(**kwargs)

        Returns
        -------
        3d-array
            numpy or dask array.
        """
        return self._get_subdata(**kwargs)

    def get_meandata(self, axis=0,
                         chunkreduction=2,
                         weights=1,  sigma_clip=None, clipping_prop={},
                         **kwargs):
        """ get a the mean 2d-array of the images [nimages, N, M]->[N, M]

        Parameters
        ----------
        chunkreduction: int or None
            rechunk and split of the image.
            If None, no rechunk

        weights: str, float or array
            multiplicative weighting coef for individual images.
            If string, this will be understood as a functuon to apply on data.
            (ie. mean, median, std etc.) | any np.{weights}(data, axis=(1,2) will work.
            otherwise this happens:
            ```
            datas = self.get_data(**kwargs)
            weights = np.asarray(np.atleast_1d(weights))[:, None, None] # broadcast
            datas *=weights
            ```
        
        sigma_clip: float or None
            sigma clipping to be applied to the data (along the stacking axis by default)
            None means, no clipping.

        clipping_prop: dict
            kwargs entering scipy.stats.sigma_clip()

        **kwargs goes to self.get_data()

        Returns
        -------
        2d-array
            mean image (dask or numpy)

        See also
        --------
        get_data: get the stack images [nimages, N, M]
        """
        npda = da if self.use_dask else np
        datas = self.get_data(**kwargs)
        
        # Should you apply weight on the data ?
        if weights is not None and weights != 1:
            if type(weights) == str:
                weights = getattr(npda,weights)(data, axis=(1,2))[:, None, None]
            else:
                weights = np.asarray(np.atleast_1d(weights))[:, None, None] # broadcast
                
            datas *=weights

        # If dasked, should you change the chunkredshuct n?
        if self.use_dask and chunkreduction is not None:
            if axis==0:
                chunk_merging_axis0 = np.asarray(np.asarray(datas.shape)/(1, 
                                                                  chunkreduction, 
                                                                  chunkreduction), dtype="int")
                datas = datas.rechunk( list(chunk_merging_axis0) )
            
            else:
                warnings.warn(f"chunkreduction only implemented for axis=0 (axis={axis} given)")

        # Is sigma clipping applied ?
        if sigma_clip is not None and sigma_clip>0:
            from astropy.stats import sigma_clip as scipy_clipping # sigma_clip is the sigma option
            clipping_prop = {**dict(sigma=sigma_clip, # how many sigma clipping
                                    axis=axis,
                                    sigma_lower=None, sigma_upper=None, maxiters=5,
                                    cenfunc='median', stdfunc='std', 
                                    masked=False),
                             **clipping_prop}
                
            if self.use_dask:
                datas = datas.map_blocks(scipy_clipping, **clipping_prop)
            else:
                datas = scipy_clipping(datas, **clipping_prop)

        # Let's go.
        return npda.mean(datas, axis=axis)

    # ------- #
    # SETTER  #
    # ------- #
    def set_images(self, images):
        """ set the images based on which the instance works
        
        = It is unlikely you need to call that directly = 

        Parameters
        ----------
        images: list
            list of image (or inheritance of) or list of delayed version.

        See also
        --------
        from_images: load the instance given a list of images.
        from_filenames: loads the instance given a list of files.
        """
        # All dasked ?
        self._images = np.atleast_1d(images).tolist()
        self._filenames = None
        
        
    # ---------- #
    #  INTERNAL  #
    # ---------- #
    # =============== #
    #  Properties     #
    # =============== #
    @property
    def headerdf(self):
        """ images header as dask.dataframe. """
        if not hasattr(self, "_headerdf") or self._headerdf is None:
            self._headerdf = _headers_to_headerdf_(self._call_down("header"), persist=True)
            
        return self._headerdf
    
    @property
    def images(self):
        """ list of images defining the instance """
        if not hasattr(self,"_images"):
            return None
        return self._images

    @property
    def filenames(self):
        """ list of image filenames. """
        if not hasattr(self,"_filenames") or self._filenames is None:
            self._filenames = dask.delayed(list)(self._call_down("filename")).compute()
            
        return self._filenames
    
    @property
    def nimages(self):
        """ number of images. """
        if not hasattr(self,"_images"):
            return None
        return len(self.images)

    @property
    def use_dask(self):
        """ is this instance using dask. """
        return self._use_dask

    # --------------- #
    # Class Property  #
    # --------------- #
    @classproperty
    def SHAPE(cls):
        """ shape of object the collection is made of"""
        return cls.COLLECTION_OF.SHAPE
