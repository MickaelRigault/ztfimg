
import dask.array as da
import numpy as np
from .base import _Collection_



class ImageCollection( _Collection_ ):
    _COLLECTION_OF = None # Generic
    
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
    def from_filenames(cls, filenames, 
                       as_path=True, 
                       use_dask=False,
                       persist=False, **kwargs):
        """ loads the instance from a list of the quadrant files.

        Parameters
        ----------
        filanames: list of four str
            filenames for the four quadrants 
        
        as_path: bool
            Set this to true if the input file is not the fullpath but you need
            ztfquery.get_file() to look for it for you.

        use_dask: bool
            Should dask be used ? The data will not be loaded but delayed 
            (dask.array)
 
        persist: bool
            = only applied if use_dask=True =
            should we use dask's persist() on data ?

        **kwargs goes to _COLLECTION_OF.from_filename

        Returns
        -------
        class instance
        """
        # _read_filenames is a _Collection_ internal method. handle dask
        images, filenames = cls._read_filenames(filenames,
                                                    as_path=as_path, use_dask=use_dask,
                                                    persist=persist, **kwargs)
        this = cls(images=images)
        this._filenames = filenames
        return this
    
    # ============= #
    #   Methods     #
    # ============= #
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
        # - Test input -#
        
        # images[0] is test to itself, so it works if only 1 image given.
        ref_type = type(images[0]) # type of the first element
        # Are all images of that type ?
        if not np.all([type(image_) == ref_type for image_ in images]):
            raise ValueError("Input images must be instances of the same class")
         
        nuse_dask = np.unique([image_.use_dask for image_ in images])
        if len(nuse_dask)>1:
            raise AttributeError("some images are dasked, some not. This is not supported.")
            
        # - store
        self._collection_of = ref_type
        self._images = np.atleast_1d(images).tolist()
        self._filenames = None # reset
        
    # ------- #
    # GETTER  #
    # ------- #
    def get_data(self, **kwargs):
        """ get a stacked version of the data.
        
        Parameters
        ----------
        **kwargs goes to individual's image.get_data()
        
        Returns
        -------
        3d-array
            [nimages, *shape], np.array or dask.array
        """
        npda = np if not self.use_dask else da
        datas = self._get_subdata(**kwargs)
        return npda.stack(datas)
        
    def call_down(self, what, *args, **kwargs):
        """ call an attribute or a method to each image.
        
        Parameters
        ----------
        what: str
            attribute or method of individual images.
            
        args, kwargs: 
            = ignored if what is an attribute = 
            method options
            
        Returns
        -------
        list 
            list of image's results
            
        See also
        --------
        map_down: map a list to the list of image
        """
        # Expose internal method
        return self._call_down(what, *args, **kwargs)
    
    def map_down(self, method, margs, *args, **kwargs):
        """ map `margs` to each image's `method`
        
        Parameters
        ----------
        method: str
            method name of the individual image
            
        margs: list or array
            parameter to be mapped as individual method's input.
            
        args, kwargs: 
            goes to each image method
            
        Returns
        -------
        list 
            list of image's results
            
        See also
        --------
        call_down: call a method or attribute on each images.
        """
        # Expose internal method
        return self._map_down(method, margs, *args, **kwargs)

    # ============ #
    #  Properties  #
    # ============ #
    @property
    def images(self):
        """ list of images """
        # expose _images
        return self._images
        
    @property
    def use_dask(self):
        """ are images dasked ? """
        # set_images forces all dask to agree
        return self.images[0].use_dask

    @property
    def nimages(self):
        """ number of images """
        return len(self.images)
