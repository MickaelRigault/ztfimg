
import numpy as np
import dask
import os
import pandas
from .collection import ScienceImageCollection


class AperturePhotometry( object ):
    
    def __init__(self, filenames, use_dask=True, load=True):
        """ """
        self._use_dask = use_dask
        self.set_filenames(filenames)
        if load:
            self.load_images()

    @classmethod
    def from_datafile(cls, datafile, filekey="filename", use_dask=True):
        """ """
        if type(datafile) is str:
            datafile = pandas.read_hdf(datafile) # reads it.
            
        return cls(datafile[filekey], use_dask=True)

    @classmethod
    def from_filenames(cls, filenames, use_dask=True, **kwargs):
        """ """
        return cls(filenames, use_dask=True, **kwargs)
    
    # =============== #
    #  Methods        #
    # =============== #
    # -------- #
    #  SETTER  #
    # -------- #
    def set_filenames(self, filenames):
        """ """
        self._filenames = np.atleast_1d(filenames).tolist()
        
    # -------- #
    #  LOADER  #
    # -------- #
    def load_images(self, filenames=None):
        """ 
        Parameters
        ----------
        filenames: [list of string] -optional-
            list of path. If given this replaces the current self.filenames if it existed.
            // requested if self.filenames is None //

        """
        if filenames is not None:
            self.set_filenames(filenames)

        self.images = ScienceImageCollection.from_filenames(self.filenames, use_dask=self._use_dask)
        
        
    # -------- #
    # GETTER   #
    # -------- #
    def get_aperture(self, x0s, y0s, radius, bkgann=None, system="xy", data="dataclean",
                         **kwargs):
        """ 
        x0, y0: [2d-array, 2d-array]
           x and y positions where you want your stamp for each images.
           len(x0) and len(y0) must be equal to self.nimages.
           for instance: if you have N images and M positions to stamps for each.
           Note: M does not need to be the same for all images.
           x0 = [[x0_0, x0_1,...,x0_M], 
                 [x1_0, x1_1, ..., x1_M], 
                 ... 
                 [xN_0, xN_1, ..., xN_M]
                 ]
            same for y0

        system: [string] -optional-
            In which system are the input x0, y0:
            - xy (ccd )
            - radec (in deg, sky)
            - uv (focalplane)

        **kwargs goes to each individual image's get_aperture """
        return self.images.get_aperture(x0s, y0s, radius, bkgann=bkgann, system=system, data=data,
                                            **kwargs)
    
    def build_apcatalog(self, radius, calibrators=["gaia","ps1"], extracat=["psfcat"], 
                        isolation=20, xykeys=["x","y"], seplimit=0.5, calkwargs={}, **kwargs):
        """ 
        calkwargs goes to get_catalog()
        kwargs goes to getcat_aperture()
        """
        cats = self.images.get_catalog(calibrators=calibrators, extracat=extracat,
                                       isolation=isolation, seplimit=seplimit, **calkwargs)
        apcat = self.images.map_down("getcat_aperture", cats, radius, xykeys=xykeys, **kwargs)
        if self._use_dask:
            return dask.delayed(pandas.concat)(apcat, keys=self.basenames)
        
        return pandas.concat(apcat, keys=self.basenames)
    
    # =============== #
    #  Properties     #
    # =============== #
    @property
    def filenames(self):
        """ """
        if not hasattr(self,"_filenames"):
            return None
        return self._filenames
    
    @property
    def basenames(self):
        """ """
        return [os.path.basename(f) for f in self.filenames]
    
    @property
    def nfiles(self):
        """ """
        if not hasattr(self,"_filenames"):
            return None
        return len(self.filenames)
    
    