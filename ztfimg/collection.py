

import dask
import numpy as np
from .science import ScienceQuadrant

class ImageCollection( object ):

    def __init__(self, images, use_dask=True):
        """ """
        self._use_dask = use_dask
        self.set_images(images)
        
    @classmethod
    def from_images(cls, images, use_dask=True, **kwargs):
        """ """
        return cls(images, use_dask=use_dask, **kwargs)
    
    # =============== #
    #  Methods        #
    # =============== #
    # ------- #
    # SETTER  #
    # ------- #
    def set_images(self, images):
        """ """
        self._images = np.atleast_1d(images).tolist()
        
    # -------- #
    # INTERNAL #
    # -------- #
    def map_down(self, what, margs, *args, **kwargs):
        """ """
        return [getattr(img, what)(marg, *args, **kwargs)
                for img, marg in zip(self.images, margs)]
    
    def call_down(self, what, isfunc, *args, **kwargs):
        """ """
        if isfunc:
            return [getattr(img,what)(*args, **kwargs) for img in self.images]
        return [getattr(img,what) for img in self.images]
    
    # =============== #
    #  Properties     #
    # =============== #
    @property
    def images(self):
        """ """
        if not hasattr(self,"_images"):
            return None
        return self._images
    
    @property
    def nimages(self):
        """ """
        if not hasattr(self,"_images"):
            return None
        return len(self.images)
    

        
class ScienceQuadrantCollection( ImageCollection ):

    QUADRANT_SHAPE = ScienceQuadrant.SHAPE
    @classmethod
    def from_filenames(cls, filenames, use_dask=True, imgkwargs={}, **kwargs):
        """ """
        filenames = np.atleast_1d(filenames).tolist()
        if use_dask:
            images = [dask.delayed(ScienceQuadrant.from_filename)(filename, use_dask=False, **imgkwargs)
                     for filename in filenames]
        else:
            images = [ScienceQuadrant.from_filename(filename, use_dask=False, **imgkwargs) for filename in filenames]
            
        return cls(images, use_dask=use_dask, **kwargs)

    # =============== #
    #  Methods        #
    # =============== #            
    # ------- #
    # GETTER  #
    # ------- #
    def get_data(self, applymask=True, maskvalue=np.NaN,
                       rmbkgd=True, whichbkgd="default", **kwargs):
        """ """
        propdown = {**dict(applymask=applymask, maskvalue=maskvalue,
                           rmbkgd=rmbkgd, whichbkgd=whichbkgd), 
                    **kwargs}
        datas = self.call_down("get_data",True, **propdown)
        if self._use_dask:
            return da.stack([da.from_delayed(f_, shape=self.QUADRANT_SHAPE, dtype="float")
                                     for f_ in f.get_data(clean=True)])
        return datas
    
    def get_stamps(self, x0s, y0s, dx, dy=None, data="dataclean", asarray=True,
                  **kwargs):
        """
        
        Parameters
        ----------
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
            same for y0.
            
        dx: [int]
            size of the stamp (same for all images)
            
        dy: [int or None] -optional-
            vertical size of the stamp (if None dy=dx)
            
        data: [string] -optional-
            name of the data you want to get the stamp of
            
        asarray: [bool] -optional-
            should the output be a 2d array (True) or a stamp object (False)
            
        Returns
        -------
        list
        """
        return [img.get_stamps(x0_, y0_, dx, dy=None, data=data, asarray=asarray, **kwargs)
                    for img, x0_, y0_ in zip(self.images, x0s, y0s)]

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
        propdown = {**dict( bkgann=bkgann, system=system, data=data),
                    **kwargs}
        return [img.get_aperture(x0_, y0_, radius, **propdown)
                    for img, x0_, y0_ in zip(self.images, x0s, y0s)]
    
    def get_catalog(self, calibrator=["gaia","ps1"], extra=["psfcat"], isolation=20, seplimit=0.5, **kwargs):
        """ """
        propdown = {**dict( calibrator=calibrator, extra=extra, isolation=isolation, seplimit=seplimit),
                    **kwargs}
        return self.call_down("get_catalog", True, **propdown)
    
    def get_calibrators(self, which=["gaia","ps1"],
                            setxy=True, drop_outside=True, drop_namag=True,
                            pixelbuffer=10, isolation=None, mergehow="inner", **kwargs):
        """ """
        propdown = {**dict( which=which,
                            setxy=setxy, drop_outside=drop_outside, drop_namag=drop_namag,
                            pixelbuffer=pixelbuffer, isolation=isolation, mergehow=mergehow),
                    **kwargs}
        
        return self.call_down("get_calibrators", True, **propdown)
    
    def get_calibrator_aperture(self, radius, which=["gaia","ps1"], xykeys=["x","y"],
                                    calkwargs={}, system="xy", **kwargs):
        """ for each image: calls get_calibrators() and then getcat_aperture()
        """
        cals = self.get_calibrators(which=which, **calkwargs)
        return self.map_down("getcat_aperture", cals, radius, xykeys=xykeys,
                                 system=system, **kwargs)
    
    
    # =============== #
    #  Properties     #
    # =============== #
