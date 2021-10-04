

import dask
import dask.array as da
import numpy as np
from .science import ScienceQuadrant
from .base import _Quadrant_, _CCD_
class ImageCollection( object ):

    QUADRANT_SHAPE = _Quadrant_.SHAPE
    
    def __init__(self, images, use_dask=True, **kwargs):
        """ """
        self._use_dask = use_dask
        self.set_images(images, **kwargs)
        
    @classmethod
    def from_images(cls, images, use_dask=True, **kwargs):
        """ """
        return cls(images, use_dask=use_dask, **kwargs)
    
    # =============== #
    #  Methods        #
    # =============== #
    # ------- #
    # GETTER  #
    # ------- #
    def get_data_rebustmean(self, chunkreduction=8, 
                            sigma=3, sigma_lower=None, sigma_upper=None, maxiters=5,
                            cenfunc='median', stdfunc='std',
                            **kwargs):
        """ robust mean along the 0th axis (images) 

        **kwargs goes to get_data()

        """
        from astropy.stats import sigma_clip
        datas = self.get_data(**kwargs)
        chunk_merging_axis0 = np.asarray(np.asarray(datas.shape)/(1, 
                                                                  chunkreduction, 
                                                                  chunkreduction), dtype="int")
        datas = datas.rechunk( list(chunk_merging_axis0) )
        datas = datas.map_blocks(sigma_clip, axis=0, 
                                 sigma=sigma, sigma_lower=sigma_lower, sigma_upper=sigma_upper, 
                                 maxiters=maxiters, cenfunc=cenfunc, stdfunc=stdfunc)
        return da.mean(datas, axis=0)
    
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

    # ---------- #
    #  DASK      #
    # ---------- #
    def gather_images(self, client):
        """ gather the self._images (works for delayed only) """
        self._images = client.gather(self._images)
        
    # ---------- #
    #  INTERNAL  #
    # ---------- #
    
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

class QuadrantCollection( ImageCollection ):
    # =============== #
    #  Methods        #
    # =============== #
    # ------- #
    # GETTER  #
    # ------- #
    def get_data(self, **kwargs):
        """ """
        datas = self.call_down("get_data",True, **kwargs)
        if self._use_dask:
            return da.stack([da.from_delayed(f_, shape=self.QUADRANT_SHAPE, dtype="float")
                                     for f_ in datas])
        return datas

class ScienceQuadrantCollection( QuadrantCollection ):


    @classmethod
    def from_filenames(cls, filenames, use_dask=True, imgkwargs={}, **kwargs):
        """ """
        filenames = np.atleast_1d(filenames).tolist()
        if use_dask:
            images = [dask.delayed(ScienceQuadrant.from_filename)(filename, use_dask=False,
                                                                      **imgkwargs)
                     for filename in filenames]
        else:
            images = [ScienceQuadrant.from_filename(filename, use_dask=False, **imgkwargs)
                          for filename in filenames]
            
        return cls(images, use_dask=use_dask, **kwargs)

    # =============== #
    #  Methods        #
    # =============== #            
    # ------- #
    # GETTER  #
    # ------- #
    def get_data(self, which=None, applymask=True,
                     rmbkgd=True, whichbkgd="default",
                     **kwargs):
        """ """
        return super().get_data(applymask=applymask, rmbkgd=rmbkgd,
                               whichbkgd=whichbkgd, which=which, **kwargs)
        

    def get_aperture(self, x0s, y0s, radius, bkgann=None, system="xy",
                         whichdata="dataclean", dataprop={},
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

        whichdata: 
            version of the image data to use for the aperture:
            - data (as stored in science images)
            - clean/dataclean (best version| masked and source background removed)
            (

        **kwargs goes to each individual image's get_aperture """
        dataprop = {**dict(which=whichdata), **dataprop}
        propdown = {**dict( bkgann=bkgann, system=system, dataprop=dataprop),
                    **kwargs}
        return [img.get_aperture(x0_, y0_, radius, **propdown)
                    for img, x0_, y0_ in zip(self.images, x0s, y0s)]
    
    def get_catalog(self, calibrator=["gaia","ps1"], extra=["psfcat"],
                        isolation=20, seplimit=0.5, **kwargs):
        """ """
        propdown = {**dict( calibrator=calibrator, extra=extra,
                            isolation=isolation, seplimit=seplimit),
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
                                    calkwargs={}, system="xy", whichdata="dataclean",
                                    **kwargs):
        """ for each image: calls get_calibrators() and then getcat_aperture()
        """
        cals = self.get_calibrators(which=which, **calkwargs)
        dataprop = {**dict(which=whichdata), **dataprop}
        return self.map_down("getcat_aperture", cals, radius, xykeys=xykeys,
                                 system=system, dataprop=dataprop, **kwargs)
    
    # =============== #
    #  Properties     #
    # =============== #


    
class RawCCDCollection( ImageCollection ):

    QUADRANT_SHAPE = _Quadrant_.SHAPE
    SHAPE = _CCD_.SHAPE

    @classmethod
    def from_filenames(cls, filenames, use_dask=True, imgkwargs={}, **kwargs):
        """ """
        filenames = np.atleast_1d(filenames).tolist()
        if use_dask:
            images = [dask.delayed(RawCCD.from_filename)(filename, use_dask=False,
                                                                **imgkwargs)
                     for filename in filenames]
        else:
            images = [RawCCD.from_filename(filename, use_dask=False, **imgkwargs)
                          for filename in filenames]
            
        return cls(images, use_dask=use_dask, **kwargs)


    
    def get_data(self, corr_overscan=False, corr_nl=False, **kwargs):
        """ """
        return super().get_data(corr_overscan=False, corr_nl=False, **kwargs)


    
