

import dask
import dask.array as da
import dask.dataframe as dd
import pandas
import warnings

import numpy as np
from .science import ScienceQuadrant
from .base import _Image_, Quadrant, CCD
from .raw import RawCCD
from astropy.utils.decorators import classproperty

# -------------- #
#
# -------------- #
def build_headerdf(files, persist=False):
    """ """
    delayed_files = io.bulk_get_file(files)   
    headers = [dask.delayed(fits.getheader)(f_) for f_ in delayed_files]
    return _headers_to_headerdf_(headers, persist=persist)

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



class _ImageCollection_( object ):
    COLLECTION_OF = _Image_
    QUADRANT_SHAPE = Quadrant.SHAPE
    
    def __init__(self, images, use_dask=True, **kwargs):
        """ """
        self._use_dask = use_dask
        self.set_images(images, **kwargs)
        
    @classmethod
    def from_images(cls, images, use_dask=True, **kwargs):
        """ """
        return cls(images, use_dask=use_dask, **kwargs)

    @classmethod
    def from_filenames(cls, filenames, use_dask=True, imgkwargs={},
                           as_path=False, persist=False, **kwargs):
        """ """
        filenames = np.atleast_1d(filenames).tolist()
        imgkwargs["as_path"] = as_path # pass this to .from_filename
        
        if use_dask:
            images = [dask.delayed(cls.COLLECTION_OF.from_filename)(filename, use_dask=False,
                                                                      **imgkwargs)
                     for filename in filenames]
            if persist:
                images = [i.persist() for i in images]
                
        else:
            images = [cls.COLLECTION_OF.from_filename(filename, use_dask=False, **imgkwargs)
                          for filename in filenames]
            
        this= cls.from_images(images, use_dask=use_dask, **kwargs)
        this._filenames = filenames
        return this
    
    # =============== #
    #  Methods        #
    # =============== #
    # ------- #
    # GETTER  #
    # ------- #
    def get_filenames(self, computed=False):
        """ """
        filenames = self.call_down("filename", False)
        if self.use_dask and computed:
            # Tricks to avoid compute and gather
            return dask.delayed(list)(filenames).compute()
        
        return filenames

    def get_singleheader(self, index, as_serie=True, use_dask=False):
        """ call the fits.getheader function from the
        filenames[index]. 
        
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
            
    def get_headerdf(self, persist=True, rebuild=False):
        """ """
        if rebuild or not hasattr(self, "_headerdf"):
            return _headers_to_headerdf_(self.call_down("header", False), persist=True)
        
        return self.headerdf

    def get_data(self, **kwargs):
        """ """
        datas = self.call_down("get_data",True, **kwargs)
        if self.use_dask:
            return da.stack([da.from_delayed(f_, shape=self.SHAPE, dtype="float")
                                     for f_ in datas])
        return np.stack(datas)

    def get_data_mean(self, chunkreduction=2,
                          weights=1,
                          clipping=False,
                          sigma=3, sigma_lower=None, sigma_upper=None, maxiters=5,
                          cenfunc='median', stdfunc='std', axis=0, **kwargs):
        """ 
        weights: [string, float or array]
            If you want to apply a multiplicative weighting coef to the individual
            images before getting there mean.
            If string, this will be understood as a functuon to apply on data.
            (ie. mean, median, std etc.) | any np.{weights}(data, axis=(1,2) will work.

            if not None, this happens:
            datas = self.get_data(**kwargs)
            weights = np.asarray(np.atleast_1d(weights))[:,None,None] # broadcast
            datas *=weights

            


        chunkreduction: [int or None]
            rechunk and split of the image. 2 or 4 is best.
            If None, no rechunk
        
        clipping: [bool] -optional-
            shall sigma clipping along the axis=0 be applied ?

            
                          
        """
        npda = da if self.use_dask else np
        
        datas = self.get_data(**kwargs)
        if weights is not None:
            if type(weights) == str:
                weights = getattr(npda,weights)(data, axis=(1,2))[:,None,None]
            else:
                weights = np.asarray(np.atleast_1d(weights))[:,None,None] # broadcast
                
            datas *=weights

        if self.use_dask:
            if axis==0 and chunkreduction is not None:
                chunk_merging_axis0 = np.asarray(np.asarray(datas.shape)/(1, 
                                                                  chunkreduction, 
                                                                  chunkreduction), dtype="int")
                datas = datas.rechunk( list(chunk_merging_axis0) )
            
            elif chunkreduction is not None:
                warnings.warn(f"chunkreduction only implemented for axis=0 (axis={axis} given)")
            
        if clipping:
            from astropy.stats import sigma_clip
            clipping_prop = dict(axis=axis, 
                                 sigma=sigma, sigma_lower=sigma_lower, sigma_upper=sigma_upper, 
                                 maxiters=maxiters, cenfunc=cenfunc,
                                 stdfunc=stdfunc, masked=False)
            if self.use_dask:
                datas = datas.map_blocks(sigma_clip, **clipping_prop)
            else:
                datas = sigma_clip(datas, **clipping_prop)
            
        return npda.mean(datas, axis=axis)

    # ------- #
    # SETTER  #
    # ------- #
    def set_images(self, images):
        """ """
        self._images = np.atleast_1d(images).tolist()
        self._filenames = None
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
    def headerdf(self):
        """ """
        if not hasattr(self, "_headerdf") or self._headerdf is None:
            self._headerdf = self.get_headerdf(rebuild=True)
            
        return self._headerdf
    @property
    def images(self):
        """ """
        if not hasattr(self,"_images"):
            return None
        return self._images

    @property
    def filenames(self):
        """ """
        if not hasattr(self,"_filenames") or self._filenames is None:
            self._filenames = dask.delayed(list)(self.call_down("filename", False)).compute()
            
        return self._filenames
    
    @property
    def nimages(self):
        """ """
        if not hasattr(self,"_images"):
            return None
        return len(self.images)

    @property
    def use_dask(self):
        """ """
        return self._use_dask

    # --------------- #
    # Class Property  #
    # --------------- #
    @classproperty
    def SHAPE(cls):
        """ shape of object the collection is made of"""
        return cls.COLLECTION_OF.SHAPE
    
# Collection of Quadrant    
class QuadrantCollection( _ImageCollection_ ):
    COLLECTION_OF = Quadrant

# Collection of CCD    
class CCDCollection( _ImageCollection_ ):
    COLLECTION_OF = CCD

# Collection of Science Quadrant
class ScienceQuadrantCollection( QuadrantCollection ):
    COLLECTION_OF = ScienceQuadrant

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
    
    def get_catalog(self, calibrators="gaia", extra=["ps1","psfcat"],
                        isolation=20, seplimit=0.5, **kwargs):
        """ """
        propdown = {**dict( calibrators=calibrators, extra=extra,
                            isolation=isolation, seplimit=seplimit),
                    **kwargs}
        return self.call_down("get_catalog", True, **propdown)
    
    def get_calibrators(self, which="gaia",
                            setxy=True, drop_outside=True, drop_namag=True,
                            pixelbuffer=10, isolation=None, mergehow="inner", **kwargs):
        """ """
        propdown = {**dict( which=which,
                            setxy=setxy, drop_outside=drop_outside, drop_namag=drop_namag,
                            pixelbuffer=pixelbuffer, isolation=isolation, mergehow=mergehow),
                    **kwargs}
        
        return self.call_down("get_calibrators", True, **propdown)
    
    def get_calibrator_aperture(self, radius, which="gaia", xykeys=["x","y"],
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



# ======================= #
#                         #
#    RAW                  #
#                         #
# ======================= #
class RawCCDCollection( CCDCollection ):
    COLLECTION_OF = RawCCD
    
    @staticmethod
    def bulk_getfile_daterange(start, end, ccdid, imgtype=None, filtercode=None,
                                   dateformat=None, persist=False):
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
        return io.bulk_get_file(zquery.get_data(exist=False),
                            as_dask="persist" if persist else "delayed")
    
    # ------------ #
    #  INITIALISE  #
    # ------------ #
    @classmethod
    def from_date(cls, date, ccdid, dateformat=None, **kwargs):
        """ """
        start = Time(date, format=dateformat).jd
        return cls.from_daterange(start, start+1, dateformat="jd", ccdid=ccdid, **kwargs)
        
    @classmethod
    def from_daterange(cls, start, end, ccdid, filtercode=None, dateformat=None,
                           persist=False, queryprop={}, **kwargs):
        """ """
        files = cls.bulk_getfile_daterange(start, end, ccdid, filtercode=filtercode, dateformat=dateformat, 
                                           persist=persist, **queryprop)
        return cls.from_filenames(files, use_dask=True, persist=persist, **kwargs)


class RawFlatCCDCollection( RawCCDCollection ):
    
    @classmethod
    def bulk_getfile_daterange(cls, start, end, ccdid, filtercode=["zr","zg","zi"],
                                   dateformat=None, persist=False,
                                   **kwargs):
        """ """
        return super().bulk_getfile_daterange(start, end, ccdid,
                                             imgtype="flat",  # forcing this.
                                              filtercode=filtercode,
                                              dateformat=dateformat, persist=persist, 
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
