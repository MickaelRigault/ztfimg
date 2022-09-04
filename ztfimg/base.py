import numpy as np
import pandas
import warnings
import dask
import dask.array as da
import dask.dataframe as dd

from dask.array.core import Array as DaskArray
from dask.delayed import Delayed, DelayedAttr

from .tools import rebin_arr, parse_vmin_vmax, ccdid_qid_to_rcid, rcid_to_ccdid_qid


__all__ = ["Quadrant", "CCD", "FocalPlane"]

class _Image_( object ):
    SHAPE = None
    # Could be any type (raw, science)

    def __init__(self, use_dask=True):
        """ """
        self._use_dask = use_dask

    @classmethod
    def _read_data(cls, filename, use_dask=True, persist=False):
        """ assuming fits format."""
        from astropy.io.fits import getdata
        if use_dask:
            # - Data
            data = da.from_delayed( dask.delayed( getdata) (filename),
                                   shape=cls.SHAPE, dtype="float")
            if persist:
                data = data.persist()

        else:
            data = getdata(filename)

        return data

    @staticmethod
    def _read_header(filename, use_dask=True, persist=False):
        """ assuming fits format. """
        from astropy.io.fits import getheader        
        if use_dask:
            header = dask.delayed(getheader)(filename)
            if persist:
                header = header.persist()
        else:
            header = getheader(filename)
            
        return header

    @staticmethod
    def _to_fits(fileout, data, header=None,  overwrite=True,
                     **kwargs):
        """ """
        import os        
        from astropy.io import fits
        dirout = os.path.dirname(fileout)
        if not os.path.isdir(dirout):
            os.makedirs(dirout, exist_ok=True)

        fits.writeto(fileout, data, header=header,
                         overwrite=overwrite, **kwargs)
        return fileout

    
    # -------- #
    #  SETTER  #
    # -------- #
    def set_header(self, header):
        """ """
        self._header = header

    def set_data(self, data):
        """ """
        self._data = data

    # -------- #
    #  GETTER  #
    # -------- #
    def get_data(self, rebin=None, rebin_stat="nanmean", data=None, rebuild=False):
        """ 

        Parameters
        ----------
        rebin: [int / None] -optional-
            Shall the data be rebinned by square of size `rebin` ?
            None means no rebinning

        rebin_stat: [string] -optional-
            numpy (dask.array) method used for rebinning the data.

        data: [string] -optional-
            If None, data="data" assumed.
            Internal option to modify the data.
            This could be any attribure value of format (int/float)
            Leave to 'data' if you are not sure.

        Returns
        -------
        2d array
        """
        npda = da if self.use_dask else np
        
        if data is None:
            data = "data"

        # This is the normal case
        if self.has_data() and not rebuild and data=="data":
            data_ = self.data.copy()
        
        else:
            if type(data) == str:
                if data == "data":
                    data_ = self.data.copy()
                elif hasattr(self, data):
                    data_ = npda.ones(self.shape) * getattr(self, data)
                else:
                    raise ValueError(
                        f"value as string can only be 'data' or a known attribute ; {data} given")
                
            elif type(data) in [int, float]:
                data_ = npda.ones(self.shape) * data
            else:
                data_ = data.copy()

        if rebin is not None:
            data_ = getattr(npda, rebin_stat)(
                rebin_arr(data_, (rebin, rebin), use_dask=self._use_dask), axis=(-2, -1))

        return data_

    def get_header(self):
        """ returns the header (self.header), see self.header
        """
        return self.header

    def get_headerkey(self, key, default=None):
        """ """
        if self.header is None:
            raise AttributeError("No header set. see the set_header() method")

        return self.header.get(key, default)

    # -------- #
    # PLOTTER  #
    # -------- #
    def show(self, ax=None, colorbar=True, cax=None, apply=None,
                 imgdata=None,
                 vmin="1", vmax="99", dataprop={}, savefile=None,
                 dpi=150, rebin=None, **kwargs):
        """ """
        import matplotlib.pyplot as mpl

        if ax is None:
            fig = mpl.figure(figsize=[5 + (1.5 if colorbar else 0), 5])
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure

        if imgdata is None:
            imgdata = self.get_data(rebin=rebin, **dataprop)
        if type(imgdata) in [DaskArray, Delayed]:
            imgdata = imgdata.compute()

        if apply is not None:
            imgdata = getattr(np, apply)(imgdata)

        vmin, vmax = parse_vmin_vmax(imgdata, vmin, vmax)
        prop = dict(origin="lower", cmap="cividis", vmin=vmin, vmax=vmax)

        im = ax.imshow(imgdata, **{**prop, **kwargs})
        if colorbar:
            fig.colorbar(im, cax=cax, ax=ax)

        if savefile is not None:
            fig.savefig(savefile, dpi=dpi)

        return ax

    # -------- #
    # PLOTTER  #
    # -------- #
    def _compute_header(self):
        """ """
        if self._use_dask and type(self._header) in [Delayed, DelayedAttr]:
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
        """ """
        return self._use_dask
    
    @property
    def data(self):
        """ """
        if not hasattr(self, "_data"):
            return None
        return self._data

    def has_data(self):
        """ """
        return self.data is not None

    @property
    def header(self, compute=True):
        """ """
        if not hasattr(self, "_header"):
            return None
        # Computes the header only if necessary
        self._compute_header()
        return self._header

    @property
    def shape(self):
        """ """
        return self.SHAPE

    # // header
    @property
    def filename(self):
        """ """
        if not hasattr(self, "_filename"):
            return None
        return self._filename

    @property
    def filtername(self):
        """ """
        return self.get_headerkey("FILTER", "unknown")

    @property
    def exptime(self):
        """ """
        return self.get_headerkey("EXPTIME", np.NaN)

    def obsjd(self):
        """ """
        return self.get_headerkey("OBSJD", None)

# -------------- #
#                #
#   QUANDRANT    #
#                #
# -------------- #


class Quadrant(_Image_):
    SHAPE = 3080, 3072

    def __init__(self, data=None, header=None, use_dask=True):
        """ """
        _ = super().__init__(use_dask=use_dask)
        if data is not None:
            self.set_data(data)

        if header is not None:
            self.set_header(header)

    @classmethod
    def from_filename(cls, filename, use_dask=True, persist=False,
                          dask_header=False, **kwargs):
        """
        Parameters
        ----------
        download: [bool] -optional-
             Downloads the maskfile if necessary.

        **kwargs goes to ztfquery.io.get_file()
        """
        data = cls._read_data(filename, use_dask=use_dask, persist=persist)
        header = cls._read_header(filename, use_dask=dask_header, persist=persist)

        # self
        this = cls(data=data, header=header, use_dask=use_dask)
        this._filename = filename
        return this        

    def to_fits(self, fileout, overwrite=True, **kwargs):
        """ right the quadrant (data and header) into a fits file. """
        return self._to_fits(fileout, data=self.data, header=self.header,
                                overwrite=overwrite, **kwargs)
    
    def get_aperture(self, x0, y0, radius,
                     imgdata=None,
                     bkgann=None, subpix=0,
                     err=None, mask=None,
                     as_dataframe=False,
                     dataprop={},                     
                     **kwargs):
        """ Get the Apeture photometry corrected from the background annulus if any.

        # Based on sep.sum_circle() #

        Parameters
        ----------
        x0, y0, radius: [array]
            Center coordinates and radius (radii) of aperture(s).
            (could be x,y, ra,dec or u,v ; see system)

        bkgann: [None/2D array] -optional-
            Length 2 tuple giving the inner and outer radius of a “background annulus”.
            If supplied, the background is estimated by averaging unmasked pixels in this annulus.

        subpix: [int] -optional-
            Subpixel sampling factor. If 0, exact overlap is calculated. 5 is acceptable.

        data: [string] -optional-
            the aperture will be applied on self.`data`

        unit: [string] -optional-
            unit of the output | counts, flux and mag are accepted.

        clean_flagout: [bool] -optional-
            remove entries that are masked or partially masked
            (remove sum_circle flag!=0)
            = Careful, this does not affects the returned flags, len(flag) remains len(x0)=len(y0) =

        get_flag: [bool]  -optional-
            shall this also return the sum_circle flags

        maskprop, noiseprop:[dict] -optional-
            options entering self.get_mask() and self.get_noise() for `mask` and `err`
            attribute of the sep.sum_circle function.

        as_dataframe: [bool]
            return format.
            If As DataFrame, this will be a dataframe with
            3xn-radius columns (f_0...f_i, f_0_e..f_i_e, f_0_f...f_i_f)
            standing for fluxes, errors, flags.


        Returns
        -------
        2D array (see unit: (counts, dcounts) | (flux, dflux) | (mag, dmag))
           + flag (see get_flag option)
        """
        if imgdata is None:
            imgdata = self.get_data(**dataprop)
            
        return self._get_aperture(imgdata, x0, y0, radius,
                                  bkgann=bkgann, subpix=subpix,
                                  err=err, mask=mask,
                                  as_dataframe=as_dataframe,
                                  **kwargs)

                                 


    @staticmethod
    def _get_aperture(imgdata,
                     x0, y0, radius,
                     bkgann=None, subpix=0,
                     use_dask=None,
                     dataprop={},
                     err=None, mask=None,
                     as_dataframe=False,
                     **kwargs):
        """ """
        from .tools import get_aperture

        if use_dask is None:
            use_dask = type(imgdata) in [DaskArray, Delayed]

        apdata = get_aperture(imgdata,
                              x0, y0, radius=radius,
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

        if type(apdata) == DaskArray:
            return dd.from_dask_array(da.stack(dic.values(), allow_unknown_chunksizes=True).T,
                                      columns=dic.keys())

        return pandas.DataFrame(dic)
                
    
    def getcat_aperture(self, catdf, radius, imgdata=None,
                        xykeys=["x", "y"], join=True,
                        dataprop={}, **kwargs):
        """ measures the aperture (using get_aperture) using a catalog dataframe as input

        # Careful, the indexing is reset (becomes index column)  when joined. #
        Parameters
        ----------
        catdf: [DataFrame]
            dataframe containing, at minimum the x and y centroid positions

        xykeys: [string, string] -optional-
            name of the x and y columns in the input dataframe

        join: [bool] -optional-
            shall the returned dataframe be a new dataframe joined
            to the input one, or simply the aperture dataframe?

        **kwargs goes to get_aperture

        Returns
        -------
        DataFrame
        """
        if imgdata is None:
            imgdata = self.get_data(**dataprop)
            
        return self._getcat_aperture(catdf, imgdata, radius,
                                     xykeys=["x", "y"],
                                     join=join, **kwargs)

    @classmethod
    def _getcat_aperture(cls, catdf, imgdata, radius,
                         xykeys=["x", "y"],
                         join=True, **kwargs):
        """ """
        if join:
            kwargs["as_dataframe"] = True
            
        x, y = catdf[xykeys].values.T
        fdata = cls._get_aperture(imgdata, x, y, radius,
                                  **kwargs)
        if join:
            # the index and drop is because dask.DataFrame do not behave as pandas.DataFrame
            return catdf.reset_index().join(fdata)

        return fdata
    
# -------------- #
#                #
#     CCD        #
#                #
# -------------- #


class CCD(_Image_):
    # Basically a Quadrant collection
    SHAPE = 3080*2, 3072*2
    _QUADRANTCLASS = Quadrant
    _POS_INVERTED = False

    def __init__(self, quadrants=None, qids=None, use_dask=True,
                     data=None, header=None, **kwargs):
        """ Most likely, you want to load the CCD object using the from_* classmethods
        - from_filenames() if you input the filename of the 4 quadrants
        - from_single_filename() if you input only 1 quadrant and what this to find the rest
        - from_filename() if you have a full-CCD image (non IPAC product).
        """
        if data is not None and quadrants is not None:
            raise ValueError("both quadrants and data given. This is confusing. Use only one.")
        
        # - Ok good to go
        _ = super().__init__(use_dask=use_dask)

        if data is not None:
            self.set_data(data)
            
        if header is not None:
            self.set_header(header)
        
        if quadrants is not None:
            if qids is None:
                qids = [None]*len(quadrants)
            elif len(quadrants) != len(qids):
                raise ValueError("quadrants and qids must have the same size.")

            [self.set_quadrant(quad_, qid=qid, **kwargs)
             for quad_, qid in zip(quadrants, qids)]
        
    # =============== #
    #  I/O            #
    # =============== #
    @classmethod
    def from_filename(cls, filename, use_dask=True, persist=False,
                          dask_header=False, **kwargs):
        """ This classmethod loads a CCD object from a full-ccd image (not quadrant).
        = 
        This method cannot be used for IPAC pipeline product (quadrant based).
        If you want to load a IPAC ccd image, use 
        - raw image: RawCCD.from_filename()
        - using a quadrant image: self.from_single_filename()
        - using the 4 quadrant images: self.from_filenames()
        = 
        
        """
        if not use_dask:
            dask_header = False
            
        data = cls._read_data(filename, use_dask=use_dask, persist=persist)
        header = cls._read_header(filename, use_dask=dask_header, persist=persist)
        return cls(data=data, header=header, use_dask=use_dask)
        
    @classmethod
    def from_single_filename(cls, filename, use_dask=True, persist=False, **kwargs):
        """ This classmethod enables to load a full CCD object given a single quadrant filename.
        The other filenames of the 3 other quadrants will be build given the input one.
        """
        import re
        qids = range(1, 5)
        scimg = [cls._QUADRANTCLASS.from_filename(re.sub(r"_o_q[1-5]*", f"_o_q{i}", filename),
                                                  use_dask=use_dask, persist=persist)
                 for i in qids]
        return cls(scimg, qids, use_dask=use_dask)

    @classmethod
    def from_filenames(cls, filenames, qids=[1, 2, 3, 4], use_dask=True,
                       persist=False, qudrantprop={}, **kwargs):
        """ """
        scimg = [cls._QUADRANTCLASS.from_filename(file_, use_dask=use_dask, persist=persist,
                                                    **qudrantprop)
                 for file_ in filenames]
        return cls(scimg, qids, use_dask=use_dask, **kwargs)


    def to_fits(self, fileout, as_quadrants=False, overwrite=True,
                    **kwargs):
        """ """
        if not as_quadrant:
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
        """ dump the current CCD image into 4 different files: one for each quadrants 

        from_data: [bool] -optional-
            option of get_quadrantdata(). If both self.data exists and self.qudrants, 
            should the data be taken from self.data (from_data=True) or from the 
            individual quadrants (from_data=False) using self.quadrants[i].get_data()

        """
        if (nfiles:=len(quadrantfiles)) != 4:
            raise ValueError(f"you need exactly 4 files ; {nfiles} given.")

        quadrant_datas = self.get_quadrantdata(from_data=from_data)
        return self._to_quadrant_fits(quadrantfiles, quadrant_datas,
                                       header=self.header, overwrite=True,
                                       **kwargs)

    @classmethod
    def _to_quadrant_fits(cls, fileouts, datas, header, overwrite=True, **kwargs):
        """ """
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
        """  **kwargs goes to self._quadrants_to_ccd() """
        data = self._quadrants_to_ccd(**kwargs)
        self.set_data(data)

    # --------- #
    #  SETTER   #
    # --------- #
    def set_data(self, data):
        """ """
        if (dshape:=data.shape) != self.SHAPE:
            raise ValueError(f"shape of the input CCD data must be {self.SHAPE} ; {dshape} given")

        # Check dask compatibility
        used_dask = "dask" in str(type(data))
        if used_dask != self.use_dask:
            warnings.warn(f"Input data and self.use_dask are not compatible. Now: use_dask={used_dask}")

        self._data = data

    def set_quadrant(self, quadrant, qid=None):
        """ """
        if qid is None:
            qid = quadrant.qid

        self.quadrants[qid] = quadrant
        self._meta = None

    # --------- #
    #  GETTER   #
    # --------- #
    def get_quadrant(self, qid):
        """ """
        return self.quadrants[qid]

    def get_quadrantheader(self):
        """ returns a DataFrame of the header quadrants """
        if not self.has_quadrants():
            return None
        
        qid_range = [1, 2, 3, 4]
        hs = [quadrant.get_header() for i in qid_range
                  if (quadrant:=self.get_quadrant(i)) is not None]
        df = pandas.concat(hs, axis=1)
        df.columns = qid_range
        return df

    def get_quadrantdata(self, rebin=False, from_data=False, npstat="mean", **kwargs):
        """ 
        npstat: string
            function used to rebin (np.mean, np.median etc)
        """
        if not self.has_quadrants() and not self.has_data():
            warnings.warn("no quadrant and no data. None returned")
            return None

        if self.has_quadrants() and not from_data:
            qdata = [self.get_quadrant(i).get_data(rebin=rebin, **kwargs)
                    for i in [1, 2, 3, 4]]
        else:
            q4, q1 = self.data[:self.qshape[0],self.qshape[1]:], data[self.qshape[0]:,self.qshape[1]:]
            q3, q2 = self.data[:self.qshape[0],:self.qshape[1]], data[self.qshape[0]:,:self.qshape[1]]
            qdata = [q1,q2,q3,q4]
            if rebin is not None:
                npda = da if self.use_dask else np
                qdata = getattr(npda, npstat)(rebin_arr(qdata, (rebin, rebin),
                                                        use_dask=self.use_dask),
                                              axis=(-2, -1))
        return qdata

        
        
    def get_data(self, rebin=None, npstat="mean",
                     rebin_quadrant=None, persist=False,
                     rebuild=False, **kwargs):
        """ ccd data

        rebin, rebin_quadrant: [None, int]
            rebinning (based on restride) the data.
            rebin affect the whole ccd, while rebin_quadrant affect the individual quadrants.
            rebin is applied after rebin_quadrant..
        """
        # the normal case
        if self.has_data() and not rebuild and not rebin_quadrant:
            data_ = self.data.copy()
        else:
            data_ = self._quadrants_to_ccd(rebin=rebin_quadrant)
           
        if rebin is not None:
            data_ = getattr(npda, npstat)(rebin_arr(data_, (rebin, rebin),
                                                        use_dask=self.use_dask),
                                              axis=(-2, -1))
        if self.use_dask and persist:
            data_ = data_.persist()
            
        return data_

    # - internal get
    def _quadrants_to_ccd(self, rebin=False):
        """ combine the ccd data into the quadrants"""
        # numpy or dask.array ?
        npda = da if self.use_dask else np
        
        d = self.get_quandrantdata(rebin=rebin)

        if not self._POS_INVERTED:
            ccd_up = npda.concatenate([d[1], d[0]], axis=1)
            ccd_down = npda.concatenate([d[2], d[3]], axis=1)
        else:
            ccd_up = npda.concatenate([d[3], d[2]], axis=1)
            ccd_down = npda.concatenate([d[0], d[1]], axis=1)

        ccd = npda.concatenate([ccd_down, ccd_up], axis=0)
        return ccd
    
    # ----------- #
    #   PLOTTER   #
    # ----------- #
    def show(self, ax=None, vmin="1", vmax="99", colorbar=False, cax=None,
                 imgdata=None,
             rebin=None, dataprop={}, savefile=None,
             dpi=150, **kwargs):
        """ """
        import matplotlib.pyplot as mpl
        if ax is None:
            fig = mpl.figure(figsize=[6, 6])
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure

        if imgdata is None:
            imgdata = self.get_data(rebin=rebin, **dataprop)
            
        if type(imgdata) in [DaskArray, Delayed]:
            imgdata = imgdata.compute()

        vmin, vmax = parse_vmin_vmax(imgdata, vmin, vmax)

        prop = {**dict(origin="lower", cmap="cividis", vmin=vmin, vmax=vmax),
                **kwargs}

        im = ax.imshow(imgdata, **prop)

        if colorbar:
            fig.colorbar(im, cax=cax, ax=ax)

        if savefile is not None:
            fig.savefig(savefile, dpi=dpi)
        return ax

    # =============== #
    #  Properties     #
    # =============== #
    @property
    def data(self):
        """ """
        if not hasattr(self, "_data"):
            if not self.has_quadrants("all"):
                return None

            self.load_data()

        return self._data

    @property
    def quadrants(self):
        """ """
        if not hasattr(self, "_quadrants"):
            self._quadrants = {k: None for k in [1, 2, 3, 4]}
        return self._quadrants

    def has_quadrants(self, logic="all"):
        """ """
        is_none = [v is not None for v in self.quadrants.values()]
        return getattr(np, logic)(is_none)

    @property
    def qshape(self):
        """ shape of an individual ccd quadrant """
        return self._QUADRANTCLASS.SHAPE

# -------------- #
#                #
#  Focal Plane   #
#                #
# -------------- #
class FocalPlane(_Image_):
    _CCDCLASS = CCD

    # Basically a CCD collection
    def __init__(self, ccds=None, ccdids=None, use_dask=True, **kwargs):
        """ """
        _ = super().__init__(use_dask=use_dask)

        if ccds is not None:
            if ccdids is None:
                ccdids = [None]*len(ccds)
            elif len(ccds) != len(ccdids):
                raise ValueError("ccds and ccdids must have the same size.")

            [self.set_ccd(ccd_, ccdid=ccdid_, **kwargs)
             for ccd_, ccdid_ in zip(ccds, ccdids)]

    # =============== #
    #  I/O            #
    # =============== #
    @classmethod
    def from_filenames(cls, filenames, rcids=None, use_dask=True,
                       persist=False, **kwargs):
        """
        rcids: [list or None]
            if None: rcids = np.arange(0,64)

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

        ccds = [cls._CCDCLASS.from_filenames(qfiles, qids=[1, 2, 3, 4],
                                                 use_dask=use_dask, persist=persist, **kwargs)
                for qfiles in ccdidlist.values]
        return cls(ccds, np.asarray(ccdidlist.index, dtype="int"),
                   use_dask=use_dask)

    @classmethod
    def from_single_filename(cls, filename, use_dask=True, persist=False, **kwargs):
        """ """
        import re
        ccdids = range(1, 17)
        ccds = [cls._CCDCLASS.from_single_filename(re.sub("_c(\d\d)_*", f"_c{i:02d}_", filename),
                                                   use_dask=use_dask, persist=persist, **kwargs)
                for i in ccdids]
        return cls(ccds, ccdids, use_dask=use_dask)

    # =============== #
    #   Methods       #
    # =============== #
    def set_ccd(self, rawccd, ccdid=None):
        """ """
        if ccdid is None:
            ccdid = rawccd.qid

        self.ccds[ccdid] = rawccd
        self._meta = None

    def get_ccd(self, ccdid):
        """ """
        return self.ccds[ccdid]

    def get_quadrant(self, rcid):
        """ """
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
    def get_datagap(which, rebin=None, fillna=np.NaN):
        """
        horizontal (or row) = between rows
        """
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
        """  """
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
                                                   da.ones(self.get_datagap(
                                                       "columns", rebin=rebin))*np.NaN,
                                                   self.get_ccd(
                                                     3).get_data(**prop),
                                                   da.ones(self.get_datagap(
                                                       "columns", rebin=rebin))*np.NaN,
                                                   self.get_ccd(
                                                     2).get_data(**prop),
                                                   da.ones(self.get_datagap(
                                                       "columns", rebin=rebin))*np.NaN,
                                                   self.get_ccd(1).get_data(**prop)), axis=1)
            line_2 = getattr(npda, "concatenate")((self.get_ccd(8).get_data(**prop),
                                                   da.ones(self.get_datagap(
                                                       "columns", rebin=rebin))*np.NaN,
                                                   self.get_ccd(
                                                     7).get_data(**prop),
                                                   da.ones(self.get_datagap(
                                                       "columns", rebin=rebin))*np.NaN,
                                                   self.get_ccd(
                                                     6).get_data(**prop),
                                                   da.ones(self.get_datagap(
                                                       "columns", rebin=rebin))*np.NaN,
                                                   self.get_ccd(5).get_data(**prop)), axis=1)
            line_3 = getattr(npda, "concatenate")((self.get_ccd(12).get_data(**prop),
                                                   da.ones(self.get_datagap(
                                                       "columns", rebin=rebin))*np.NaN,
                                                   self.get_ccd(
                                                       11).get_data(**prop),
                                                   da.ones(self.get_datagap(
                                                       "columns", rebin=rebin))*np.NaN,
                                                   self.get_ccd(
                                                       10).get_data(**prop),
                                                   da.ones(self.get_datagap(
                                                       "columns", rebin=rebin))*np.NaN,
                                                   self.get_ccd(9).get_data(**prop)), axis=1)
            line_4 = getattr(npda, "concatenate")((self.get_ccd(16).get_data(**prop),
                                                   da.ones(self.get_datagap(
                                                       "columns", rebin=rebin))*np.NaN,
                                                   self.get_ccd(
                                                       15).get_data(**prop),
                                                   da.ones(self.get_datagap(
                                                       "columns", rebin=rebin))*np.NaN,
                                                   self.get_ccd(
                                                       14).get_data(**prop),
                                                   da.ones(self.get_datagap(
                                                       "columns", rebin=rebin))*np.NaN,
                                                   self.get_ccd(13).get_data(**prop)), axis=1)
            size_shape = self.get_datagap("rows", rebin=rebin)[0]

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

    def show(self, ax=None, vmin="1", vmax="99", colorbar=False, cax=None,
                 imgdata=None,
                 rebin=None, incl_gap=True, dataprop={},
                 savefile=None, dpi=150,
                 **kwargs):
        """ """
        import matplotlib.pyplot as mpl
        if ax is None:
            fig = mpl.figure(figsize=[6, 6])
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure

        if imgdata is None:
            imgdata = self.get_data(rebin=rebin, incl_gap=incl_gap, **dataprop)
            
        if type(imgdata) in [DaskArray, Delayed]:
            imgdata = imgdata.compute()

        vmin, vmax = parse_vmin_vmax(imgdata, vmin, vmax)

        prop = {**dict(origin="lower", cmap="cividis", vmin=vmin, vmax=vmax),
                **kwargs}

        im = ax.imshow(imgdata, **prop)

        if colorbar:
            fig.colorbar(im, cax=cax, ax=ax)

        if savefile is not None:
            fig.savefig(savefile, dpi=dpi)

        return ax

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
        """ """
        if not hasattr(self, "_ccds"):
            self._ccds = {k: None for k in np.arange(1, 17)}

        return self._ccds

    def has_ccds(self, logic="all"):
        """ """
        is_none = [v is not None for v in self.ccds.values()]
        return getattr(np, logic)(is_none)

    @property
    def shape_full(self):
        """ shape with gap"""
        print("gap missing")
        return self.ccdshape*4

    @property
    def shape(self):
        """ shape without gap"""
        return self.ccdshape*4

    @property
    def ccdshape(self):
        """ """
        return self.qshape*2

    @property
    def qshape(self):
        """ """
        return np.asarray([3080, 3072])
