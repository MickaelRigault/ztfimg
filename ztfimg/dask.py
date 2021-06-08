

import pandas
import dask.dataframe as dd

from ztfquery import io as zio

from .astrometry import read_radec
    

class DaskScienceFiles( object ):

    _TO_STORED = ["rcid"]

    def __init__(self, filenames, **kwargs):
        """ """
        self.set_filenames(filenames, **kwargs)

    @classmethod
    def sciimgfiles_to_datafiles(cls, sciimgfiles, fileout,per_rcid=True, **kwargs):
        """ """
        this = cls(sciimgfiles, persist=True, **kwargs)
        return this.store_to(filesout, per_rcid=per_rcid, **kwargs)
        
    def set_filenames(self, filenames, npartitions=None, chunksize=10, persist=False, insert_radec=True,
                          **kwargs):
        """ """
        # 
        ddfile = dd.from_pandas(pandas.DataFrame(filenames, columns=["filename"]),
                                    npartitions=npartitions, chunksize=chunksize)
        
        # compute head for structuring
        dhead = ddfile.head(n=1)["filename"].apply(zio.parse_filename)
        dbase = ddfile["filename"].apply(zio.parse_filename, meta=dhead, as_serie=True)
        
        dbase["filename"] = ddfile["filename"]
        self._datafile = dbase

        if insert_radec:
            self.insert_radec()
            
        if persist:
            self.persist()

        self.reset()

    def store_to(self, filename, per_rcid=True, key="data", format="table", allow_compute=True, **kwargs):
        """ """
        if allow_compute:
            self.compute()
            
        hdf_prop = {**dict(key=key, format=format),**kwargs}
        if not per_rcid:
            self.datafile.to_hdf(filename, **hdf_prop)

        for rcid in range(64):
            dd = self.datafile.query(f"rcid == {rcid}")
            dd.to_hdf(filename.replace(".h5",f"_rcid{rcid}.h5"), **hdf_prop)

    def insert_radec(self):
        """ """
        self.datafile[["ra","dec"]] = self.datafile["filename"].apply( read_radec, as_serie=True,
                                                                        meta=pandas.DataFrame( columns=["ra","dec"])
                                                                    )

    def persist(self):
        """ """
        self._datafile = self._datafile.persist()

    def compute(self):
        """ """
        self._datafile = self.datafile.compute()
        
    def reset(self):
        """ """
        for k in self._TO_STORED:
            setattr(self,f"_{k}",None)

    # ------- #
    #  GETTER #
    # ------- #
    def get_rcid(self, recompute=False):
        """ use self.rcid """
        if recompute:
            return self.datafile.groupby("rcid").size().compute()
        
        return self.rcid

    def get_rcid_data(self, rcid):
        """ """
        return self.datafile.query(f"rcid == '{rcid}")
    
    # ================= #
    #   Properties      #
    # ================= #
    @property
    def datafile(self):
        """ """
        return self._datafile

    @property
    def rcid(self):
        """ """
        if not hasattr(self,"_rcid") or self._rcid is None:
            self._rcid = self.get_rcid(recompute=True)
        return self._rcid
