

import pandas
import dask.dataframe as dd

from ztfquery import io

class DaskScienceFiles( object ):

    _TO_STORED = ["rcid"]
    
    def __init__(self, filenames, **kwargs):
        """ """
        self.set_filenames(filenames, **kwargs)
    def set_filenames(self, filenames, npartitions=None, chunksize=10, persist=False, **kwargs):
        """ """
        # 
        ddfile = dd.from_pandas(pandas.DataFrame(filenames, columns=["filename"]),
                                    npartitions=npartitions, chunksize=chunksize)
        
        # compute head for structuring
        dhead = ddfile.head(n=1)["filename"].apply(io.parse_filename)
        dbase = ddfile["filename"].apply(io.parse_filename, meta=dhead, as_serie=True)
        
        dbase["filename"] = ddfile["filename"]
        
        self._datafile = dbase
        if persist:
            self.persist()

        self.reset()

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
