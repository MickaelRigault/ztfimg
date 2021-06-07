

import pandas
import dask.dataframe as dd

from ztfquery import io


class DaskScienceFiles( object ):
    """ """
    def __init__(self, filenames, **kwargs):
        """ """
        self.set_filenames(filenames, **kwargs)
        
    def set_filenames(self, filenames, npartitions=None, chunksize=10, persist=False):
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
            
    def persist(self):
        """ """
        self._datafile = self._datafile.persist()

    # ================= #
    #   Properties      #
    # ================= #
    @property
    def datafile(self):
        """ """
        return self._datafile
