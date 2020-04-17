""" """
import os
import h5py
import pandas

from .  import io
DB_PATH = os.path.join(io.LOCALSOURCE,"calibrator")


# ========================= #
#                           #
#  PS1 Calibrator Stars     #
#                           #
# ========================= #

class PS1CalCatalog():
    
    def __init__(self, rcid, fieldid):
        """ """
        self._rcid = rcid
        self._fieldid = fieldid
        self.load_data()
        
    def get_calibrator_file(self):
        """ """
        return os.path.join(DB_PATH+"/PS1cal_v2_rc%02d.hdf5"%(self.rcid))
    
    def load_data(self):
        """ """
        hf = h5py.File(self.get_calibrator_file(), "r")
        self._data = pandas.DataFrame( hf.get(f"FieldID_%06d"%self.fieldid)[()] )
        hf.close()
        
    # =============== #
    #  Properties     #
    # =============== #
    @property
    def rcid(self):
        """ ZTF CCD ID & QID (RCID 0->63) """
        return self._rcid

    @property
    def fieldid(self):
        """ ZTF Field ID """
        return self._fieldid
    
    @property
    def data(self):
        """ DataFrame of the PS1 Calibrators """
        return self._data
    
    @property
    def ncalibrators(self):
        """ Number of calibrator stars """
        return len(self.data)
