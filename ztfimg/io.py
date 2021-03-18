""" """
import os
import warnings
import pandas
import numpy as np

LOCALSOURCE   = os.getenv('ZTFDATA',"./Data/")
CALIBRATOR_PATH = os.path.join(LOCALSOURCE,"calibrator")


# ========================= #
#                           #
#  PS1 Calibrator Stars     #
#                           #
# ========================= #
class _CatCalibrator_():
    """ """
    def __init__(self, rcid, fieldid, radec=None, load=True):
        """ """
        self._rcid = rcid
        self._fieldid = fieldid
        self.set_centroid(radec)
        if load:
            self.load_data()

    # =============== #
    #  Properties     #
    # =============== #
    # -------- #
    #  LOADER  #
    # -------- #
    def load_data(self, download=True):
        """ """
        filename = self.get_calibrator_file()
        
        if not os.path.isfile(filename):
            if download:
                warnings.warn("Downloading the data.")
                self._data = self.download_data(store=True)
            else:
                raise IOError(f"No file named {filename} and download=False")
        else:
            try:
                self._data = pandas.read_hdf(filename, key=self.get_key())
                
            except KeyError as keyerr:
                warnings.warn(f"KeyError captured: {keyerr}")
                if download:
                    warnings.warn("Downloading the data.")
                    self._data = self.download_data(store=True)
                else:
                    KeyError(f"{keyerr} and download=False")
            
    def download_data(self, store=True, **kwargs):
        """ """
        raise NotImplementedError("You must define the download_data() method")

    # -------- #
    #  SETTER  #
    # -------- #            
    def set_centroid(self, radec):
        """ """
        self._centroid = radec

    # -------- #
    #  GETTER  #
    # -------- #        
    def get_calibrator_file(self):
        """ """
        return os.path.join( os.path.join(CALIBRATOR_PATH, self._DIR, f"{self.BASENAME}_{self.rcid:02d}.hdf5") )

    def get_key(self):
        """ """
        return f"FieldID_{self.fieldid:06d}"

    def get_centroid(self, from_cat=False):
        """ """
        if not from_cat:
            if not hasattr(self,"_centroid") or self._centroid is None:
                from ztfquery import fields
                if self.fieldid not in fields.FIELD_DATAFRAME.index:
                    raise ValueError(f"{self.fieldid} is not a standard ZTF field. Cannot guess the centroid, please run set_centroid().")
                self._centroid = fields.get_rcid_centroid(self.rcid, self.fieldid)
                
            return self._centroid
        else:
            return np.mean(self.data[["ra","dec"]].values, axis=0)

        
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

class GaiaCalibrators( _CatCalibrator_ ):
    _DIR = "gaiadr3"
    BASENAME = "gaiadr3"
    VIZIER_CAT = "I/350/gaiaedr3"


    def download_data(self, store=True, **kwargs):
        """ """
        return self.fetch_vizier_catalog(store=store, **kwargs)
    
    def fetch_vizier_catalog(self, radius= 1, r_unit="deg",
                               column_filters={'Gmag': '10..20'},
                               store=True,
                               **kwargs):
        """ query online gaia-catalog in Vizier (I/350/gaiaedr3, eDR3) using astroquery.
        This function requieres an internet connection.
        
        Parameters
        ----------
        ra, dec: [float]
        center of the Catalog [in degree]

        center: [string] 'ra dec'
        position of the center of the catalog to query.
        (we use the radec of center of the quadrant)
        
        radius: [string] 'value unit'
        radius of the region to query. For instance '1d' means a
        1 degree raduis
        (from the center of the quadrant to the border it is about 0.65 deg)

        extracolumns: [list-of-string] -optional-
        Add extra column from the V/139 catalog that will be added to
        the basic query (default: position, ID, object-type, magnitudes)
        column_filters: [dict] -optional-
        Selection criterium for the queried catalog.
        (we have chosen G badn, it coers from 300 to 1000 nm in wavelength)

        **kwargs goes to Catalog.__init__

        Returns
        -------
        GAIA Catalog (child of Catalog)
        """
        
        from astroquery import vizier
        from astropy import coordinates, units
        columns = ["Source","PS1","SDSSDR13",
                   "RA_ICRS","DE_ICRS",
                   "Gmag", "e_Gmag","GmagCorr",
                   "RPmag","e_RPmag",
                   "BPmag","e_BPmag",
                   "FG","e_FG","FGCorr",
                   "FRP","e_FRP", "FBP","e_FBP",
                   "Plx","e_Plx","PM",
                   "pmRA","e_pmRA","pmDE","e_pmDE"]
            
        # Clean names
        mv_columns = {c:c.lower() for c in columns}
        mv_columns["RA_ICRS"]  = "ra"
        mv_columns["DE_ICRS"]  = "dec"        
        if "PS1" in columns:
            mv_columns["PS1"]  = "ps1_id"
        if "SDSSDR13" in columns:            
            mv_columns["SDSSDR13"]  ="sdssdr13_id"
            
        ra, dec = self.get_centroid()
        #
        # - Vizier Query
        coord = coordinates.SkyCoord(ra=ra, dec=dec, unit=(units.deg,units.deg))
        angle = coordinates.Angle(radius, r_unit)
        v = vizier.Vizier(columns, column_filters=column_filters)
        v.ROW_LIMIT = -1
        # cache is False is necessary, notably when running in a computing center.
        gaiatable = v.query_region(coord, radius=angle, catalog=self.VIZIER_CAT,
                                       cache=False).values()[0]
        gaiatable['colormag'] = gaiatable['BPmag'] - gaiatable['RPmag']
        # - 
        #
        
        dataframe = gaiatable.to_pandas().set_index('Source').rename(mv_columns, axis=1)
        if store:
            filename = self.get_calibrator_file()
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            dataframe.to_hdf(filename, key=self.get_key())
            
        return dataframe
        

class PS1Calibrators( _CatCalibrator_ ):
    _DIR = "ps1"
    BASENAME = "PS1cal_v2"

    def load_data(self):
        """ """
        import h5py
        hf = h5py.File(self.get_calibrator_file(), "r")
        self._data = pandas.DataFrame( hf.get(self.get_key())[()] )
        hf.close()
