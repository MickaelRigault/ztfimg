""" """
import os
import warnings
import pandas
import numpy as np
import time

LOCALSOURCE   = os.getenv('ZTFDATA',"./Data/")
CALIBRATOR_PATH = os.path.join(LOCALSOURCE,"calibrator")



# ========================= #
#                           #
#  PS1 Calibrator Stars     #
#                           #
# ========================= #
class _CatCalibrator_():
    """ """
    def __init__(self, rcid, fieldid, radec=None, load=True, **kwargs):
        """ """
        self._rcid = rcid
        self._fieldid = fieldid
        self.set_centroid(radec)
        if load:
            self.load_data(**kwargs)

    # =============== #
    #  Statics        #
    # =============== #
    @classmethod
    def load_catentry(cls, rcid, field, radec=None, **kwargs):
        """ """
        hdffilename = cls.build_calibrator_filename(rcid)
        hdffile = pandas.HDFStore( hdffilename )
        return cls.load_catentry(hdffile, field, radec=radec, **kwargs)
    
    @classmethod
    def get_catentry(cls, hdf, field, radec=None, store=True **kwargs):
        """ get or download if necessary """
        key = f"/FieldID_{field:06d}"
        if key in list(hdf.keys()):
            return hdf.get(key)

        cat = cls.download_catalog(radec, radius=1, r_unit="deg", **kwargs)
        if store: hdf.put(key.replace("/FieldID","FieldID"), cat)
        return cat
        
    # =============== #
    #  Method         #
    # =============== #
    @classmethod
    def bulk_load_from_files(cls, files, client=None, store=True, force_dl=False, **kwargs):
        """ """
        from astropy.io import fits
        from ztfquery import io
        filedata = io.get_filedataframe(files)
        
        if not filedata["rcid"].nunique() == 1:
            raise ValueError("Only unique rcid list files has been implemented")
        else:
            rcid = filedata["rcid"].unique()[0]
        
        filedata["ra"]  = filedata["filename"].apply(lambda f: fits.getval(f, "RAD") )
        filedata["dec"] = filedata["filename"].apply(lambda f: fits.getval(f, "DECD") )

        return cls.bulk_load_data(rcid,
                                  filedata["field"].values,
                                  radecs=filedata[["ra","dec"]].values,
                                  client=client, store=store, force_dl=force_dl)
        
    @classmethod
    def bulk_load_data(cls, rcid, fieldids, radecs=None, client=None, store=True,
                           force_dl=False, **kwargs):
        """ """
        # fieldids -> fieldid
        fieldid = np.atleast_1d(fieldids)
        requested_keys = np.asarray([f"/FieldID_{f_:06d}" for f_ in fieldid])
        
        # radecs -> radec
        if radecs is not None:
            radec   = np.atleast_2d(radecs)
            if len(fieldid) != len(radec):
                raise ValueError(f"fieldid and radec must have the same size ({len(fieldid)} vs. {len(radec)})")
        else:
            radec = None

        # - Build the object
        this = cls(rcid, fieldid=fieldid, radec=radec, load=False)
        
        # Load/create the hdf file
        hdf = pandas.HDFStore( this.get_calibrator_file() )

        # Are some keys already known ?
        if force_dl:
            warnings.warn(f"force downloading")
            is_known_key = np.asarray(np.zeros(len(requested_keys)), dtype="bool")
        else:
            is_known_key = np.in1d(requested_keys, list(hdf.keys()))
            
        if np.all(is_known_key):
            # All already stored works
            return [hdf.get(f) for f in requested_keys]
        
        #
        # download the missing ones
        radecs_file = radecs[~is_known_key]
        warnings.warn(f"downloading {len(radecs_file)} files")
        future_cats = cls.bulk_download_data(radecs_file, client=client, npartitions=20,
                                                 as_dask="futures")
        dl_cats = client.gather(future_cats)

        # ...and store them if needed
        if store:
            for i, key in enumerate( requested_keys[~is_known_key] ):
                hdf.put(key.replace("/FieldID","FieldID"), dl_cats[i])
                
        hdf.close()
        # Retry
        hdf = pandas.HDFStore( this.get_calibrator_file() )
        return [hdf.get(f) for f in requested_keys]
        
        #
        # Returns
        
            
    @classmethod
    def bulk_download_data(cls, radecs, client=None, npartitions=20, as_dask="delayed"):
        """ """
        from dask import bag
        radecs   = np.atleast_2d(radecs)

        dbag = bag.from_sequence(radecs, npartitions=npartitions)
        catalogs = dbag.map( cls.download_catalog )
        if as_dask is not None:
            if as_dask == "delayed":
                return catalogs
            if as_dask == "futures":
                return client.compute(catalogs)
            raise ValueError(f"as_dask can only be delayed or future, {as_dask} given")
        
        return catalogs.compute()
            
        
        
        
    # -------- #
    #  LOADER  #
    # -------- #        
    def load_data(self, download=True, force_dl=False, store=True, dl_wait=None):
        """ """
        if not force_dl:
            filename = self.get_calibrator_file()
        else:
            download = True
            
        if force_dl or not os.path.isfile(filename):
            if download:
                warnings.warn("Downloading the data.")
                self._data = self.download_data(store=store, wait=dl_wait)
            else:
                raise IOError(f"No file named {filename} and download=False")
        else:
            try:
                self._data = pandas.read_hdf(filename, key=self.get_key())
                
            except KeyError as keyerr:
                warnings.warn(f"KeyError captured: {keyerr}")
                if download:
                    warnings.warn("Downloading the data.")
                    self._data = self.download_data(store=store, wait=dl_wait)
                else:
                    KeyError(f"{keyerr} and download=False")
            
    def download_data(self, store=True, radec=None, **kwargs):
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
        return self.build_calibrator_filename(self.rcid)
    
    @classmethod
    def build_calibrator_filename(cls, rcid):
        """ """
        return os.path.join( os.path.join(CALIBRATOR_PATH, cls._DIR, f"{cls.BASENAME}_rc{self.rcid:02d}.hdf5") )

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

    def download_data(self, store=True, wait=None, radec=None, **kwargs):
        """ you can ask the code to wait (in sec) before running fetch_vizier_catalog """
        if wait is not None: time.sleep(wait)
            
        ra, dec = self.get_centroid() if radec is None else radec
        catdf = self.download_catalog(ra, dec, **kwargs)
        if store:
            filename=self.get_calibrator_file()
            dirname = os.path.dirname(filename)
            if not os.path.isdir(dirname):
                os.makedirs(dirname, exist_ok=True)

            catdf.to_hdf(filename, key=self.get_key())
            
        return catdf

    
    @classmethod
    def download_catalog(cls, radec, radius=1, r_unit="deg",
                            columns=None, column_filters={'Gmag': '10..20'}):
        """ """
        from astroquery import vizier
        from astropy import coordinates, units
        # Check that it's good format
        ra, dec = np.asarray(radec, dtype="float")
        
        if columns is None:
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

        #
        # Downloading
        coord = coordinates.SkyCoord(ra=ra, dec=dec, unit=(units.deg,units.deg))
        angle = coordinates.Angle(radius, r_unit)
        v = vizier.Vizier(columns, column_filters=column_filters)
        v.ROW_LIMIT = -1
        # cache is False is necessary, notably when running in a computing center.
        gaiatable = v.query_region(coord, radius=angle, catalog=cls.VIZIER_CAT, cache=False).values()
        if len(gaiatable)==0:
            raise IOError(f"cannot query the region {ra}, {dec} of {radius}{r_unit} for {catalog}")
        else:
            gaiatable = gaiatable[0]
        
        gaiatable['colormag'] = gaiatable['BPmag'] - gaiatable['RPmag']
        # - 
        #
        return gaiatable.to_pandas().set_index('Source').rename(mv_columns, axis=1)
    
    
class PS1Calibrators( _CatCalibrator_ ):
    _DIR = "ps1"
    BASENAME = "PS1cal_v2"

    def load_data(self):
        """ """
        import h5py
        hf = h5py.File(self.get_calibrator_file(), "r")
        self._data = pandas.DataFrame( hf.get(self.get_key())[()] )
        hf.close()
