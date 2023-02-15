
""" I/O for ztfimg data. """
import os
import warnings
import pandas
import numpy as np
import time

LOCALSOURCE = os.getenv('ZTFDATA',"./Data/")
CALIBRATOR_PATH = os.path.join(LOCALSOURCE,"calibrator")
PACKAGE_PATH = os.path.dirname(os.path.realpath(__file__))
NONLINEARITY_FILE = os.path.join(PACKAGE_PATH, "data/ccd_amp_coeff_v2.txt")


def get_test_image():
    """ returns the path to the test image and its mask. """
    sciimg = os.path.join(PACKAGE_PATH, "data/ztf_20200924431759_000655_zr_c13_o_q3_sciimg.fits")
    maskimg = os.path.join(PACKAGE_PATH, "data/ztf_20200924431759_000655_zr_c13_o_q3_mskimg.fits")
    return sciimg, maskimg

# ========================= #
#                           #
#  IN2P3 CATALOGS           #
#                           #
# ========================= #
def get_ps1_catalog(ra, dec, radius, source="ccin2p3"):
    """ """
    if source in ["in2p3","ccin2p3","cc"]:
        return get_catalog_from_ccin2p3(ra, dec, radius, "ps1")
    
    raise NotImplementedError("Only query to CC-IN2P3 implemented")

def get_catalog_from_ccin2p3(ra, dec, radius, which, enrich=True):
    """  fetch an catalog stored at the ccin2p3.

    Parameters
    ----------
    ra, dec: float
        central point coordinates in decimal degrees or sexagesimal
    
    radius: float
        radius of circle in degrees

    which: str
        Name of the catalog:
        - ps1
        - gaia_dr2
        - sdss
    
    enrich: bool
        IN2P3 catalog have ra,dec coordinates stored in radian
        as coord_ra/dec and flux in nJy
        Shall this add the ra, dec keys coords (in deg) in degree and the magnitude ?

    Returns
    -------
    DataFrame
    """
    from .utils.tools import get_htm_intersect, njy_to_mag
    from astropy.table import Table
    IN2P3_LOCATION = "/sps/lsst/datasets/refcats/htm/v1/"
    IN2P3_CATNAME = {"ps1":"ps1_pv3_3pi_20170110",
                     "gaia_dr2":"gaia_dr2_20190808",
                     "sdss":"sdss-dr9-fink-v5b"}
    
    if which not in IN2P3_CATNAME:
        raise NotImplementedError(f" Only {list(IN2P3_CATNAME.keys())} CC-IN2P3 catalogs implemented ; {which} given")
    
    hmt_id = get_htm_intersect(ra, dec, radius, depth=7)
    dirpath = os.path.join(IN2P3_LOCATION, IN2P3_CATNAME[which])
    cat = pandas.concat([Table.read(os.path.join(dirpath, f"{htm_id_}.fits"), format="fits").to_pandas()
                            for htm_id_ in hmt_id]).reset_index(drop=True)
    if enrich:
        # - ra, dec in degrees
        cat[["ra","dec"]] = cat[["coord_ra","coord_dec"]]*180/np.pi
        # - mags
        fluxcol = [col for col in  cat.columns if col.endswith("_flux")]
        fluxcolerr = [col for col in  cat.columns if col.endswith("_fluxErr")]
        magcol = [col.replace("_flux","_mag") for col in fluxcol]
        magcolerr = [col.replace("_flux","_mag") for col in fluxcolerr]
        cat[magcol], cat[magcolerr] = njy_to_mag(cat[fluxcol].values,cat[fluxcolerr].values)
        
    return cat

# ========================= #
#                           #
#  PS1 Calibrator Stars     #
#                           #
# ========================= #
def parse_input(rcids, fields, radecs):
    """ """
    rcid  = np.atleast_1d(rcids)
    field = np.atleast_1d(fields)
    if len(rcid) == 1:
        rcid = [rcid[0]]*len(field)
    elif len(field) == 1:
        field = [field[0]]*len(rcid)
    elif len(field) != len(rcid):
        raise ValueError(r"fields and rcids must have the same size (or by size 1) ; {len(field)} vs. {len(rcid)}")
    if radecs is not None:
        radec = np.atleast_2d(radecs)
        if len(radec) != len(field):
            raise ValueError(r"radecs and fields must have the same size (or by size 1) ; {len(radec)} vs. {len(field)}")
    else:
        radec = [None]*len(field)
    
    return np.asarray(rcid), np.asarray(field), np.asarray(radec)

def get_nonlinearity_table():
    """ """
    from .utils.tools import ccdid_qid_to_rcid
    nl_table = pandas.read_csv(NONLINEARITY_FILE, comment='#', header=None, sep='\s+', usecols=[0, 1, 2, 3, 4],
                                      names=["ccdid", "ampname", "qid", "a", "b"])
    nl_table["qid"] += 1 # qid in the file is actually AMP_ID that starts at 0, while qid starts at 1.
    nl_table["rcid"] = ccdid_qid_to_rcid(nl_table["ccdid"].astype("int"), nl_table["qid"].astype("int"))
    return nl_table.set_index("rcid").sort_index()

class _CatCalibrator_( object ):

    def __init__(self, rcid, fieldid, radec=None, load=True, **kwargs):
        """ """
        self._rcid = rcid
        self._fieldid = fieldid
        self.set_centroid(radec)
        if load:
            self.load_data(**kwargs)
        
    @classmethod
    def fetch_data(cls, rcid, field, radec=None, squeeze=True, drop_duplicate=True, **kwargs):
        """ Direct access to the data.

        **kwargs goes to load_data

        Parameters
        ----------
        rcid, field: int or list of int
            rcid(s) and field(s) of the data you are looking for.
            for instance: 
            - this returns a multi-index dataframe 
               io.PS1Calibrators.fetch_data([48,49], [751,752])
               io.PS1Calibrators.fetch_data([48,49], 752)
            - and this a DataFrame
               io.PS1Calibrators.fetch_data(48, 751)


        drop_duplicate: [bool]
            // ignored if rcid and field are non iterable //
            drop duplicated entry while keeping the first.

        Returns
        -------
        DataFrame
        """
        from collections.abc import Iterable
        # Looping over rcid
        if isinstance(rcid, Iterable):
            data = pandas.concat([cls.fetch_data(rcid_, field, radec=radec, squeeze=squeeze, **kwargs)
                                      for rcid_ in rcid], keys=rcid).drop_duplicates()
            return data if not drop_duplicate else data.drop_duplicates(keep="first")
        
        # Looping over fieldid
        if isinstance(field, Iterable):
            data = pandas.concat([cls.fetch_data(rcid, field_, radec=radec, squeeze=squeeze, **kwargs)
                                      for field_ in field], keys=field)
            return data if not drop_duplicate else data.drop_duplicates(keep="first")
        
        if kwargs.get("test_input", True):
            rcid, field, radec = parse_input(rcid, field, radec)
        
        data = [cls(rcid_, field_, radec=radec_, load=True).data
                for rcid_, field_, radec_ in zip(rcid, field, radec)]
            
        if squeeze and len(data)==1:
            return data[0]
        
        return data
    
    
    @classmethod
    def bulk_load_data(cls, rcids, fields, radecs=None, npartitions=20, 
                            client=None, as_dask="gather", **kwargs):
        """ load list of dataframes given the rcids, fields, and radecs

        Parameters
        ----------
        rcids, fields: [int or list of]
            rcid (0->63) and field corresponding to your observations.
            rcid and or field could be single values, but if not, 
            they must have the same size.
            e.g. rcids=20 and fields=[1,2,3,4] works
                 rcids=[20,23] and fields=10 works
                 rcids=[20,23] and fields=[1,2,3,4] does not.

        radecs: [2d array or None] -optional-
            list of coordinates. 
            requested if fields are not normal ZTF fields
            and if the catalog are not already stored
            radecs must be a list matching the size of the rcids or fields, 
            whichever is not 1 (except if both are).
            
        npartitions: [int] -optional-
            How many parallel downloading should be call at once ?
            if too high the server might not handleling it.
            
        client: [Dask.Client or None] -optional-
            Dask client used to compute the job.
            - requerested except if as_dask='delayed' or None -
            
        as_dask: [string or None] -optional-
            What part of dask do you want to get
            - 'delayed': the list of delayed functions
            - 'futures': the list of futures functions (client.compute(delayed)
            - 'gather': the actual result data.
            - None: 'delayed' if no client input 'gather' otherwise.
            delayed and futures give back the priority ; 
            gather waits until the result are done.
            
        Returns
        -------
        list 
            of delayed, futures, None ; see as_dask)
        """
        from dask import delayed, bag
        
        rcid, field, radec = parse_input(rcids, fields, radecs)
        nentries = len(field)
        indexes = np.arange(nentries)
        if npartitions is not None:
            indexes = np.array_split(indexes, npartitions)
            
        # Cannot use bag directly for the code needs 3 inputs
        catalogs = [delayed(cls.fetch_data)(rcid[index_], field[index_], radec[index_], squeeze=False, **kwargs)  
                        for index_ in indexes]
        
        if as_dask is not None:
            if as_dask == "delayed":
                return catalogs
            if as_dask == "futures":
                return client.compute(catalogs)
            if as_dask == "gather":
                return client.gather(client.compute(catalogs))
            raise ValueError(f"as_dask can only be delayed or future, {as_dask} given")
        
        cdata = client.gather(client.compute(catalogs))

        import itertools
        return list(itertools.chain(*cdata))
        
        
    @classmethod
    def bulk_download_data(cls, radecs, npartitions=20, 
                            client=None, as_dask=None):
        """ Uses dask to download catalog data associated to the coordinates
        
        Parameters
        ----------
        radecs: [2d array]
            list of coordinates (in deg). eg. [[ra1, dec1], [ra2, dec2]...]
            
        npartitions: [int] -optional-
            How many parallel downloading should be call at once ?
            if too high the server might not handleling it.
            
        client: [Dask.Client or None] -optional-
            Dask client used to compute the job.
            - requerested except if as_dask='delayed' or None -
            
        as_dask: [string or None] -optional-
            What part of dask do you want to get
            - 'delayed': the list of delayed functions
            - 'futures': the list of futures functions (client.compute(delayed)
            - 'gather': the actual result data.
            - None: 'delayed' if no client input 'gather' otherwise.
            delayed and futures give back the priority ; 
            gather waits until the result are done.
            
        Returns
        -------
        list (or delayed, futures, None)
        (see as_dask)
        """
        from dask import bag
        
        radec = np.atleast_2d(radecs)
        
        dbag = bag.from_sequence(radecs, npartitions=npartitions)
        catalogs = dbag.map( cls.download_catalog )
        if as_dask is None:
            as_dask = "gather" if client is not None else "delayed"
            
        if as_dask == "delayed":
            return catalogs
        if as_dask == "futures":
            return client.compute(catalogs)
        if as_dask == "gather":
            return client.gather( client.compute(catalogs) )
            
        raise ValueError(f"as_dask can only be delayed or future, {as_dask} given")
        
    # -------- #
    #  LOADER  #
    # -------- #        
    def load_data(self, download=True, force_dl=False, store=True, dl_wait=None):
        """ Load data from already stored calibrator files and download 
        if needed.
        
        Parameters
        ----------
        download: [bool] -optional-
            If the data is not already stored, shall we download it ?
            
        force_dl: [bool] -optional-
            Should we force the download and then bypass already stored
            data if exist ?
            
        store: [bool] -optional-
            If data have been downloaded, should it store it ?
            
        dl_wait: [float] -optional-
            = in sec =
            shall this pause a before downloading the data ?
            This proves useful in case of massive downloading, 
            to avoid overload the serveur.
            
        Return
        ------
        None
        """
        if not force_dl:
            filename = self.get_calibrator_filename()
        else:
            download = True
            
        if force_dl or not os.path.isfile(filename):
            if download:
                warnings.warn("Downloading the data.")
                self.set_data( self.download_data(store=store, wait=dl_wait) )
            else:
                raise IOError(f"No file named {filename} and download=False")
        else:
            try:
                self.set_data(pandas.read_parquet(filename))
                
            except KeyError as keyerr:
                warnings.warn(f"KeyError captured: {keyerr}")
                if download:
                    warnings.warn("Downloading the data.")
                    self.set_data( self.download_data(store=store, wait=dl_wait) )
                else:
                    KeyError(f"{keyerr} and download=False")
            
    def download_data(self, store=True, radec=None, **kwargs):
        """ """
        raise NotImplementedError("You must define the download_data() method")

    def store(self, data=None):
        """ """
        if data is None:
            data = self.data
            
        filename = self.get_calibrator_filename()
        dirname  = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            os.makedirs(dirname, exist_ok=True)
            
        return data.to_parquet(filename)
        
    # -------- #
    #  SETTER  #
    # -------- #            
    def set_centroid(self, radec):
        """ """
        self._centroid = radec

    def set_data(self, data):
        """ """
        self._data = data
        
    # -------- #
    #  GETTER  #
    # -------- #        
    def get_calibrator_filename(self):
        """ """
        return self.build_calibrator_filename(self.rcid, self.fieldid)
    
    @classmethod
    def build_calibrator_filename(cls, rcid, fieldid, makedirs=False):
        """ """
        rcid = int(rcid)
        fieldid = int(fieldid)
        fileout = os.path.join( os.path.join(CALIBRATOR_PATH, cls.BASENAME, f"{rcid:02d}",
                                            f"{cls.BASENAME}_rc{rcid:02d}_{fieldid:06d}.parquet") )
        if makedirs:
            dirname = os.path.dirname(fileout)
            if not os.path.isdir(dirname):
                os.makedirs(dirname, exist_ok=True)
            
        return fileout
    
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
    #  Internal       #
    # =============== #
    @classmethod
    def _bulk_hdf5_to_parquet_(cls, client=None, as_dask="delayed"):
        """ """
        import dask        
        from glob import glob

        def move_single(rcid, hdffile):
            hdf_ = pandas.HDFStore(hdffile)
            for key in hdf_.keys():
                field = key.split("_")[-1]
                d_ = hdf_.get(key)
                d_.to_parquet(cls.build_calibrator_filename(rcid, field, makedirs=True))

            hdf_.close()
            return None

        dirname = os.path.join(CALIBRATOR_PATH, cls.BASENAME, f"*.hdf5")
        hdffiles = {f.split("_")[-1].split(".")[0].replace("rc",""):f for f in glob(f"{dirname}")}

        dd_moved = [dask.delayed(move_single)(k,v) for k,v in hdffiles.items()]
        
        if as_dask is None:
            as_dask = "gather" if client is not None else "delayed"
            
        if as_dask == "delayed":
            return dd_moved
        if as_dask == "futures":
            return client.compute(dd_moved)
        if as_dask == "gather":
            return client.gather( client.compute(dd_moved) )
        
        raise ValueError(f"as_dask can only be delayed or future, {as_dask} given")
        
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


# =============== #
#                 #
#    Gaia         #
#                 #
# =============== #

class GaiaCalibrators( _CatCalibrator_ ):

    BASENAME = "gaiadr3"
    VIZIER_CAT = "I/350/gaiaedr3"

    def download_data(self, store=True, wait=None, radec=None, **kwargs):
        """ you can ask the code to wait (in sec) before running fetch_vizier_catalog """
        if wait is not None: time.sleep(wait)
        if radec is None:
            radec = self.get_centroid()
            
        catdf = self.download_catalog(radec, **kwargs)
        if store:
            self.store(catdf)
            
        return catdf

    @classmethod
    def download_catalog(cls, radec, radius=1, r_unit="deg",
                            columns=None, column_filters={'Gmag': '<22'}):
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
        gaiatable = v.query_region(coord, radius=angle, catalog=cls.VIZIER_CAT, cache=False)
        if gaiatable is None:
            raise IOError(f"cannot query the region {ra}, {dec} of {radius}{r_unit} for {catalog}")
        gaiatable = gaiatable.values()
        if len(gaiatable) == 0:
            raise IOError(f"querying the region {ra}, {dec} of {radius}{r_unit} for {catalog} returns 0 entries")
        
        gaiatable = gaiatable[0]
        gaiatable['colormag'] = gaiatable['BPmag'] - gaiatable['RPmag']
        # - 
        #
        return gaiatable.to_pandas().set_index("Source").rename(mv_columns, axis=1)


# =============== #
#                 #
#    PS1          #
#                 #
# =============== #

class PS1Calibrators( _CatCalibrator_ ):
    _DIR = "ps1"
    BASENAME = "ps1"

    @classmethod
    def fetch_radecdata(cls, radec=None, **kwargs):
        """ """
        from ztfquery import fields
        fidccid = fields.get_fields_containing_target(*radec, inclccd=True, buffer=0.3) # Make sure it does not fall within a gap.
        datas = []
        for fieldid_ccd in fidccid:
            fieldid,ccd = np.asarray(fieldid_ccd.split("_"), dtype="int")
            rcid = fields.ccdid_qid_to_rcid(ccd, np.asarray([1,2,3,4]))
            datas.append(cls.fetch_data(rcid, [fieldid]))
            
        data = pandas.concat(datas).drop_duplicates()
        return data.reset_index().rename({"level_0":"rcid", "level_1":"fieldid", "level_2":"index"},
                                        axis=1)
    
    def download_data(self, store=True, radec=None, wait=None, **kwargs):
        """ Actually, this down not download but build it from existing files """
        if wait is not None: time.sleep(wait)
        if radec is None:
            radec = self.get_centroid()
            
        catdf = self.fetch_radecdata(radec, **kwargs)
        if store:
            self.store(catdf)
            
        return catdf
