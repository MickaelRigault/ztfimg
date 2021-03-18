""" Tools to match catalogs """

import warnings
import numpy as np
import pandas
from astropy.io import fits
from astropy import coordinates, units


"""
usage:

refpsffile = "../example/fromirsa/ztf_000519_zr_c09_q3_refpsfcat.fits"
scipsffile = "../example/fromirsa/ztf_20180216349352_000519_zr_c09_o_q3_psfcat.fits"
pmatch = matching.PSFCatMatch(scipsffile,refpsffile)

pmatch.match() # run the matching, but done automatically if you forgot
pmatch.get_matched_entries(["ra","dec", "mag","sigmag","snr"]) # any self.scipsf.columns

"""

def get_isolated(catdf, catdf_ref=None, xkey="ra", ykey="dec", keyunit="deg", 
                seplimit=10, sepunits="arcsec"):
    """ """
    import pandas
    from astropy import coordinates, units
    if catdf_ref is None:
        catdf_ref = catdf
        
    seplimit =  seplimit* getattr(units,sepunits) 
    
    #
    # - SkyCoord
    sk = coordinates.SkyCoord(catdf[xkey], catdf[ykey], unit=keyunit)
    skref = coordinates.SkyCoord(catdf_ref[xkey], catdf_ref[ykey], unit=keyunit)
    idx2, idx1, d2d, d3d = sk.search_around_sky(skref, seplimit=seplimit)
    unique, counts = np.unique(idx1, return_counts=True)
    iso = pandas.Series(True, index = catdf.index, name="isolated")
    tiso = pandas.Series(counts==1, index = catdf.iloc[unique].index, name="isolated")
    iso.loc[tiso.index] = tiso
    return iso


class CatalogCollection():
    """ """
    def __init__(self):
        """ """
        
    # ============== #
    #  METHODS       #
    # ============== #
    def set_catalog(self, dataframe, label, clearmatches=True):
        """ """
        self.catalogs[label] = dataframe
        if clearmatches:
            for catmatched in self.catmatch.keys():
                if label in catmatched:
                    _ = self.catmatch.pop(catmatched)
            
    def match(self, catin, catref, seplimit=1*units.arcsec):
        """ """
        catmatch = CatMatch.from_dataframe(self.catalogs[catin], self.catalogs[catref])
        catmatch.match(seplimit=seplimit)
        self.catmatch[f'{catin}_{catref}'] = catmatch

    def get_matched_entries(self, key, catin, catref, allowinversed=True):
        """ """
        matchedref = f'{catin}_{catref}'
        if matchedref not in self.catmatch:
            if allowinversed and f'{catref}_{catin}' in self.catmatch:
                return self.get_matched_entries(key,catref, catin, allowinversed=False)

            self.match(catin, catref)
                
        return self.catmatch[matchedref].get_matched_entries(key, catinlabel=catin, catreflabel=catref)

    def get_matched_index(self, index, catin, catref, allowinverted=True):
        """ """
        if f'{catin}_{catref}' in self.catmatch:
            return self.catmatch[f'{catin}_{catref}'].get_matched_catinindex(index)
        if allowinverted and f'{catref}_{catin}' in self.catmatch:
            return self.catmatch[f'{catin}_{catref}'].get_matched_refindex(index)
        raise ValueError("Unmatched catalogs. See self.match()")

    def get_isolated(self, cat, isolation=20*units.arcsec):
        """ get an isolated version of the given catalog.
        
        Isolated means that, inside the same catalog, there is no other sources with `isolation`
        Parameters
        ----------
        cat: [string]
            Name of the catalog 
            
        isolation: [astropy.Quantity] -optional-
            distance around which not other sources should exist within the same catalog.
        
        Returns
        -------
        Pandas.DataFrame (filter `cat`)
        """
        self.match(cat, cat, seplimit=isolation)
        matcharray = self.catmatch[f'{cat}_{cat}'].matchdict['catrefidx']
        unique, counts = np.unique(matcharray, return_counts=True)
        return self.catalogs[cat][counts==1]
    
    # ============== #
    #  Properties    #
    # ============== #
    @property
    def catalogs(self):
        """ Dictionary containing the loaded catalogs """
        if not hasattr(self,"_catalogs") or self._catalogs is None:
            self._catalogs = {}
        return self._catalogs

    @property
    def ncatalogs(self):
        """ number of stored catalogs """
        
    @property
    def labels(self):
        """ """
        return self.catalogs.keys()

    @property
    def catmatch(self):
        """ Matching between catalogs """
        if not hasattr(self,"_catmatch") or self._catmatch is None:
            self._catmatch = {}
        return self._catmatch


class CatMatch( object ):
    """ """
    def __init__(self, catin_fitsfile=None, catref_fitsfile=None):
        """ """
        self.load_file(catin_fitsfile, catref_fitsfile)

    @classmethod
    def from_dataframe(cls, catin, catref):
        """ """
        this = cls()
        this._set_dataframe_("catin", catin)
        this._set_dataframe_("catref", catref)
        return this
    
    # ============== #
    #  METHODS       #
    # ============== #
    # -------- #
    #  I/O     #
    # -------- #
    def load_file(self, catin_fitsfile,  catref_fitsfile):
        """ """
        if catin_fitsfile is not None:
            self._set_dataframe_("catin",pandas.DataFrame(fits.open(catin_fitsfile)[1].data))
            
        if catref_fitsfile is not None:
            self._set_dataframe_("catref",pandas.DataFrame(fits.open(catref_fitsfile)[1].data))
            
    def _set_dataframe_(self, which, dataframe):
        """ """
        setattr(self, f'_{which}', dataframe.astype("f8"))#pandas.DataFrame(fits.open(catin_fitsfile)[1].data)
        setattr(self, f'_{which}sky', None)
        
    # -------- #
    #  Main    #
    # -------- #
    def match(self, seplimit=1*units.arcsec):
        """ run the catalog matching of the catin and catref. 
        This is based on their RA and Dec coordinates with maximum separation of `seplimit`
        """
        catrefidx, catinidx, d2d, d3d = self.catinsky.search_around_sky(self.catrefsky, seplimit=seplimit)
        self._matchdict = {"catrefidx":self.catref.index[catrefidx].values,
                          "catinidx":self.catin.index[catinidx].values,
                          "angsep":d2d}
    # -------- #
    #  GETTER  #
    # -------- #
    def get_matched_entries(self, columns, catinlabel="catin", catreflabel="catref"):
        """ """
        catrefindex, catindex = self.get_matched_catrefindex( self.matchdict["catinidx"])
        bool_used = np.in1d(self.matchdict["catinidx"],catrefindex)
        dict_ = {f"{catreflabel}_index":catrefindex,
                 f"{catinlabel}_index":catindex,
                "angsep_arcsec":self.matchdict["angsep"].to("arcsec")
                }
        for k in np.atleast_1d(columns):
            dict_[f"{catreflabel}_{k}"] = self.catref.loc[catrefindex, k].values
            dict_[f"{catinlabel}_{k}"] = self.catin.loc[catindex, k].values
            
        return pandas.DataFrame(dict_)
    
    def get_matched_catinindex(self, catrefindex):
        """  Provide a catin index (or list of) and get the associated match refpsf index (list of)
        If several entries did match the same refindex, this raises a warning and only the nearest is returned. 
        """
        catrefindex = np.asarray(np.atleast_1d(catrefindex))
        bool_ = np.in1d(catrefindex, self.matchdict["catrefidx"])
        indexmask =  np.argwhere(np.in1d(self.matchdict["catrefidx"], catrefindex[bool_])).flatten()
        if len(indexmask)>len(catrefindex[bool_]):
            warnings.warn("At least one catalogue entry has several index matched. Nearest used")
            return np.asarray([self.matchdict["catinidx"][np.argwhere(self.matchdict["catrefidx"]==i)[
                                  np.argmin(self.matchdict["angsep"][self.matchdict["catrefidx"]==i])]][0]
                for i in catrefindex[bool_]])

        return self.matchdict["catinidx"][indexmask].flatten(), catrefindex[bool_]
    
    def get_matched_catrefindex(self, catinindex):
        """  Provide a catin index (or list of) and get the associated match refpsf index (list of) 
        If several entries did match the same refindex, this raises a warning and only the nearest is returned. 
        """
        catinindex = np.asarray(np.atleast_1d(catinindex))
        bool_ = np.in1d(catinindex, self.matchdict["catinidx"])
        indexmask =  np.argwhere(np.in1d(self.matchdict["catinidx"], catinindex[bool_])).flatten()
        if len(indexmask)>len(catinindex[bool_]):
            warnings.warn("At least one catalogue entry has several index matched. Nearest used")
            return np.asarray([self.matchdict["catrefidx"][np.argwhere(self.matchdict["catinidx"]==i)[
                                  np.argmin(self.matchdict["angsep"][self.matchdict["catinidx"]==i])]][0]
                for i in catinindex[bool_]])

        return self.matchdict["catrefidx"][indexmask].flatten(), catinindex[bool_]
    
    # ============== #
    #  Properties    #
    # ============== #
    # // Ref PSF
    @property
    def catref(self):
        """ Reference input catalog (DataFrame) """
        if not hasattr(self,"_catref"):
            return None
        return self._catref
    
    def has_catref(self):
        """ """
        return self.catref is not None
    
    @property
    def ncatref_entries(self):
        """ number of entry of the catref catalog """
        return len(self.catref) if self.has_catref() else None
    
    @property
    def catrefsky(self):
        """ astropy.coordinates.SkyCoord for the catref RA and Dec """
        if not hasattr(self,"_catrefsky") or self._catrefsky is None:
            if self.has_catref():
                self._catrefsky = coordinates.SkyCoord(self.catref["ra"]*units.deg, self.catref["dec"]*units.deg)
            else:
                self._catrefsky = None
        return self._catrefsky
    
    # // Science PSF
    @property
    def catin(self):
        """ Input catalog (DataFrame) """
        if not hasattr(self,"_catin"):
            return None
        return self._catin
    
    def has_catin(self):
        """ """
        return self.catin is not None
    
    @property
    def ncatin_entries(self):
        """ number of entry of the catin catalog """
        return len(self.catref) if self.has_catref() else None
    
    @property
    def catinsky(self):
        """ astropy.coordinates.SkyCoord for the catin RA and Dec """
        if not hasattr(self,"_catinsky") or self._catinsky is None:
            if self.has_catin():
                self._catinsky = coordinates.SkyCoord(self.catin["ra"]*units.deg, self.catin["dec"]*units.deg)
            else:
                self._catinsky = None
        return self._catinsky
        
                
    # // Matching
    @property
    def matchdict(self):
        """ """
        if not hasattr(self,"_matchdict"):
            self.match()
        return self._matchdict
