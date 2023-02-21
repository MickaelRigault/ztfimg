""" Tools to match catalogs """

import os
import warnings
import pandas
import numpy as np

from astropy import coordinates, units

__all__ = ["get_field_catalog",
           "download_vizier_catalog",
           "get_isolated",
           "match_and_merge"]

# -------------- #
#                #
#  Access        #
#                #
# -------------- #
def get_field_catalog(name, fieldid, rcid=None, ccdid=None, **kwargs):
    """ get a catalog stored a field catalog
    
    Parameters
    ----------
    name: str
        name of the catalog
        - ps1
        
    fieldid: int or list
        ztf id of the field
        
    rcid: int or list
        id of the quadrant (0->63)
        = requested if ccdid is None = 
        
    ccdid: int or list
        id of the ccd (1->16)
        = ignored if rcid is not None =
        
    Returns
    -------
    DataFrame
        catalog 
    """
    from ztfquery.io import LOCALSOURCE    
    # Tests if the field catalog exists.
    if name not in ["ps1"]:
        raise NotImplementedError(f"{name} field catalog is not implemented.")
        
    # directory where the catalogs are stored
    dircat = os.path.join(LOCALSOURCE, "calibrator", name)
    
    # RCID
    # is rcid given
    if rcid is None:
        if ccdid is None:
            raise ValueError("you must input either rcid or ccdid. Both are None")
            
        # converts ccdid into rcid
        from ztfimg.utils.tools import ccdid_qid_to_rcid
        rcid = ccdid_qid_to_rcid(ccdid, np.arange(1,5)) # array
    else:
        rcid = np.atleast_1d(rcid)  # array
        
    # FIELD        
    fieldid = np.atleast_1d(fieldid) # array
    
    # double for loops because IO structures
    cats = []
    for rcid_ in rcid: # usually a few
        for fieldid_ in fieldid: # usually a few
            catfile_ = os.path.join(dircat, f"{rcid_:02d}", 
                                    f"{name}_rc{rcid_:02d}_{fieldid_:06d}.parquet")
            cat = pandas.read_parquet(catfile_, **kwargs)
            cat["fieldid"] = fieldid_
            cat["rcid"] = rcid_
            cats.append(cat)
            
    # get the dataframes
    return pandas.concat(cats)

def download_vizier_catalog(name,
                            radec, radius=1, r_unit="deg",
                            columns=None, column_filters={},
                            use_tap=False,
                            rakey="RAJ2000", deckey="DEJ2000",
                            **kwargs):
    """ download data from the vizier system

    Parameters
    ----------
    name: string
        name of a vizier calalog.
        known short-names:
        - 'gaia' or 'gaiadr3' -> I/350/gaiaedr3
        - 'ps1' -> II/349/ps1

    radec: [float, float]
        center cone search coordinates (RA, Dec ; in degree).

    radius: float
        radius of the cone search

    r_unit: string
        unit of the cone search radius. (deg, arcsec etc).

    columns: list
        If provided, this will query this specific columns. 
        (see detailed doc in astroquery.vizier.Vizier)
        
    column_filters: dict
        provide filtering of the query;
        (see detailed doc in astroquery.vizier.Vizier)

    use_tap: bool
        = Not yet available =

    rakey: str
        column name of the catalog corresponding to the R.A.

    deckey: str
        column name of the catalog corresponding to the Declination

    **kwargs goes to astroquery.vizier.Vizier
        
    Returns
    -------
    DataFrame
        the catalog
    """

    # name cleaning    
    NAMES_SHORTCUT = {"gaia": "gaiadr3",
                      "panstarrs":"ps1"}
        
    VIZIERCAT = {"gaiadr3": "I/350/gaiaedr3",
                 "ps1": "II/349/ps1"}
                     
    name = NAMES_SHORTCUT.get(name, name) # shortcut
    viziername = VIZIERCAT.get(name, name) # convert to vizier

    # imports
    from astroquery import vizier
    from astropy import coordinates, units

    # metadata for vizier
    coord = coordinates.SkyCoord(*radec, unit=(units.deg,units.deg))
    angle = coordinates.Angle(radius, r_unit)

    prop = dict(column_filters=column_filters, row_limit=-1)
    if columns is not None:
        prop["columns"] = columns
    v = vizier.Vizier( **{**prop, **kwargs} )

    # cache is False is necessary, especially when running from a computing center
    cattable = v.query_region(coord, radius=angle, catalog=viziername, cache=False)

    # Check output
    if cattable is None:
        raise IOError(f"cannot query the region {radec} of {radius}{r_unit} for {viziername}")

    # Worked. 
    cattable = cattable.values()
    if len(cattable) == 0: # but empty
        raise IOError(f"querying the region {radec} of {radius}{r_unit} for {viziername} returns 0 entries")

    # Good to go
    cattable = cattable[0]
    catdata = cattable.to_pandas() # pandas.DataFrame

    # RA, Dec renaming for package interaction purposes
    if rakey in catdata:
        catdata["ra"] = catdata[rakey]
    else:
        warnings.warn(f"no {rakey} column in cat. ra columns set to NaN")
        catdata["ra"] = np.NaN
    
    if deckey in catdata:
        catdata["dec"] = catdata[deckey]
    else:
        warnings.warn(f"no {deckey} column in cat. dec columns set to NaN")
        catdata["dec"] = np.NaN

    # out
    return catdata
    
# -------------- #
#                #
#  Application   #
#                #
# -------------- # 
def get_isolated(catdf, catdf_ref=None, xkey="ra", ykey="dec", keyunit="deg", 
                seplimit=10, sepunits="arcsec"):
    """ """
    
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

def match_and_merge(left, right,
                    radec_key1=["ra","dec"], radec_key2=["ra","dec"], seplimit=0.5,
                    mergehow="inner", suffixes=('_l', '_r'),
                    reset_index=False, **kwargs):
    """     
    Parameters
    ----------
    cat1, cat2: [DataFrames]
        pandas.DataFrame containing, at miminum, the radec_key1/2.
        ra and dec entries (see radec_key1) must by in deg.
        
    radec_key1,radec_key2: [string,string] -optional-
        name of the ra and dec coordinates for catalog 1 and 2, respectively.
        
    seplimit: [float] -optional-
        maximal distance (in arcsec) for the matching.
        
    Returns
    -------
    DataFrame


    Example
    -------
    refpsffile = "../example/fromirsa/ztf_000519_zr_c09_q3_refpsfcat.fits"
    scipsffile = "../example/fromirsa/ztf_20180216349352_000519_zr_c09_o_q3_psfcat.fits"
    pmatch = matching.PSFCatMatch(scipsffile,refpsffile)
    
    pmatch.match() # run the matching, but done automatically if you forgot
    pmatch.get_matched_entries(["ra","dec", "mag","sigmag","snr"]) # any self.scipsf.columns

    """
    
    indexl, indexr = get_coordmatching_indexes(left, right, seplimit=seplimit)
    right.loc[indexr, "index_left"] = indexl
    mcat =  pandas.merge(left, right, left_index=True, right_on="index_left",
                        suffixes=suffixes, how=mergehow, **kwargs).drop(columns="index_left")
    if reset_index:
        mcat = mcat.reset_index(drop=True)
        
    return mcat

def get_coordmatching_indexes(left, right,
                              radec_keyl=["ra","dec"],
                              radec_keyr=["ra","dec"],
                              seplimit=0.5):
    """ get the dataframe indexes corresponding to the matching rows

    Parameters
    ----------
    left, right: [DataFrames]
        pandas.DataFrame containing, at miminum, the radec_key l or r.
        ra and dec entries (see radec_key1) must by in deg.
        
    radec_keyl,radec_keyr: [string,string] -optional-
        name of the ra and dec coordinates for catalog left or right, respectively.
        
    seplimit: [float] -optional-
        maximal distance (in arcsec) for the matching.
        
    Returns
    -------
    list:
        index, index
    """
    # SkyCoord construction
    skyl = coordinates.SkyCoord(left[radec_keyl].values, unit="deg")
    skyr = coordinates.SkyCoord(right[radec_keyr].values, unit="deg")
    # Matching by distance
    idx, d2d, d3d = skyl.match_to_catalog_sky(skyr)
    # remove targets too far
    sep_constraint = d2d < seplimit*units.arcsec
    indexl = left.iloc[sep_constraint].index
    indexr = right.iloc[idx[sep_constraint]].index    
    # get indexes
    return indexl, indexr
