""" Tools to match catalogs """

import os
import pandas
import numpy as np

from astropy import coordinates, units
from ztfquery.io import LOCALSOURCE
CALIBRATOR_PATH = os.path.join(LOCALSOURCE, "calibrator")



__all__ = ["get_field_catalog",
           "get_isolated",
           "match_and_merge"]

# -------------- #
#                #
#  Access        #
#                #
# -------------- # 
def get_field_catalog(which, fieldid, rcid=None, ccdid=None, **kwargs):
    """ get a catalog stored a field catalog
    
    Parameters
    ----------
    which: str
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


    
    # Tests if the field catalog exists.
    if which not in ["ps1"]:
        raise NotImplementedError(f"{which} field catalog is not implemented.")
        
    # directory where the catalogs are stored
    dircat = os.path.join(CALIBRATOR_PATH, which)
    
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
            catfile_ = os.path.join( dircat, f"{rcid_:02d}", 
                                    f"{which}_rc{rcid_:02d}_{fieldid_:06d}.parquet")
            cat = pandas.read_parquet(catfile_, **kwargs)
            cat["fieldid"] = fieldid_
            cat["rcid"] = rcid_
            cats.append(cat)
            
    # get the dataframes
    return pandas.concat(cats)




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
