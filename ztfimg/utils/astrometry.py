""" WCS Class handler """

import numpy as np
from astropy.wcs import WCS as astropyWCS
import pandas
from . import tools

def read_radec(filename, ext=0, as_serie=False):
    """ """
    ra, dec = WCS.from_filename(filename).get_center("radec")

    if as_serie:
        return pandas.Series([ra,dec], index=["ra","dec"])
    
    return ra,dec

class WCSHolder( object ):

    # =============== #
    #  Methods        #
    # =============== #
    # --------- #
    #  LOADER   #
    # --------- #        
    def load_wcs(self, header, pointingkey=["RA","DEC"]):
        """ """
        # - wcs
        self.set_wcs( astropyWCS(header), pointingkey=pointingkey)
        
        # - pointing        
        pra, pdec = header.get(pointingkey[0], None), header.get(pointingkey[1], None)
        if pra is None or pdec is None:
            return None
            
        from astropy import coordinates, units
        sc = coordinates.SkyCoord(pra, pdec, unit=(units.hourangle, units.deg))

        self.set_pointing(sc.ra.value, sc.dec.value)
        
    # --------- #
    #  SETTER   #
    # --------- #
    def set_wcs(self, wcs, pointingkey=["RA","DEC"]):
        """ """
        self._wcs = wcs

    def set_pointing(self, ra, dec):
        """ """
        self._pointing = ra, dec

    # --------- #
    #  GETTER   #
    # --------- #
    # --------- #
    #  Convert  #
    # --------- #

    # xy
    def xy_to_radec(self, x, y, reorder=True):
        """ get sky ra, dec [in deg] coordinates given the (x,y) quadrant positions  """
        if reorder and hasattr(self, "shape"):
            x = self.shape[1] -x -1 # starts at 0
            y = self.shape[0] -y -1 # starts at 0
                        
        return self.wcs.all_pix2world(np.asarray([np.atleast_1d(x),
                                                  np.atleast_1d(y)]).T,
                                      0).T
    
    def xy_to_uv(self, x, y, reorder=True):
        """ w,y to u, v tangent plane projection (in arcsec from pointing center). 
        This uses pixels_to_coords->coords_to_uv
        """
        ra, dec = self.xy_to_radec(x, y, reorder=reorder)
        return self.radec_to_uv(ra, dec)

    def xy_to_ij(self, x, y, qid=None):
        """ converts quadrant coordinates (x,y) into ccd coordinates (i,j) """
        # recall:
        # q2 | q1
        # --------
        # q3 | q4
        if qid is None:
            if not hasattr(self, "qid"):
                raise ValueError("self.qid does not exist, please provide the qid")
            
            qid = self.qid
        
        if self.qid in [1, 4]:
            x = x+self.shape[1]

        if self.qid in [1, 2]:
            y = y+self.shape[0]

        return np.stack([x, y]) # i, j


    # radec
    def radec_to_xy(self, ra, dec, reorder=True):
        """ get the (x,y) quadrant positions given the sky ra, dec [in deg] coordinates """
        
        x, y = self.wcs.all_world2pix(np.asarray([np.atleast_1d(ra),
                                                  np.atleast_1d(dec)]).T,
                                      0).T
        if reorder and hasattr(self, "shape"):
            x = self.shape[1] -x -1 # starts at 0
            y = self.shape[0] -y -1 # starts at 0

        return np.stack([x, y])
    
    def radec_to_uv(self, ra, dec):
        """ radec to u, v (tangent plane projection in arcsec from pointing center) """
        # -1, 1 to have ra pointing rightward
        return np.asarray(tools.project([ra, dec], self.pointing))*180/np.pi * 3600  * np.atleast_1d(-1,1)

    def radec_to_ij(self, ra, dec, reorder=True, qid=None):
        """ radec to ccd coordinates (i,j) """
        x, y = self.radec_to_xy(ra, dec, reorder=True)
        return self.xy_to_ij(x, j, qid=qid)

    
    # uv
    def uv_to_xy(self, u, v, reorder=True):
        """ get the x, y quadrant position given the tangent plane coordinates u, v """
        ra, dec = self.uv_to_radec(u, v)
        return self.radec_to_xy(ra, dec, reorder=reorder)
    
    def uv_to_radec(self, u, v):
        """ get the ra, dec coordinates given the tangent plane coordinates u, v """
        # deproject contains the 3600
        # -1, 1 to have ra pointing rightward        
        return np.asarray(tools.deproject([np.asarray(u)*-1, np.asarray(v)], self.pointing))*180/np.pi

    def uv_to_ij(self, u, v, reorder=True, qid=None):
        """ get the i,j ccd pixel coordinates given the tangent plane coordinates u, v """
        x, y = self.uv_to_xy(u, v, reorder=True)
        return self.xy_to_ij(x, y, qid=None)

    # ij
    def ij_to_xy(self, i, j, qid=None):
        """ converts ccd coordinates (i,j) into quadrant coordinates (x,y) """
        # recall:
        # q2 | q1
        # --------
        # q3 | q4
        if qid is None:
            if not hasattr(self, "qid"):
                raise ValueError("self.qid does not exist, please provide the qid")
            
            qid = self.qid
        
        if self.qid in [1, 4]:
            i = i-self.shape[1]

        if self.qid in [1, 2]:
            j = j-self.shape[0]

        return np.stack([i, j]) #  # x, y

    def ij_to_radec(self, i, j, reorder=True, qid=None):
        """ get the (ra,dec) sky coordinates from the i,j ccd coordinates """
        x, y = self.ij_to_xy(i,j, qid=qid)
        return self.xy_to_radec(x,y, reorder=True)

    def ij_to_uv(self, i, j, reorder=True, qid=None):
        """ get the tangent plane coordinates from the i,j ccd coordinates """
        x, y = self.ij_to_xy(i,j, qid=qid)
        return self.xy_to_uv(x, y, reorder=reorder)

    # =============== #
    #  Properties     #
    # =============== #
    @property
    def wcs(self):
        """ """
        if not hasattr(self, "_wcs"):
            return None
        return self._wcs

    def has_wcs(self):
        """ """
        return self.wcs is None
        
    @property
    def pointing(self):
        """ requested telescope pointing [in degree] """
        if not hasattr(self, "_pointing"):
            return None
            
        return self._pointing


class WCS( WCSHolder ):
    def __init__(self, astropywcs=None, pointing=None):
        """ load the ztfimg.WCS solution built upon astropy's WCS solution. 
        = it knows how to convert (x,y), (u,v) and (ra,dec) systems =

        Parameters
        ----------
        astropywcs: [astropy.wcs.WCS or None] -optional-
            astropy wcs solution. 

        pointing: [ [float,float] or None] -optional-
            pointing RA, Dec coordinates (in deg)

        Returns
        -------
        ztfimg.WCS solution
        """
        if astropywcs is not None:
            self.set_wcs( astropywcs )
        if pointing is not None:
            self.set_pointing(*pointing)

    @classmethod
    def from_header(cls, header, pointingkey=["RA","DEC"]):
        """ load the ztfimg.WCS solution built upon astropy's WCS solution. 
        given the header information
        
        = it knows how to convert (x,y), (u,v) and (ra,dec) systems =

        Parameters
        ----------
        header: [header] 
            header containing the WCS solution keys

        pointing: [strings] -optional-
            header keys cointaining the RA, DEC position of the telescope pointing.

        Returns
        -------
        ztfimg.WCS solution
        """
        this = cls()
        this.load_wcs(header, pointingkey=pointingkey)
        return this
    
    @classmethod
    def from_filename(cls, filename, pointingkey=["RA","DEC"], suffix=None, **kwargs):
        """ load the ztfimg.WCS solution built upon astropy's WCS solution. 
        given the filename information.

        This looks for the header file (or that of the associated suffix) and 
        calls the from_header() class method
        
        = it knows how to convert (x,y), (u,v) and (ra,dec) systems =

        Parameters
        ----------
        filename: [string] 
            name of the file having the header containing the wcs information.
            (see suffix). This call ztfquery.io.get_file(). It looks for the given 
            file locally and download it if necessary (see kwargs).
            
        suffix: [string or None] -optional-
            ztfquery.io.get_file() option. It enables to change the suffix of the 
            given filename. 
            
        pointing: [strings] -optional-
            header keys cointaining the RA, DEC position of the telescope pointing.

        **kwargs goes to ztfquery.io.get_file()

        Returns
        -------
        ztfimg.WCS solution
        """
        from ztfquery import io
        header = io.fits.getheader( io.get_file(filename, suffix=suffix, **kwargs) )
        return cls.from_header(header, pointingkey=pointingkey)
    
    
