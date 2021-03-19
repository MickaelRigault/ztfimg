""" WCS Class handler """

import numpy as np
from astropy.wcs import WCS

from . import tools


class WCSHolder( object ):
    """ """


    # =============== #
    #  Methods        #
    # =============== #
    # --------- #
    #  LOADER   #
    # --------- #        
    def load_wcs(self, header, pointingkey=["RA","DEC"]):
        """ """
        # - wcs
        self.set_wcs( WCS(header) )
        
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
    def set_wcs(self, wcs):
        """ """
        self._wcs = wcs

    def set_pointing(self, ra, dec):
        """ """
        self._pointing = ra, dec
        
    # --------- #
    #  Convert  #
    # --------- #
    def xy_to_radec(self, x, y):
        """ get sky ra, dec [in deg] coordinates given the (x,y) ccd positions  """
        return self.wcs.all_pix2world(np.asarray([np.atleast_1d(x),
                                                  np.atleast_1d(y)]).T,
                                      0).T
    
    def xy_to_uv(self, x, y):
        """ w,y to u, v tangent plane projection (in arcsec from pointing center). 
        This uses pixels_to_coords->coords_to_uv
        """
        ra, dec = self.xy_to_radec( x, y)
        return self.radec_to_uv(ra, dec)

    # coords -> 
    def radec_to_xy(self, ra, dec):
        """ get the (x,y) ccd positions given the sky ra, dec [in deg] corrdinates """
        return self.wcs.all_world2pix(np.asarray([np.atleast_1d(ra),
                                                  np.atleast_1d(dec)]).T,
                                      0).T
    
    def radec_to_uv(self, ra, dec):
        """ radec to u, v (tangent plane projection in arcsec from pointing center) """
        return np.asarray(tools.project([ra, dec], self.pointing))*180/np.pi * 3600
    
    # uv -> 
    def uv_to_xy(self, u, v):
        """ get the x, y ccd position given the tangent plane coordinates u, v """
        ra, dec = self.uv_to_radec(u, v)
        return self.radec_to_xy(ra, dec)
    
    def uv_to_radec(self, u, v):
        """ get the ra, dec coordinates given the tangent plane coordinates u, v """
        return np.asarray(tools.deproject([u, v], self.pointing))*180/np.pi

    # =============== #
    #  Properties     #
    # =============== #
    @property
    def wcs(self):
        """ """
        if not hasattr(self, "_wcs"):
            return None
        self._wcs

    def has_wcs(self):
        """ """
        return self.wcs is None
        
    @property
    def pointing(self):
        """ requested telescope pointing [in degree] """
        if not hasattr(self, "_pointing"):
            return self._pointing
            
        return self._pointing
