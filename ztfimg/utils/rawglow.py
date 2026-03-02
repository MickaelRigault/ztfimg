
import numpy as np
from scipy import optimize

class SideGlow():
    """ """
    def __init__(self, rawdata):
        """ """
        self._rawdata = np.asarray(rawdata, dtype="float")
        self._raw1d = np.nanmedian(self._rawdata, axis=0)
        self._raw1dpixels = np.arange(len(self._raw1d))
        
    # Model to fit
    @staticmethod
    def get_glow_model(pixels_, baseline, p0, ampl=1):
        """ """
        return baseline + ampl*np.exp(-pixels_/p0)

    def fit_glowparam(self, guess, nfirst=300, nveto=30):
        """ """
        pixels_to_fit, data_to_fit = self.get_firstpixels(nfirst=nfirst, 
                                                          nveto=nveto)
        def get_leastsquare(params):
            """ """
            model = self.get_glow_model(pixels_to_fit, *params)
            return np.nansum( (model-data_to_fit)**2)
    
        return optimize.minimize(get_leastsquare, x0=guess)

    def get_firstpixels(self, nfirst=300, nveto=30):
        """ """
        return np.arange(nveto, nfirst), self.data1d[nveto:nfirst]
        
    def get_skylevel(self, firstpixels=None, nlast=100):
        """ """
        if firstpixels is None:
            _, firstpixels = self.get_firstpixels()
            
        # value of the sky, asymptote of the first pixels 
        return np.nanmedian(firstpixels[-nlast:])

    def get_corrected_data(self, param=None, guess=None, rm_skylevel=False):
        """ """
        
        if param is None:
            if guess is None:
                skylevel = self.get_skylevel()
                guess = [skylevel, 50, 50]
            param = self.fit_glowparam(guess)["x"]

        if not rm_skylevel:
            param[0] = 0
            
        model1d = self.get_glow_model(self.pixels1d, *param)
        return self.data - model1d[None, :]

    # ============= #
    #  properties   #
    # ============= #
    @property
    def data(self):
        """ """
        return self._rawdata

    @property
    def data1d(self):
        """ vertical median stacking of data """
        return self._raw1d

    @property
    def pixels1d(self):
        """ pixels sampling associated to data1d """
        return self._raw1dpixels
