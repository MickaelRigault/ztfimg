""" DAOPhot associated tools """

import numpy as np
from scipy import stats

from . import stamps

def daoarray_to_pixel(arr):
    """ """
    from .tools import restride
    return np.sum(restride(np.mean([arr[1:,1:], arr[:-1,:-1] ], axis=0),
                                    2),
                            axis=(2,3))

class DAOPhotReader( object ):
    """ """
    def __init__(self, psffile):
        """ """
        if psffile is not None:
            self.load(psffile)
        
    # --------- #
    #  LOADER   #
    # --------- #
    def load(self, psffile):
        """ read and parse the daophot.psf file. """
        self._datalines = open(psffile).read().splitlines()
        _modelname, _structurewidth, _nparamprofile, _nstructures, _ , _instnorm, _central_height_adu, _xcenter, _ycenter = self._datalines[0].split()
        self._modelname = str(_modelname)
        self._structurewidth = int(_structurewidth) 
        self._nparamprofile = int(_nparamprofile) 
        self._nstructures = int(_nstructures) 
        self._instnorm = float(_instnorm) 
        self._centralheight_adu = float(_central_height_adu)
        self._xcenter = float(_xcenter) 
        self._ycenter = float(_ycenter) 
        self._profile_shape = np.asarray(self._datalines[1].split(), dtype="float")
        
        datatable = np.asarray(" ".join(self._datalines[2:]).replace("-"," -").replace("E -","E-").split(), dtype="float")
        self._structures = np.asarray([np.asarray(datatable[self._structurewidth*self._structurewidth*i:(i+1)*self._structurewidth*self._structurewidth
                                                ].reshape(self._structurewidth,
                                                          self._structurewidth),
                                        dtype="float")                  
                           for i in range(self.nstructures)])
        
        self._load_basemodel_()
        
    def _load_basemodel_(self):
        """ """
        if self.modelname == "GAUSSIAN":
            s0 = self.get_structstamp(0)
            # 2x because daophot oversample by 2
            sigmax,sigmay = 2* np.asarray(self._profile_shape, dtype="float")/np.sqrt(2*np.log(2))
            pixels = stamps.get_pixel_to_consider(0,self._structurewidth,0,self._structurewidth)
            self._covariance = [[(sigmax)**2,0], [0, (sigmay)**2]]
            self._basenorm = np.sqrt((2*np.pi)**2 * np.linalg.det(self._covariance))
            self._basedata = stats.multivariate_normal.pdf(pixels,
                                                           [s0.xref,s0.yref], 
                                                           cov=self._covariance
                                                        ).reshape(self._structurewidth,
                                                                  self._structurewidth)
            
        else:
            raise NotImplementedError("Only modelname=='GAUSSIAN' has been implemented.")
            
    # --------- #
    #  GETTER   #
    # --------- #
    def get_psf(self, coef_base, coefstruct, normed=True, asstamp=True, inpixels=False):
        """ Get the psf model using the linear combination of the PSF elements: base+ structure 
        
        Parameters
        ----------
        coef_base: [float]
            amplitude of the base-profile
            
        coefstruct: [1d-array]
            A amplitude per structure, i.e. [1,1,1] is self.nstructures=3
        
        normed: [bool] -optional-
            Shall the sum of all pixel be equal to 1 ?
        
        asstamp: [bool] -optional-
            Shall this return a stamp or an array ?
        
        inpixels: [bool] -optional-
            Shall this be in daophot oversampled unit or in actual pixels ?
            
        Returns
        -------
        Stamp/array (see asstamp)
        
        """
        data = coef_base*self.get_basestamp(asstamp=False, inpixels=False)
        struct = np.dot(self.structures.T, coefstruct).T
        array_ = data+struct
        if normed:
            array_/=array_.sum()

        if inpixels:
            array_ = daoarray_to_pixel(array_)

        if not asstamp:
            return array_
        return stamps.Stamp(array_)
    
    def get_basestamp(self, asstamp=True, inpixels=False):
        """ get the base-profile model
        
        Parameters
        ----------
        asstamp: [bool] -optional-
            Shall this return a stamp or an array ?
            
        inpixels: [bool] -optional-
            Shall this be in daophot oversampled unit or in actual pixels ?

        Returns
        -------
        Stamp/array (see asstamp)
        """
        array_ = self._basedata.copy()
        if inpixels:
            array_ = daoarray_to_pixel(array_)
            
        if not asstamp:
            return array_
        return stamps.Stamp(array_)
    
    def get_structstamp(self, index, asstamp=True, inpixels=False):
        """ Get the i-th structure. 
        
        Parameters
        ----------
        index: [int]
            index of the stamps (self.structures[index])
        
        asstamp: [bool] -optional-
            Shall this return a stamp or an array ?

        inpixels: [bool] -optional-
            Shall this be in daophot oversampled unit or in actual pixels ?

        Returns
        -------
        Stamp/array (see asstamp)
        """
        array_ = self.structures[index].copy()
        if inpixels:
            array_ = daoarray_to_pixel(array_)
            
        if not asstamp:
            return array_
        return stamps.Stamp(self.structures[index])
    
    # --------- #
    #  PLOTTER  #
    # --------- #
    def show(self, savefile=None, **kwargs):
        """ """
        import matplotlib.pyplot as mpl
        fig = mpl.figure(figsize=[8,2.5])
        prop = {**dict(cmap=mpl.cm.coolwarm),**kwargs}

        axg = fig.add_subplot(1,4,1)
        _ = self.get_basestamp().show(ax=axg, **prop)
        axg.set_title(f"base {self.modelname}")
        for i in range(self.nstructures):
            ax_ = fig.add_subplot(1,4,i+2)
            self.get_structstamp(i).show(ax=ax_, **prop)
            ax_.set_title(f"structure ext {i}", fontsize="medium")

        fig.tight_layout()
        if savefile is not None:
            fig.savefig(savefile)
            
        return fig
    
    # ================== #
    #    Properties      #
    # ================== #
    @property
    def basedata(self):
        """ 2d-array of the base model profile """
        return self._basedata
    
    @property
    def modelname(self):
        """ Name of the profile base profile """
        return self._modelname

    @property
    def nparamprofile(self):
        """ Number of parameter for the profile """
        return self._nparamprofile

    @property
    def structures(self):
        """ """
        return self._structures
    
    @property
    def structurewidth(self):
        """ Width of the structure """
        return self._structurewidth
    
    @property
    def nstructures(self):
        """ Width of the structure """
        return self._nstructures
    
    @property
    def instnorm(self):
        """ The instrumental magnitude corresponding to the 
        point-spread function of unit normalization """
        return self._instnorm
    
    @property
    def centralheight(self):
        """  Central height, in ADU, of the analytic function 
        which is used as the first-order approximation
        to the point-spread function in the stellar core """
        return self._centralheight_adu
    
    @property
    def centralpixel(self):
        """  X and Y coordinates of the center of the frame """
        return self._xcenter,self._ycenter
    
    
