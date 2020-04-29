""" Stamps """
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def get_pixel_to_consider(xmin, xmax, ymin, ymax):
    """ """
    pixels_    = np.mgrid[int(xmin):int(xmax), int(ymin):int(ymax)].T
    init_shape = np.shape(pixels_)[:2]
    return np.asarray(pixels_.reshape(init_shape[0]*init_shape[1], 2), dtype="int")


class Stamp(object):
    
    def  __init__(self, datapatch=None, x0=None, y0=None):
        """ """
        self.set_data(datapatch)
        self.set_coord_ref(x0, y0)
        
    # ============== #
    #  Methods       #
    # ============== #
    # --------- #
    #  SETTER   #
    # --------- #
    def set_data(self, datapatch):
        """ """
        self._data = datapatch
        
    def set_coord_ref(self, xref, yref):
        """ """
        if xref is None:
            xref = 0.5*(self.shape[1]-1)
        if yref is None:
            yref = 0.5*(self.shape[0]-1)
            
        self._xref = xref
        self._yref = yref
        
    # --------- #
    #  LOADER   #
    # --------- #
    def load_interpolator(self, method="linear", bounds_error=False, 
                          fill_value=np.NaN, **kwargs):
        """ Loads the RegularGridInterpolator"""
        self._x = np.arange(self.shape[1])# - self.xoffset
        self._y = np.arange(self.shape[0])# - self.yoffset
        self._current_fillvalue = fill_value
        self._interpolator = RegularGridInterpolator((self._x,self._y), self.data, 
                                                      method=method, 
                                                      bounds_error=bounds_error, 
                                                      fill_value=fill_value, 
                                                      **kwargs)
    # --------- #
    #  GETTER   #
    # --------- #
    def project_to(self, xnew, ynew):
        """ returns a flat shape """
        return self.interpolator(np.moveaxis(np.array([ynew+self.yoffset, xnew+self.xoffset]),0,-1))

    def project_to_grid(self, xmax, ymax, xmin=0, ymin=0, newxoffset=0, newyoffset=0):
        """ returns a flat shape """
        pixels = get_pixel_to_consider(xmin, xmax, ymin, ymax  ) - [newxoffset,newyoffset]
        return self.project_to(*np.moveaxis(pixels,0,-1)).reshape(int(ymax)-int(ymin),int(xmax)-int(xmin))

    def insert_to(self, data, x, y, ascopy=True):
        """ Insert the current stamp inside the given data (or a copy of)
        
        Parameters
        ----------
        data: [2D array]
            The image array the stamp should be added to

        x, y: [float]
            Centroid of stamp inside the image

        ascopy: [bool] -optional-
            Shall this update the given data (if so nothing is returned)
            of shall return a copy of the data with the stamp added to it ?

        Returns
        -------
        None or 2D array (see ascopy)
        
        """
        fake = self.get_centered(np.asarray(self.shape)+2,fill_value=0,  asstamp=False)
    
        dy,dx = np.asarray(fake.shape)
        x_slice = slice(int(x-dx/2+0.5), int(x+dx/2+0.5))
        y_slice = slice(int(y-dy/2+0.5), int(y+dy/2+0.5))
        if ascopy:
            dataf = data.copy()
            dataf[y_slice].T[x_slice] += fake.data
            return dataf
        data[y_slice].T[x_slice] += fake.data
    

    def get_centered(self, shape=None, fill_value=np.NaN, asstamp=False):
        """ Get an centered version of the stamp, centered on a grid of the given size
        
        Parameters
        ----------
        shape: [2D array or None] -optional-
            size of the new grid (height, width), if None current shape used

        fill_value: [float or None] -optional-
            Value of the pixels extrapolated. 
            If None, the current is used. 
            Reloads self.load_interpolator() if necessary

        asstamp: [bool] -optional-
            Shall this return a 2D array or a Stamp

        Returns
        -------
        array or Stamp (see asstamp)

        """
        if shape is None:
            shape = self.shape
        
        if not hasattr(self, "_fill_value") or (fill_value is not None and self._current_fillvalue != fill_value):
            self.load_interpolator(fill_value=fill_value)
        
        centroid_shift = (np.asarray(shape)-1)/2- self.central_pixel
        centerddata = self.project_to_grid(*shape, newxoffset=centroid_shift[0], newyoffset=centroid_shift[1])
        if not asstamp:
            return centerddata
        
        return Stamp(centerddata)



    # --------- #
    #  PLOTTER  #
    # --------- #
    def show(self, ax=None, savefile=None, **kwargs):
        """ """
        import matplotlib.pyplot as mpl
        if ax is None:
            fig = mpl.figure(figsize=[6,4])
            ax = fig.add_axes([0.1,0.1,0.8,0.8])
        else:
            fig = ax.figure
            
        prop = {**dict(origin="lower"), **kwargs}
        ax.imshow(self.data, **prop)
        
        ax.axvline(self.xref, color="w",ls="-", lw=1)
        ax.axhline(self.yref, color="w",ls="-", lw=1)
        
    # ============== #
    #  Properties    #
    # ============== #
    @property
    def data(self):
        """ """
        return self._data
    
    @property
    def shape(self):
        """ """
        return self.data.shape
    
    @property
    def xref(self):
        """ """
        return self._xref
    
    @property
    def yref(self):
        """ """
        return self._yref
    
    @property
    def xoffset(self):
        """ """
        return self.xref - self.central_pixel[0]
    
    @property
    def yoffset(self):
        """ """
        return self.yref - self.central_pixel[1]

    @property
    def central_pixel(self):
        """ returns the (x,y) pixel coordinate """
        return np.asarray(self.shape[::-1])/2 - 0.5
    
    @property
    def interpolator(self):
        """ """
        if not hasattr(self, "_interpolator"):
            self.load_interpolator()
        return self._interpolator

    
class StampCollection():
    """ """
    def __init__(self, data, x0, y0, indexes=None):
        """ """
        self.set_stamps(data, x0=x0, y0=y0, indexes=None)
        
    # ============== #
    #  Methods       #
    # ============== #
    # --------- #
    #  SETTER   #
    # --------- #
    
    def set_stamps(self, data, x0=None, y0=None, indexes=None):
        """ """
        if indexes is None:
            indexes = np.arange(len(x0))
        if x0 is None:
            x0 = [None for i in range(len(data))]
            
        if y0 is None:
            y0 = [None for i in range(len(data))]
            
        self._stamp = {i_:Stamp(data_, x0_, y0_)
                       for i_,data_, x0_, y0_ in zip(indexes,data, x0, y0)}
    # --------- #
    #  LOADER   #
    # --------- #
    def load_interpolator(self, method="linear", bounds_error=False, 
                          fill_value=np.NaN, indexes=None, **kwargs):
        """ """
        if indexes is None:
            indexes = self.indexes

        return [self.stamp[i].load_interpolator(method=method, bounds_error=bounds_error, 
                                        fill_value=fill_value, **kwargs)
               for i in indexes]
    
    # --------- #
    #  GETTER   #
    # --------- #
    def get_stamp(self, index):
        """ """
        return self.stamp[index]
    
    def project_to(self, xnew, ynew, indexes=None, get="full"):
        """ """
        if indexes is None:
            indexes = self.indexes
        return [self.stamp[i].project_to(xnew, ynew) for i in indexes]
        
    def project_to_grid(self, xmax, ymax, xmin=0, ymin=0, newxoffset=0, newyoffset=0, indexes=None):
        """ returns a flat shape """
        if indexes is None:
            indexes = self.indexes
        return [self.stamp[i].project_to_grid(xmax,ymax, xmin, ymin) for i in indexes]


    
    # ============== #
    #  Properties    #
    # ============== #
    @property
    def stamp(self):
        """ dict containing the stamps """
        return self._stamp
    
    @property
    def indexes(self):
        """ """
        return self.stamp.keys()
        
