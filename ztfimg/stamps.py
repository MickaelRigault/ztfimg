""" Stamps """
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def get_pixel_to_consider(xmin, xmax, ymin, ymax):
    """ """
    pixels_    = np.mgrid[int(xmin):int(xmax), int(ymin):int(ymax)].T
    init_shape = np.shape(pixels_)[:2]
    return np.asarray(pixels_.reshape(init_shape[0]*init_shape[1], 2), dtype="int")

def stamp_it( array, x0, y0, dx, dy=None, asarray=False):
    """ """
    if dy is None:
        dy = dx

    if len(np.atleast_1d(x0))>1:
        if not asarray:
            raise NotImplementedError("for multiple x0,y0, only asarray=True implemented. Loop over stamp_it() with single.")
        if len(x0) != len(y0):
            raise ValueError(f"x0 and y0 must have the same size ({len(x0)} and {len(y0)} given, respectively).")
        shapearr = np.shape(array)
        allstamps = np.ones( (len(x0), dy, dx)) * np.NaN
        flagout = (x0-dx/2+0.5<0) | (y0-dy/2+0.5<0) | \
                  (y0+dy/2+0.5>shapearr[0]) | (x0+dx/2+0.5>shapearr[1])

        allstamps[~flagout] = [_stamp_it_unique_(array, x_, y_, dx=dx, dy=dy, asarray=True)  for x_,y_ in zip(x0[~flagout], y0[~flagout])]
        return allstamps
    else:
        return _stamp_it_unique_(array, x0, y0, dx, dy=dy, asarray=asarray)
    

def _stamp_it_unique_(array, x0, y0, dx, dy, asarray=False):
    """ """
    lower_pixel = np.asarray([np.round(x0-dx/2+0.5), np.round(y0-dy/2+0.5)], dtype="int")
    upper_pixel = np.asarray([np.round(x0+dx/2+0.5), np.round(y0+dy/2+0.5)], dtype="int")
    x_slice = slice(lower_pixel[0], upper_pixel[0])
    y_slice = slice(lower_pixel[1], upper_pixel[1])
    data_patch = array[y_slice].T[x_slice].T
    if asarray:
        return data_patch

    return Stamp(data_patch, x0-lower_pixel[0], y0-lower_pixel[1])


class Stamp(object):
    
    def  __init__(self, datapatch=None, x0=None, y0=None):
        """ Initialize the stamp
        
        Parameters
        ----------
        datapatch: [2d-array] -optional-
            Stamp data (2d-array)

        x0, y0: [float] -optional-
            Coordinate of the centroid. 
            If None, the center of the stamp in that direction will be used.
            
        Returns
        -------
        Stamp
        """
        self.set_data(datapatch)
        self.set_coord_ref(x0, y0)
        
    # ============== #
    #  Methods       #
    # ============== #
    # --------- #
    #  SETTER   #
    # --------- #
    def set_data(self, datapatch):
        """ set the stamp data (2d-array) """
        self._data = datapatch
        
    def set_coord_ref(self, xref, yref):
        """ set the coordinates of the stamp's centroid 
        
        If None, the center of the stamp in that direction will be used.
        (shape-1)/2.
        """
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
        """ Loads the RegularGridInterpolator 

        Based on scipy.interpolate.RegularGridInterpolator

        Parameters
        ----------
        method: [string] -optional-
            The method of interpolation to perform. 
            Supported are “linear” and “nearest”. 
            This parameter will become the default for the object’s __call__ method.

        bounds_error: [bool] -optional-
            If True, when interpolated values are requested outside of the domain 
            of the input data, a ValueError is raised. 
            If False, then fill_value is used.

        fill_value: [float] -optional-
            The value to use for points outside of the interpolation domain. 
            If None, values outside the domain are extrapolated.
        
        **kwargs goes to RegularGridInterpolator

        Returns
        -------
        Void
        """
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
        """ returns a flat shape 
        
        Evaluate the grid at the given pixel centroids `xnew`, `ynew`

        Returns
        -------
        array of the shape of xnew/ynew
        """
        return self.interpolator(np.moveaxis(np.array([ynew+self.yoffset, xnew+self.xoffset]),0,-1))

    def project_to_grid(self, xmax, ymax, xmin=0, ymin=0, newxoffset=0, newyoffset=0):
        """ returns a flat shape 
        
        xmax, ymax, xmin, ymin: [float]
            boundaries of the new grid. 

        newxoffset, newyoffset: [float] -optional-
            offset from the center of the new shape.

        Returns
        -------
        1d data (flatten grid)
        """
        pixels = get_pixel_to_consider(xmin, xmax, ymin, ymax  ) - [newxoffset,newyoffset]
        return self.project_to(*np.moveaxis(pixels,0,-1)).reshape(int(ymax)-int(ymin), int(xmax)-int(xmin))

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
        fake = self.get_centered(np.asarray(self.shape)+2,fill_value=0,  asstamp=False).T
    
        dy,dx = np.asarray(fake.shape)
        x_slice = slice(int(x-dx/2+0.5), int(x+dx/2+0.5))
        y_slice = slice(int(y-dy/2+0.5), int(y+dy/2+0.5))
        if ascopy:
            dataf = data.copy()
            dataf[y_slice].T[x_slice] += fake
            return dataf
        data[y_slice].T[x_slice] += fake
    

    def get_centered(self, shape=None, fill_value=np.NaN, asstamp=False, centroid=None):
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

        centroid: [None or 2d-array] -optional-
            Coordinates of the centroid in the new stamp.
            - None: This is be the center of the new stamp
            - 2d-array: new x0,y0.
            for instance, to reproduce the same as the current stamp:
            newstamp = self.get_centered(self.shape, asstamp=True, centroid=(self.xref, self.yref))

        Returns
        -------
        array or Stamp (see asstamp)

        """
        if shape is None:
            shape = self.shape
        
        if not hasattr(self, "_fill_value") or (fill_value is not None and self._current_fillvalue != fill_value):
            self.load_interpolator(fill_value=fill_value)

        if centroid is None:
            centroid = (np.asarray(shape)-1)/2

        centroid_shift = centroid - self.central_pixel
        centerddata = self.project_to_grid(*shape, newxoffset=centroid_shift[0], newyoffset=centroid_shift[1])
        if not asstamp:
            return centerddata
        
        return Stamp(centerddata, x0=centroid[0], y0=centroid[1])



    # --------- #
    #  PLOTTER  #
    # --------- #
    def show(self, ax=None, savefile=None, **kwargs):
        """ show the stamps (imshow)
        
        Parameters
        ----------
        ax:  [matplotlib's Axes] -optional-
            axes where the plot will be made. 
            A figure and an axes will be created is None

        savefile: [string] -optional-
            Path to where the figure will be stored.
            // ignored if None //

        Returns
        -------
        figure
        """
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
        
        if savefile:
            fig.savefig(savefile)
            
        return fig
    # ============== #
    #  Properties    #
    # ============== #
    @property
    def data(self):
        """ 2d-array data of the stamp """
        return self._data
    
    @property
    def shape(self):
        """ shape of the stamp's 2d array """
        return self.data.shape
    
    @property
    def xref(self):
        """ x-coordinate of the centroid """
        return self._xref
    
    @property
    def yref(self):
        """ y-coordinate of the centroid """
        return self._yref
    
    @property
    def xoffset(self):
        """ x-offset of the centroid with respect to the stamp center """
        return self.xref - self.central_pixel[0]
    
    @property
    def yoffset(self):
        """ y-offset of the centroid with respect to the stamp center """
        return self.yref - self.central_pixel[1]

    @property
    def central_pixel(self):
        """ returns the (x,y) pixel coordinate of the stamp center """
        return np.asarray(self.shape[::-1])/2 - 0.5
    
    @property
    def interpolator(self):
        """ RegularGridInterpolator loaded with the stamp data. see  load_interpolator() """
        if not hasattr(self, "_interpolator"):
            self.load_interpolator()
        return self._interpolator

    @property
    def pixels(self):
        """ """
        return get_pixel_to_consider(0, self.shape[1],0, self.shape[0])
        
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
        """ Set stamps from list of 2d-arrays
        
        Parameters
        ----------
        data: [3d-array] 
            List of 2d-arrays, each will become a stamp.
            
        x0, y0: [2d-array or None] -optional-
            Centroid of the stamps (same lens as data)
            if None, the stamp's centroid will be used.

        indexes: [1d-array] -optional-
            
        """
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
        
