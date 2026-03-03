import pandas
import numpy as np

class SourceVignets():
    """ """
    def __init__(self, vignets, centroids, sources=None):
        """ """
        self._vignets = np.asarray(vignets)
        self._centroids = np.asarray(centroids)
        self._sources = sources

    @classmethod
    def from_image(cls, img, bkgd=None, err=None, 
                   padding=[2, 3], a_range=[0, 3],
                   thresh_=5, **kwargs):
        """ """
        from scipy.stats import median_abs_deviation
        from .tools import extract_sources
        
        if err is None:
            scatter = median_abs_deviation(img, scale="normal")
            err = np.ones( img.shape ) * scatter
            
        if bkgd is None:
            bkgd = np.median(img)

        # images 'background corrected' is expected.
        img = img-bkgd # copy hence no img -= bkgd
        sources = extract_sources(img, err=err, thresh_=thresh_, **kwargs)
        return cls.from_image_and_source(img, sources)
        
    @classmethod
    def from_image_and_source(cls, img, sources, padding=[2, 3], a_range=[0, 3]):
        """ """
        # make sure you are not affecting inputs
        sources = sources.copy() 
        # select only correct sources
        if a_range is not None:
            sources = sources[sources["a"].between(*a_range)]

        # compute the expedted width from the median width and height
        sources["width"] = sources["xmax"]-sources["xmin"]
        sources["height"] = sources["ymax"]-sources["ymin"]
        
        half_width = int( sources["width"].median()/2 ) + padding[1]
        half_height = int( sources["height"].median()/2 ) + padding[0]
        
        # consider only sources off the edges
        offedge_flag = sources["x"].between(half_width, img.shape[1]-half_width) * \
                       sources["y"].between(half_height, img.shape[0]-half_height)
        sources = sources[ offedge_flag ].reset_index()

        # build the vignets
        # get the vignets of each sources (same size for each)
        vignets = np.asarray([img[int(y0)+1-half_height: int(y0)+1+half_height, 
                                  int(x0)+1-half_width: int(x0)+1+half_width]
                              for x0, y0 in zip(sources["x"], sources["y"]) ])
        
        # get the exact centroid (x, y) in the new vignets coordinates
        int_centroid = np.asarray([ [(int(x0)+1-x0), (int(y0)+1-y0)] for x0, y0 in zip(sources["x"], sources["y"]) ])
        centroids = (np.asarray([half_width, half_height]) - int_centroid).T

        # and build the instance
        return  cls(vignets, centroids, sources=sources)

    def get_gaussian_psf(self, sources=None, incl_flux=False):
        """ """
        # use Gaussian2D template
        from astropy.modeling.models import Gaussian2D
        
        if sources is None:
            sources = self._sources
            
        if sources is None:
            raise ValueError("No sources set and none given.")
            
        
        # for some reason astropy's Gaussian2D is not normalize.
        norm = 1/(2 * np.pi * np.sqrt(sources["x2"]) * np.sqrt(sources["y2"]))
        if incl_flux:
            flux = sources["flux"]
        else:
            flux = 1 # normalized

        x0s, y0s = self.centroids
        gaussians = Gaussian2D(amplitude=(norm * flux).values[:,None, None],
                               x_mean=x0s[:,None, None], 
                               y_mean=y0s[:,None, None], 
                               x_stddev= np.sqrt(sources["x2"]).values[:,None, None], 
                               y_stddev= np.sqrt(sources["y2"]).values[:,None, None], 
                               theta=sources["theta"].values[:,None, None])
        
        return gaussians(self.xpixels[None, :], self.ypixels[None, :])

    def get_moments(self, psf_weighted=True, psf=None, join_sources=False, sources=None):
        """ """
        vignets = self.vignets.copy()
        if psf_weighted:
            if psf is None:
                psf = self.get_gaussian_psf(incl_flux=False)
    
            vignets *= psf
             
        ampl_per_vignets = vignets.sum(axis=(-2, -1))
    
        moments = {# second moments
                "m_x2": (self.x_m_x0**2 * vignets).sum(axis=(-2,-1)) / ampl_per_vignets,
                "m_y2": (self.y_m_y0**2 * vignets).sum(axis=(-2,-1)) / ampl_per_vignets,
                "m_xy": (self.x_m_x0 * self.y_m_y0 * vignets).sum(axis=(-2,-1)) / ampl_per_vignets,
                # third moments
                "m_x3": (self.x_m_x0**3 * vignets).sum(axis=(-2,-1)) / ampl_per_vignets,
                "m_y3": (self.y_m_y0**3 * vignets).sum(axis=(-2,-1)) / ampl_per_vignets,
                "m_x2y": (self.x_m_x0**2 * self.y_m_y0 * vignets).sum(axis=(-2,-1)) / ampl_per_vignets,
                "m_xy2": (self.x_m_x0 * self.y_m_y0**2 * vignets).sum(axis=(-2,-1)) / ampl_per_vignets,
               }

        df = pandas.DataFrame(moments, index=self.sources.index if self.has_sources() else None)
        if join_sources:
            if sources is None:
                if self.has_sources():
                    sources = self.sources.copy()
                else:
                    raise ValueError("cannot join with sources, none provided and none set.")
            df = df.join(sources)

        return df
            
    def has_sources(self):
        """ """
        return self.sources is not None

    # -------- #
    # PLOTTER  #
    # -------- #        
    def show_psfmodel(self, index, psf=None, axes=None):
        import matplotlib.pyplot as plt
        from matplotlib.colors import PowerNorm
        
        if psf is None:
            psf = self.get_gaussian_psf(incl_flux=True)
            
        fig, (ax, ax2, ax3) = plt.subplots(ncols=3)
        
        norm = PowerNorm(1, *np.percentile(self.vignets[index], [1, 99]))
        prop_imshow = dict(origin="lower", norm=norm)
        
        # vignets
        ax.imshow(self.vignets[index], **prop_imshow)
        ax.scatter(*self.centroids[:, index], marker="x", color="tab:orange")

        # psf model
        ax2.imshow(psf[index], **prop_imshow)
        ax2.scatter(*self.centroids[:, index], marker="x", color="tab:orange")
        
        ax3.imshow(psf[index]-self.vignets[index], **prop_imshow)
        ax3.scatter(*self.centroids[:, index], marker="x", color="tab:orange")
        return fig

    # ============= #
    #   Properties  #
    # ============= #
    @property
    def vignets(self):
        """ """
        return self._vignets

    @property
    def centroids(self):
        """ """
        return self._centroids

    @property
    def sources(self):
        """ source catalog if any """
        return self._sources
        
    @property
    def xpixels(self):
        """ """
        return np.arange(self.vignets.shape[-1])[None,:]
    
    @property
    def ypixels(self):
        """"""
        return  np.arange(self.vignets.shape[-2])[:, None]
        
    @property
    def x_m_x0(self):
        """ """
        x0s = self.centroids[0]
        return (np.arange(self.vignets.shape[-1]) - x0s[:, None])[:,None,:]

    @property
    def y_m_y0(self):
        """ """
        y0s = self.centroids[1]        
        return (np.arange(self.vignets.shape[-2]) - y0s[:, None])[...,None]
        # get the pixels coordinates in offset from the centroid
        
