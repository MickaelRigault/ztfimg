# ztfimg
ZTF Images tools | like a pure-python very simple version of ds9 for ZTF

# Installation
## Getting the PS1Calibrator files

Ask Mickael

# ztfimg

## usage

Say you have a sciimg and its associated mask:
```python
sciimg = "ztf_20200204197199_000812_zr_c04_o_q3_sciimg.fits"
mask = "ztf_20200204197199_000812_zr_c04_o_q3_mskimg.fits"
```
then:
```python
from ztfimg import image
z = image.ZTFImage(sciimg, mask)
z.show(show_ps1cal=False) # set true if you do have the datafile
```
![](examples/sciimg_masked_bkgdsub.png)


This image is masked for bad pixels values (see bitmasking below) and background subtracted. The background is the default sep.Background ; [sep](https://sep.readthedocs.io/en/v1.0.x/api/sep.extract.html) is a python version of sextractor. 


### Bitmasking access
```python
# Here is the default masking, True, means datamasked=nan for these cases
z.get_mask( tracks=True, ghosts=True, spillage=True,spikes=True,
            dead=True, nan=True, saturated=True, brightstarhalo=True,
            lowresponsivity=True, highresponsivity=True, noisy=True,
            sexsources=False, psfsources=False)
```

### Extracting sources with SEP

You can run `sep.extract` to extract ellipses à la sextractor. simply run:
```python
z.extract_sources(data="dataclean", setmag=True)
```

Access the results as:
```python
z.sources
```

### Matching sources and the PS1Catalog
Nothing is easier, do:
```python
z.match_sources_and_ps1cat()
```
The `ztfimg.CatMatch` resulting instance is stored as `z.sources_ps1cat_match`. See methods inthere. 

You can also directly get matched values such as (x,y) ccd positions, (ra,dec) or mag values by doing
```python
z.get_sources_ps1cat_matched_entries(["ra","dec","x","y","mag"])
```
```
ps1_index	source_index	angsep_arcsec	ps1_ra	source_ra	ps1_dec	source_dec	ps1_x	source_x	ps1_y	source_y	ps1_mag	source_mag
0	0	200	0.177830	80.458424	80.458457	59.081236	59.081190	1524.971933	1524.929682	109.828648	109.999139	14.300	14.298286
1	1	439	0.091733	80.580818	80.580863	59.042257	59.042246	1315.658142	1315.579032	269.438577	269.482766	14.407	14.391963
2	2	2692	0.134097	80.643587	80.643647	58.627628	58.627608	1341.017199	1340.913537	1748.049916	1748.132412	14.422	14.414028
3	3	203	0.212481	80.378699	80.378784	59.076712	59.076673	1671.527068	1671.384957	111.507782	111.662257	14.432	14.426520
4	4	1222	0.195256	80.465035	80.465061	58.895667	58.895614	1577.243713	1577.213421	767.680720	767.871187	14.340	14.335927
5	5	2185	0.161587	79.866613	79.866664	58.679673	58.679636	2752.683959	2752.603432	1419.232736	1419.370579	14.317	14.305793
6	6	1496	0.072166	81.037901	81.037929	58.867203	58.867217	538.472196	538.415826	966.704057	966.660447	14.529	14.509354
7	7	1344	0.120711	80.904353	80.904390	58.892260	58.892233	774.908561	774.848910	855.870462	855.973706	14.542	14.543835
8	8	970	0.173730	80.733465	80.733534	58.952595	58.952562	1067.479463	1067.365401	613.298427	613.426647	14.342	14.32015
```

### counts<->flux<->mag

you have all the `count_to_flux` or `mag_to_counts` ets combinations as methods.

### (x,y)<->(ra,dec)

use `z.coords_to_pixels(ra, dec)` or `z.pixels_to_coords(x,y)`
