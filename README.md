# ztfphot
ZTF Photometry tools | simplification of astrobject dedicated to ZTF.

# Installation
## Getting the PS1Calibrator files

Ask Mickael

# ztfphot

## usage

Say you have a sciimg and its associated mask:
```python
sciimg = "ztf_20200204197199_000812_zr_c04_o_q3_sciimg.fits"
mask = "ztf_20200204197199_000812_zr_c04_o_q3_mskimg.fits"
```
then:
```python
from ztfphot import image
z = image.ZTFImage(sciimg, mask)
z.show(show_ps1cal=False) # set true if you do have the datafile
```

### easy bitmask access
```python
# Here is the default masking, True, means datamasked=nan for these cases
z.get_mask( tracks=True, ghosts=True, spillage=True,spikes=True,
            dead=True, nan=True, saturated=True, brightstarhalo=True,
            lowresponsivity=True, highresponsivity=True, noisy=True,
            sexsources=False, psfsources=False)
```
