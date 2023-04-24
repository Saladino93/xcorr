# xcorr

Code useful for cross-correlation cosmological analyses.

## Examples

```Python
from xcorr.spectra import spectra as sp
...
Cgg = sp.MapsReader(gmask, gmask, False, False, binning, nside, lmax_sht = lmax, filename = filename)
...
ggcounts = Cgg(deltagk, deltagh, alpha, alpha, lmax = lmax)
...
```
The above code defines an object, based on the popular and well tested `pymaster`, and it take the spectrum of two fields, accounting for any extra rescaling `alpha`, up to some `lmax`. 

Sometimes it is important to have a high-resolution of the maps, while needing only to do calculations up to some mode.


For a big project using this, see the ACT DR6 CMB lensing cross-correlation with DES Y3 Maglim surveys.

----------

Will list in the future the most important packages, projects or contributions to this package.


