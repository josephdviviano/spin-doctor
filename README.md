spin doctor
-----------

A set of tools for automatically removing artefacts associated with the SPIRAL
EPI sequence on GE hardware.

Steps:

1) Before combining spiral in and out images, run melodic ICA on each.
2) Use `flag_components.py` to identify compoents with:
    - High slice loadings relative to other slices in the same component ('single slice components')
    - Lots of mid-frequency power in the fft of that same slice
3) Use `fsl_regfilt` to remove these components from the data.
4) Combine spiral in and out.
5) Carry on with life.

