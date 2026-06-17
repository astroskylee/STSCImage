# STCI

STCI stands for **SpaceTelescopeColorImage**.

STCI is a small astronomy image-composition tool for making display-ready
color JPEGs from space telescope cutouts and other aligned mono images.

## Installation

```bash
pip install STCI
```

The installed Python import module is `STCI`.

## Example usage

`mk_colorimg` creates one color JPEG from either a 3-channel RGB array or three mono images in `(R, G, B)` order. For Euclid-style color images, the default mapping is `NIR_J`, `NIR_Y`, `VIS`.

```python
from STCI import mk_colorimg

mk_colorimg(
    [
        "cutout_H.fits",    # R channel
        "cutout_VIS.fits",  # G channel
        "cutout_Y.fits",    # B channel
    ],
    output_jpg="target_mtf_vis_y_h.jpg",
    input_mode="raw",
)
```

For a NumPy RGB cube:

```python
from STCI import mk_colorimg

mk_colorimg(rgb_array, output_jpg="target_color.jpg", input_mode="normalized")
```

## Download a Euclid Color Image

`Euclidimg` downloads Euclid DR1 `VIS`, `NIR_Y`, `NIR_J`, and `NIR_H` FITS cutouts, then renders one color JPEG using the `NIR_J / NIR_Y / VIS` channel order.

```python
from STCI import Euclidimg

result = Euclidimg(
    ra=50.7163333,
    dec=-39.7693889,
    size=5.0,
    path="euclid_color",
    cred="Euclid/cred.txt",
    output_jpg="EUCLJ032251.92-394609.8.jpg",
    ReplaceL=True,
    RGB="auto",
)

print(result["jpg"])
print(result["fits"])
print(result["rgb_bands"])
```

Arguments:

- `ra`, `dec`: target coordinates in degrees.
- `size`: cutout radius in arcsec. For example, `size=5.0` makes a `10" x 10"` image.
- `path`: output directory for the FITS files and JPEG.
- `cred`: Euclid credentials file passed to `astroquery.esa.euclid`.
- `output_jpg`: optional JPEG filename written inside `path`.
- `ReplaceL`: if `False`, skip replacing the CIELab L* channel with the stretched blue luminosity channel.
- `RGB`: Euclid bands in `(R, G, B)` order. Use `"auto"` for the default available-band choice, or pass bands such as `("NIR_H", "NIR_Y", "VIS")` or `("H", "Y", "VIS")`.

The returned dictionary contains the selected FITS paths and the final JPEG path. If the first overlapping mosaic tile for a band is empty or all zero, the downloader tries the next matching tile.

## Download Euclid FITS Only

```python
from Download_Euclid import EUC_download

fits_paths = EUC_download(
    ra=50.7163333,
    dec=-39.7693889,
    size=5.0,
    path="euclid_fits",
    cred="Euclid/cred.txt",
)
```
