# SCOPE

Color-image composition tools for astronomical observations.

## Example usage

`mk_colorimg` creates one color JPEG from either a 3-channel RGB array or three mono images in `(R, G, B)` order. For Euclid-style color images, pass the redder band as red, VIS as green, and the bluer NIR band as blue.

```python
from Tian_color import mk_colorimg

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
from Tian_color import mk_colorimg

mk_colorimg(rgb_array, output_jpg="target_color.jpg", input_mode="normalized")
```
