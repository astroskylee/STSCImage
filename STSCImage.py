"""Public entry point for STSCImage.

STSCImage stands for Space Telescope Science Color Image. The implementation
currently lives in ``Tian_color.py``; this module provides the renamed import
path while keeping older notebooks compatible.
"""

from pathlib import Path

from Tian_color import *  # noqa: F401,F403


def Euclidimg(ra, dec, size, path, cred, output_jpg="Euclid_color.jpg"):
    """Download Euclid FITS cutouts and render one color JPEG."""

    from Download_Euclid import EUC_download

    outdir = Path(path)
    fits_paths = EUC_download(ra, dec, size, outdir, cred)
    jpg_path = outdir / output_jpg
    mk_colorimg(
        [fits_paths["NIR_J"], fits_paths["NIR_Y"], fits_paths["VIS"]],
        output_jpg=jpg_path,
        input_mode="raw",
    )
    return {"fits": fits_paths, "jpg": jpg_path}
