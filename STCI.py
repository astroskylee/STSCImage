"""Public entry point for STCI."""

from pathlib import Path

from Tian_color import *  # noqa: F401,F403


_BAND_ALIASES = {
    "VIS": "VIS",
    "V": "VIS",
    "Y": "NIR_Y",
    "J": "NIR_J",
    "H": "NIR_H",
    "NIRY": "NIR_Y",
    "NIRJ": "NIR_J",
    "NIRH": "NIR_H",
    "NIR_Y": "NIR_Y",
    "NIR_J": "NIR_J",
    "NIR_H": "NIR_H",
}

_AUTO_RGB = (
    ("NIR_J", "NIR_Y", "VIS"),
    ("NIR_H", "NIR_Y", "VIS"),
    ("NIR_H", "NIR_J", "VIS"),
    ("NIR_J", "NIR_H", "VIS"),
)


def _band_name(band):
    key = str(band).strip().upper().replace("-", "_")
    if key not in _BAND_ALIASES:
        raise ValueError(f"Unknown Euclid band: {band}")
    return _BAND_ALIASES[key]


def _rgb_bands(fits_paths, rgb):
    if rgb == "auto" or rgb is None:
        for bands in _AUTO_RGB:
            if all(band in fits_paths for band in bands):
                return bands
        raise RuntimeError("No usable Euclid RGB band combination found.")

    if isinstance(rgb, str):
        rgb = rgb.replace("/", ",").split(",")
    if len(rgb) != 3:
        raise ValueError("RGB must be 'auto' or three Euclid bands in R/G/B order.")
    bands = tuple(_band_name(band) for band in rgb)
    missing = [band for band in bands if band not in fits_paths]
    if missing:
        raise RuntimeError(f"Requested Euclid bands were not downloaded: {missing}")
    return bands


def Euclidimg(ra, dec, size, path, cred, output_jpg="Euclid_color.jpg", ReplaceL=True, RGB="auto"):
    """Download Euclid FITS cutouts and render one color JPEG."""

    from Download_Euclid import EUC_download

    outdir = Path(path)
    fits_paths = EUC_download(ra, dec, size, outdir, cred)
    jpg_path = outdir / output_jpg
    rgb_bands = _rgb_bands(fits_paths, RGB)
    mk_colorimg(
        [fits_paths[band] for band in rgb_bands],
        output_jpg=jpg_path,
        input_mode="raw",
        config=ComposeConfig(replace_luminance=bool(ReplaceL)),
    )
    return {"fits": fits_paths, "jpg": jpg_path, "rgb_bands": rgb_bands}
