from __future__ import annotations

from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astroquery.esa.euclid.core import EuclidClass


BANDS = ("VIS", "NIR_Y", "NIR_J", "NIR_H")


def _query(ra: float, dec: float) -> str:
    return (
        "SELECT file_name, file_path, filter_name, product_id "
        "FROM dr1.mosaic_product "
        "WHERE (product_type like '%Mer%Mosaic%') "
        "AND ((instrument_name='VIS' AND filter_name='VIS') "
        "OR (instrument_name='NISP' AND filter_name IN ('NIR_Y','NIR_J','NIR_H'))) "
        "AND category='SCIENCE' AND fov IS NOT NULL AND "
        f"INTERSECTS(CIRCLE('ICRS',{ra},{dec},0.0166667),fov)=1 "
        "ORDER BY product_id ASC"
    )


def _is_all_zero(path: Path) -> bool:
    with fits.open(path, memmap=False) as hdul:
        data = next(hdu.data for hdu in hdul if hdu.data is not None)
    return np.count_nonzero(np.nan_to_num(data, nan=0.0)) == 0


def EUC_download(ra, dec, size, path, cred):
    """
    Download Euclid DR1 VIS/Y/J/H FITS cutouts.

    Parameters
    ----------
    ra, dec : float
        Target coordinates in degrees.
    size : float
        Cutout radius in arcsec.
    path : str or Path
        Output directory.
    cred : str or Path
        Euclid credentials file.
    """

    outdir = Path(path)
    outdir.mkdir(parents=True, exist_ok=True)
    coord = SkyCoord(float(ra), float(dec), unit="deg")

    euclid = EuclidClass(environment="IDR")
    euclid.login(credentials_file=str(cred))

    rows = euclid.launch_job(_query(float(ra), float(dec)), verbose=False).get_results()
    outputs = {}

    for band in BANDS:
        for i, row in enumerate(rows[rows["filter_name"] == band], start=1):
            outfile = outdir / f"{band}_tile{i}.fits"
            euclid.get_cutout(
                file_path=f"{row['file_path']}/{row['file_name']}",
                instrument="None",
                id=str(row["product_id"]),
                coordinate=coord,
                radius=u.Quantity(float(size), u.arcsec),
                output_file=str(outfile),
            )
            if outfile.stat().st_size > 0 and not _is_all_zero(outfile):
                outputs[band] = outfile
                break
        if band not in outputs:
            raise RuntimeError(f"No non-zero {band} cutout found.")

    return outputs
