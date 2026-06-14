from __future__ import annotations

"""
Core processing functions for a PixInsight-like RGB composition pipeline.

These functions were written against PixInsight documentation and local PCL
source files available on this machine. The key references used are:

- C:\\Program Files\\PixInsight\\doc\\tools\\BackgroundNeutralization\\BackgroundNeutralization.html
- C:\\Program Files\\PixInsight\\doc\\tools\\ColorCalibration\\ColorCalibration.html
- C:\\Program Files\\PixInsight\\doc\\tools\\HistogramTransformation\\HistogramTransformation.html
- C:\\Program Files\\PixInsight\\doc\\tools\\ScreenTransferFunction\\ScreenTransferFunction.html
- C:\\Program Files\\PixInsight\\doc\\tools\\ChannelCombination\\ChannelCombination.html
- C:\\Program Files\\PixInsight\\doc\\tools\\ChannelExtraction\\ChannelExtraction.html
- C:\\Program Files\\PixInsight\\doc\\scripts\\CorrectMagentaStars\\CorrectMagentaStars.html
- C:\\Program Files\\PixInsight\\src\\pcl\\RGBColorSystem.cpp
- C:\\Program Files\\PixInsight\\include\\pcl\\RGBColorSystem.h

Design note:
- The current user-facing CLI is:
  raw mono R/G/B FITS -> shared linear normalization -> PI-like color pipeline -> final color JPEG.
- Pipeline summary:

  1. Shared linear input normalization

         s = 1 / max( max(R_raw), max(G_raw), max(B_raw) )
         R0 = clip(s * R_raw, 0, 1)
         G0 = clip(s * G_raw, 0, 1)
         B0 = clip(s * B_raw, 0, 1)

  2. ChannelCombination

         RGB0(x, y) = [ R0(x, y), G0(x, y), B0(x, y) ]

  3. BackgroundNeutralization

         b_ref = max(b_R, b_G, b_B)
         I'_c = I_c + (b_ref - b_c)
         I_BN = clip(s_BN * I', 0, 1)

  4. ColorCalibration

         s_c = w_c - b0
         g_c = min(s_R, s_G, s_B) / s_c
         I_CC,c = q * ( g_c * (I_BN,c - b0) + b0 )

  5. linked STF + HT transfer

         madn_c = 1.4826 * median(|I_c - median(I_c)|)
         c0 = clip(mean(med_c + shadows_clipping * madn_c), 0, 1)
         m  = x_bg * (1 - target_background) / (x_bg + target_background - 2*x_bg*target_background)
         I_STF = HT(I_CC; shadows=c0, highlights=1, midtones=m)

  6. Replace CIELab lightness with stretched blue

         (L, a, b) = RGB_to_Lab(I_STF)
         I_LB = Lab_to_RGB(B_stretched, a, b)

  7. Post-HT dimming

         I_DIM = MTF(0.75, I_LB)

  8. Invert + SCNR + Invert + SCNR

         I_1 = 1 - I_DIM
         G_neutral = min(G, (R + B)/2)
         G' = (1 - amount) * G + amount * G_neutral
         I_2 = 1 - SCNR(I_1)
         I_SCNR = SCNR(I_2)

  9. Final saturation boost

         (H, S, V, L) = RGB_to_HSVL(I_SCNR)
         S' = clip(S * (1 + saturation_amount), 0, 1)
         I_final = HSVL_to_RGB(H, S', V, L)

  In short:

      raw FITS
      -> shared linear scale
      -> BN
      -> CC
      -> linked STF/HT
      -> replace L* with B
      -> HT darken
      -> SCNR cleanup
      -> saturation
      -> final JPEG
- The background ROI search is custom glue code, not a PixInsight built-in.
- Classic ColorCalibration and BN are implemented in the empirical way we
  derived while matching PI outputs, not SPCC.
- SCNR follows the AverageNeutral interpretation plus preserved CIE L*,
  which is the behavior that matched the PI reference best in our tests.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from astropy.io import fits
from matplotlib import colors as mpl_colors
from PIL import Image
from skimage import color as skcolor


@dataclass(frozen=True)
class ComposeConfig:
    """
    Tunable parameters for the RGB pipeline.

    PI mapping:
    - `shadows_clipping` and `target_background` correspond to ScreenTransferFunction
      auto-stretch settings.
    - `bn_target_background` is the background pedestal applied after
      BackgroundNeutralization. A small positive pedestal is important for raw
      FITS images, because driving the neutralized background all the way to
      zero makes the subsequent linked STF far too aggressive.
    - `ht_dim_midtones` corresponds to a second HistogramTransformation where the
      midtones slider is moved to the right, here to 0.625.
    - `scnr1_amount` and `scnr2_amount` control the two SCNR passes.
    - `saturation_amount` is the final saturation boost.
    """

    shadows_clipping: float = -2.8
    target_background: float = 0.25
    bn_target_background: float = 0.03
    ht_dim_midtones: float = 0.625
    scnr1_amount: float = 0.50
    scnr2_amount: float = 0.80
    saturation_amount: float = 0.80


@dataclass(frozen=True)
class AsinhComposeConfig:
    """
    Tunable parameters for the Euclid-style asinh RGB pipeline.

    This mirrors `EuclidSLcolor_Tian.py`:
    - raw mono FITS channels are stretched directly, without a shared
      pre-normalization step
    - per-channel asinh stretch uses band-specific q values
    - optional luminosity replacement matches the original
      `make_rgb(..., mode="arcsinh")` behavior
    """

    q_red: float = 0.5
    q_green: float = 1.0
    q_blue: float = 500.0
    clip: float = 99.85
    use_luminosity: bool = True


@dataclass(frozen=True)
class BackgroundReference:
    """
    Background ROI and summary statistics used by BN/CC.

    PI mapping:
    - BackgroundNeutralization and ColorCalibration both require a background
      reference. PixInsight lets the user define this by view or ROI.
    - Here we store the ROI rectangle plus the per-channel medians derived from
      the selected background block.
    """

    rect: tuple[int, int, int, int]
    medians: np.ndarray
    avg_devs: np.ndarray
    high: float


SRGB_X = np.array([0.648431, 0.321152, 0.155886], dtype=np.float64)
SRGB_Y = np.array([0.330856, 0.597871, 0.066044], dtype=np.float64)
SRGB_LUMA = np.array([0.222491, 0.716888, 0.060621], dtype=np.float64)

CIE_EPSILON = 216.0 / 24389.0
CIE_KAPPA_116 = 24389.0 / 3132.0


def load_fits_image_raw(path: Path) -> np.ndarray:
    """
    Load a 2D FITS image as raw floating-point pixel values.

    PI mapping:
    - This is not a PI process module by itself; it is file I/O glue for the
      user-facing pipeline where the inputs are the original mono FITS frames.
    - Unlike the earlier prototype, this function does not apply any per-file
      min/max normalization. It preserves the raw sample values from the FITS
      image so we do not accidentally "pre-stretch" each channel on input.

    Current role in this project:
    - This is the intended input path for the command-line pipeline.
    - The three raw channel images are read here first, and only afterwards do
      we apply one shared linear normalization across all channels.
    """

    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None:
                data = np.asarray(hdu.data)
                break
        else:
            raise ValueError(f"No image data found in {path}")

    while data.ndim > 2:
        data = data[0]
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D image in {path}, got shape {data.shape}")

    return np.ascontiguousarray(data, dtype=np.float32)


def normalize_raw_channels_common(
    red: np.ndarray, green: np.ndarray, blue: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Apply one shared linear scale factor to three raw FITS channels.

    PI mapping:
    - This is preprocessing glue, not a PixInsight process.
    - The goal is to accept original mono FITS inputs without doing an
      independent stretch on each channel.
    - We use one common multiplicative scale for R/G/B, so relative channel
      amplitudes are preserved before BN/CC.

    Formula:

        s = 1 / max( max(R_raw), max(G_raw), max(B_raw) )

        R_lin = min( s * R_raw, 1 )
        G_lin = min( s * G_raw, 1 )
        B_lin = min( s * B_raw, 1 )

    Notes:
    - This is a linear normalization, not a nonlinear stretch like STF/HT.
    - Negative values are preserved after the shared scaling. This matters for
      raw FITS backgrounds: clipping negative sky fluctuations to 0 would distort
      the background distribution and can make linked STF far too aggressive.
    - Only the upper end is clipped to 1 to keep obviously saturated highlights
      bounded before the later PI-like processing steps.
    - We deliberately do not normalize each channel independently, because one
      common scale preserves the raw inter-channel amplitude ratios before
      BackgroundNeutralization and ColorCalibration rebalance colors later.
    """

    if red.shape != green.shape or red.shape != blue.shape:
        raise ValueError(f"Input image shapes do not match: {red.shape}, {green.shape}, {blue.shape}")

    peak = max(float(np.nanmax(red)), float(np.nanmax(green)), float(np.nanmax(blue)))
    if not np.isfinite(peak) or peak <= 0:
        zeros = np.zeros_like(red, dtype=np.float32)
        return zeros, zeros.copy(), zeros.copy(), 1.0

    scale = 1.0 / peak
    red_lin = np.minimum(red.astype(np.float32) * scale, 1.0)
    green_lin = np.minimum(green.astype(np.float32) * scale, 1.0)
    blue_lin = np.minimum(blue.astype(np.float32) * scale, 1.0)
    return red_lin, green_lin, blue_lin, float(scale)


def _asinh_channel(channel: np.ndarray, q: float, clip: float = 99.85) -> np.ndarray:
    """
    Stretch one normalized channel with an asinh mapping and rescale to [0, 1].

    The channel is treated as a display-preparation image rather than a raw flux
    map. This keeps the implementation compact and consistent with the existing
    Euclid-style asinh renders used elsewhere in the workspace.
    """

    x = np.nan_to_num(np.asarray(channel, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, 0.0, None)
    stretched = np.arcsinh(x * float(q))
    if clip < 100.0:
        stretched = np.clip(stretched, 0.0, np.percentile(stretched, clip))
    lo = float(np.min(stretched))
    hi = float(np.max(stretched))
    denom = hi - lo
    if denom <= 0:
        return np.zeros_like(stretched, dtype=np.float32)
    scaled = (stretched - lo) / denom
    return (np.asarray(255.0 * scaled, dtype=np.uint8).astype(np.float32)) / 255.0


def replace_luminosity_channel(
    rgb_image: np.ndarray, rgb_channel_for_luminosity: int = 2
) -> np.ndarray:
    """
    Replace the CIE Lab lightness channel with one RGB channel.

    This matches the helper used in `EuclidSLcolor_Tian.py`.
    """

    rgb = np.asarray(rgb_image)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image, got shape {rgb.shape}")
    if not 0 <= rgb_channel_for_luminosity <= 2:
        raise ValueError("rgb_channel_for_luminosity must be 0, 1, or 2")

    if rgb.dtype == np.uint8:
        rgb_float = rgb.astype(np.float32) / 255.0
    elif np.issubdtype(rgb.dtype, np.floating):
        if np.nanmax(rgb) <= 1.0 and np.nanmin(rgb) >= 0.0:
            rgb_float = np.clip(rgb, 0.0, 1.0).astype(np.float32)
        else:
            rgb_float = (np.clip(rgb, 0.0, 255.0) / 255.0).astype(np.float32)
    else:
        rgb_float = (np.clip(rgb, 0, 255).astype(np.float32) / 255.0)

    replaced = replace_lab_l_with_blue(rgb_float, rgb_float[..., rgb_channel_for_luminosity])
    return np.round(np.clip(replaced, 0.0, 1.0) * 255.0).astype(np.uint8)


def load_raster_image(path: Path) -> np.ndarray:
    """
    Load a single-channel image from FITS/TIFF/common raster formats.

    PI mapping:
    - Not a PI process. This is a legacy convenience helper for experiments and
      notebooks; it is not the main CLI input path anymore.
    - Integer rasters are scaled by their full dtype range; float rasters outside
      [0, 1] are min/max normalized.

    Important note:
    - The current command-line tool does not use this function for raw FITS
      inputs. The CLI reads raw FITS values with `load_fits_image_raw()` and
      then applies one shared linear scale to the three channels together.
    """

    suffix = path.suffix.lower()
    if suffix in {".fits", ".fit", ".fts"}:
        data = load_fits_image_raw(path)
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max <= data_min:
            return np.zeros_like(data, dtype=np.float32)
        return (data - data_min) / (data_max - data_min)

    if suffix in {".tif", ".tiff"}:
        data = tifffile.imread(path)
    else:
        data = np.asarray(Image.open(path))

    data = np.asarray(data)
    if data.ndim == 3:
        if data.shape[-1] >= 3:
            data = data[..., :3].mean(axis=2)
        else:
            data = data[..., 0]
    if data.ndim != 2:
        raise ValueError(f"Expected a single-channel image in {path}, got shape {data.shape}")

    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        data = data.astype(np.float32) / float(info.max)
    else:
        data = data.astype(np.float32)
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max > 1.0 or data_min < 0.0:
            if data_max <= data_min:
                return np.zeros_like(data, dtype=np.float32)
            data = (data - data_min) / (data_max - data_min)
    return np.clip(data, 0.0, 1.0)


def load_rgb_image(path: Path) -> np.ndarray:
    """
    Load a true RGB image and normalize it to floating point [0, 1].

    PI mapping:
    - Not a PixInsight process. This is a legacy convenience helper for
      experiments, notebooks, and ad hoc comparisons; it is not the current
      command-line entrypoint.
    - Integer rasters are scaled by the full dtype range, matching the usual
      image-IO convention for normalized processing.
    - Floating rasters are kept if already in [0, 1]; otherwise they are
      min/max normalized.

    Important note:
    - The current command-line tool expects three raw mono FITS channels, not a
      precombined RGB image. This function is retained because it is still handy
      for quick exploratory tests in Python.

    Accepted layouts:

        (H, W, 3)    standard raster RGB layout
        (3, H, W)    channel-first RGB cube, common in some scientific files

    Output:

        RGB_out(x, y, c) in [0, 1], with channels ordered as R, G, B.
    """

    suffix = path.suffix.lower()
    if suffix in {".fits", ".fit", ".fts"}:
        with fits.open(path) as hdul:
            for hdu in hdul:
                if hdu.data is not None:
                    data = np.asarray(hdu.data)
                    break
            else:
                raise ValueError(f"No image data found in {path}")
    elif suffix in {".tif", ".tiff"}:
        data = tifffile.imread(path)
    else:
        data = np.asarray(Image.open(path))

    data = np.asarray(data)
    while data.ndim > 3 and data.shape[0] == 1:
        data = data[0]

    if data.ndim != 3:
        raise ValueError(f"Expected an RGB image in {path}, got shape {data.shape}")

    if data.shape[-1] >= 3:
        data = data[..., :3]
    elif data.shape[0] == 3:
        data = np.moveaxis(data, 0, -1)
    else:
        raise ValueError(f"Expected 3 RGB channels in {path}, got shape {data.shape}")

    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        data = data.astype(np.float32) / float(info.max)
    else:
        data = data.astype(np.float32)
        data_min = float(np.min(data))
        data_max = float(np.max(data))
        if data_max > 1.0 or data_min < 0.0:
            if data_max <= data_min:
                return np.zeros_like(data, dtype=np.float32)
            data = (data - data_min) / (data_max - data_min)

    return np.clip(data, 0.0, 1.0)


def save_tiff16(path: Path, image: np.ndarray) -> None:
    """
    Save an image as 16-bit TIFF.

    PI mapping:
    - Not a PI process, but this is the export format we want because many tools
      read 16-bit TIFF more reliably than floating TIFF in everyday workflows.
    - Pixel values are clipped to [0, 1] and quantized to uint16.
    - We export with the display orientation corresponding to
      `origin='lower'`, which means we vertically flip the NumPy array before
      writing the raster file so the saved TIFF looks like a FITS image shown
      with lower origin.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.flip(np.clip(image, 0.0, 1.0), axis=0)
    image16 = np.round(clipped * 65535.0).astype(np.uint16)
    tifffile.imwrite(path, image16, photometric="rgb" if image.ndim == 3 else "minisblack")


def save_jpeg(path: Path, image: np.ndarray, quality: int = 100) -> None:
    """
    Save an RGB image as a high-quality JPEG.

    PI mapping:
    - Not a PI process. This is just the final export layer for the user-facing
      script.
    - Pixel values are clipped to [0, 1], quantized to 8-bit, and written with
      `quality=100` and `subsampling=0` to minimize JPEG chroma loss.
    - We also request `keep_rgb=True` when Pillow supports it, because that
      keeps the encoder in RGB space instead of applying an extra YCbCr color
      conversion. In practice this makes the JPEG look closer to the TIFF.
    - We export with the display orientation corresponding to
      `origin='lower'`, so the saved JPEG matches the usual FITS display
      convention by vertically flipping the array before raster export.

    Quantization formula:

        J_8bit = round( 255 * clip(I, 0, 1) )

    Notes:
    - JPEG is inherently 8-bit and lossy, so this export is for viewing and
      sharing, not for machine-level numeric comparisons.
    """

    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"JPEG export expects an RGB image, got shape {image.shape}")

    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.flip(np.clip(image, 0.0, 1.0), axis=0)
    image8 = np.round(clipped * 255.0).astype(np.uint8)
    image_pil = Image.fromarray(image8, mode="RGB")
    try:
        image_pil.save(path, format="JPEG", quality=quality, subsampling=0, keep_rgb=True)
    except TypeError:
        image_pil.save(path, format="JPEG", quality=quality, subsampling=0)


def channel_avg_dev(values: np.ndarray) -> float:
    """
    Compute the average absolute deviation from the median.

    PI mapping:
    - PixInsight exposes `avgDev()` as a robust dispersion statistic.
    - We use it only in our custom ROI-selection heuristic; this is not a direct
      reimplementation of a PI module.
    """

    median = np.median(values)
    return float(np.mean(np.abs(values - median)))


def mean_channel_stats(image: np.ndarray, rect: tuple[int, int, int, int]) -> tuple[float, float]:
    """
    Average per-channel median and average deviation over an ROI.

    PI mapping:
    - Custom helper used to score candidate background blocks.
    - This mirrors the same robust statistics we queried from PI while building
      the reference workflow.
    """

    sub = roi_slice(image, rect)
    if sub.ndim == 2:
        sub = sub[..., None]
    medians = []
    avg_devs = []
    for c in range(sub.shape[2]):
        medians.append(float(np.median(sub[..., c])))
        avg_devs.append(channel_avg_dev(sub[..., c]))
    return float(np.mean(medians)), float(np.mean(avg_devs))


def per_channel_stats(image: np.ndarray, rect: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Return per-channel median and average deviation for an ROI.

    PI mapping:
    - Custom helper that stores the channel-wise background summary needed later
      by BackgroundNeutralization and ColorCalibration.
    """

    sub = roi_slice(image, rect)
    if sub.ndim == 2:
        sub = sub[..., None]
    medians = []
    avg_devs = []
    for c in range(sub.shape[2]):
        medians.append(float(np.median(sub[..., c])))
        avg_devs.append(channel_avg_dev(sub[..., c]))
    return np.array(medians, dtype=np.float64), np.array(avg_devs, dtype=np.float64)


def roi_slice(image: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    """
    Slice an image by `(x0, y0, x1, y1)`.

    PI mapping:
    - Utility glue for ROI-based operations. PI tools often use ROI rectangles in
      this same style.
    """

    x0, y0, x1, y1 = rect
    return image[y0:y1, x0:x1]


def find_background_reference(image: np.ndarray) -> BackgroundReference:
    """
    Select a dark, flat ROI to act as the background reference.

    PI mapping:
    - This is custom logic, not a built-in PI module.
    - The goal is to automate the ROI that would normally be supplied to
      BackgroundNeutralization and ColorCalibration.
    - We scan a 6x6 grid, avoid the center, and prefer blocks with low median and
      low average deviation.
    """

    height, width = image.shape[:2]
    roi_w = min(width, max(8, int(round(width * 0.22))))
    roi_h = min(height, max(8, int(round(height * 0.22))))
    best_rect = None
    best_score = None

    for iy in range(6):
        for ix in range(6):
            x0 = int(round(ix * max(0, width - roi_w) / 5))
            y0 = int(round(iy * max(0, height - roi_h) / 5))
            rect = (x0, y0, x0 + roi_w, y0 + roi_h)
            cx = x0 + 0.5 * roi_w
            cy = y0 + 0.5 * roi_h
            dx = (cx - 0.5 * width) / max(1, width)
            dy = (cy - 0.5 * height) / max(1, height)
            if float(np.hypot(dx, dy)) < 0.12:
                continue
            median, avg_dev = mean_channel_stats(image, rect)
            score = median + 2.0 * avg_dev
            if best_score is None or score < best_score:
                best_score = score
                best_rect = rect

    if best_rect is None:
        best_rect = (0, 0, roi_w, roi_h)

    medians, avg_devs = per_channel_stats(image, best_rect)
    median_mean = float(np.mean(medians))
    avg_dev_mean = float(np.mean(avg_devs))
    high = max(0.02, min(0.25, median_mean + 3.0 * avg_dev_mean))
    return BackgroundReference(rect=best_rect, medians=medians, avg_devs=avg_devs, high=high)


def background_neutralization(
    rgb: np.ndarray, background: BackgroundReference, target_background: float = 0.03
) -> tuple[np.ndarray, float]:
    """
    Equalize the RGB background components with a shared rescale if needed.

    PI mapping:
    - Corresponds to BackgroundNeutralization in `RescaleAsNeeded` mode.
    - PI documentation describes this process as equalizing the average red,
      green, and blue background components through per-channel linear transforms.
    - Our current empirical match uses a small positive neutral target
      background pedestal, then applies a shared scale if any channel would
      exceed 1.

    Working formula used here:

    Let b_c be the per-channel background median on the selected ROI and let
    b_ref be a small positive target pedestal, by default 0.03.

    First shift each channel toward the same background level:

        I'_c(x, y) = I_c(x, y) + (b_ref - b_c)

    Then, in the spirit of PI's `RescaleAsNeeded`, apply a common scale if the
    shifted image would overflow:

        s = 1 / max(I')   if max(I') > 1
        s = 1             otherwise

        I''_c(x, y) = s * I'_c(x, y)

    So the final output is:

        I_out = min(I'', 1)

    We intentionally do not clip the lower end here. Preserving negative or
    slightly sub-pedestal fluctuations produces a much more PI-like linked STF
    response on raw astronomical images.
    """

    shifted = rgb + (float(target_background) - background.medians.reshape(1, 1, 3))
    peak = float(np.max(shifted))
    scale = 1.0 / peak if peak > 1.0 else 1.0
    out = shifted * scale
    return np.minimum(out, 1.0), float(scale)


def color_calibration(rgb: np.ndarray, background: BackgroundReference, bn_scale: float) -> np.ndarray:
    """
    Apply classic ColorCalibration using the whole image as white reference.

    PI mapping:
    - Corresponds to ColorCalibration with a whole-image white reference and the
      selected ROI as background reference.
    - PI documentation states that channel factors are computed from the white
      reference and applied by multiplication.
    - Our empirical match keeps the background level fixed, scales each channel
      by the minimum white-reference signal, and uses the inverse BN scale as the
      global normalization factor.

    Working formula used here:

    Let b0 be the median background level on the background ROI after BN.
    Let w_c be the mean of channel c on the white-reference mask.
    Define the signal above background:

        s_c = w_c - b0

    Choose gains relative to the smallest channel signal so we only scale down:

        g_c = min(s_R, s_G, s_B) / s_c

    Then apply the channel calibration around the background pedestal:

        I'_c(x, y) = g_c * ( I_c(x, y) - b0 ) + b0

    Finally apply the global scale we empirically found to match PI:

        q = 1 / s_BN

        I_out,c(x, y) = q * I'_c(x, y)
    """

    bg_roi = roi_slice(rgb, background.rect)
    bg_level = float(np.median(bg_roi.reshape(-1, 3), axis=0)[0])
    white_mask = (rgb > 0.0).all(axis=2) & (rgb < 0.98).all(axis=2)
    white_mean = rgb[white_mask].mean(axis=0)
    signal = white_mean - bg_level
    gains = np.min(signal) / signal
    global_scale = 1.0 / bn_scale
    out = global_scale * (gains.reshape(1, 1, 3) * (rgb - bg_level) + bg_level)
    return np.clip(out, 0.0, 1.0)


def madn(channel: np.ndarray) -> float:
    """
    Robust sigma estimate from the median absolute deviation.

    PI mapping:
    - STF auto-stretch in PI is driven by median plus/minus a multiple of the
      robust deviation. In scripts this is commonly expressed as `MAD * 1.4826`.

    Formula:

        MAD = median( |x - median(x)| )
        MADN = 1.4826 * MAD

    The factor 1.4826 converts MAD to a Gaussian-equivalent sigma estimate.
    """

    med = np.median(channel)
    return float(np.median(np.abs(channel - med)) * 1.4826)


def mtf(m: float, x: np.ndarray) -> np.ndarray:
    """
    Evaluate PixInsight's midtones transfer function.

    PI mapping:
    - HistogramTransformation documentation defines the MTF as a rational
      function through (0,0), (m,1/2), and (1,1).
    - This is the core nonlinearity used by both STF display stretches and HT.

    Main formula for 0 < x < 1 and 0 < m < 1:

        MTF(m, x) = ((m - 1) * x) / (((2m - 1) * x) - m)

    Boundary behavior:

        x <= 0  -> 0
        x >= 1  -> 1

    Special cases:

        m = 0.5 -> identity, so MTF(0.5, x) = x
        m <= 0  -> 1 on the open interval (0, 1)
        m >= 1  -> 0 on the open interval (0, 1)

    Interpretation:

        m < 0.5  brightens midtones
        m > 0.5  darkens midtones
    """

    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    out[x <= 0] = 0.0
    out[x >= 1] = 1.0
    mid = (x > 0) & (x < 1)
    xm = x[mid]
    if abs(m - 0.5) < 1.0e-15:
        out[mid] = xm
    elif m <= 0:
        out[mid] = 1.0
    elif m >= 1:
        out[mid] = 0.0
    else:
        out[mid] = ((m - 1.0) * xm) / (((2.0 * m) - 1.0) * xm - m)
    return out


def inverse_mtf(y: float, x: float) -> float:
    """
    Solve for the midtones parameter that maps `x` to `y`.

    PI mapping:
    - STF auto-stretch chooses a target background and solves the MTF parameter
      needed to send the estimated background there.

    Inverting the MTF relation gives:

        m = x * (1 - y) / (x + y - 2xy)

    where:

        x = the current background estimate after shadows clipping
        y = the desired target background

    In our STF-style usage:

        x = mean(median_c) - c0
        y = target_background
    """

    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    denom = x + y - (2.0 * x * y)
    if abs(denom) < 1.0e-15:
        return 0.5
    return float((x * (1.0 - y)) / denom)


def histogram_transform(
    image: np.ndarray,
    shadows: float,
    highlights: float,
    midtones: float,
    range_low: float = 0.0,
    range_high: float = 1.0,
) -> np.ndarray:
    """
    Apply a PI-style histogram transformation.

    PI mapping:
    - HistogramTransformation first remaps the selected input interval into
      [0, 1], clips shadows/highlights, and then applies the MTF.
    - We keep the same parameter order used by PI's scripting interface:
      shadows, highlights, midtones, range low, range high.

    Working formula:

    1. Dynamic-range normalization:

        u = (I - range_low) / (range_high - range_low)

    2. Shadows/highlights remap:

        v = (u - shadows) / (highlights - shadows)

    3. Clamp to the unit interval:

        v_clamped = clip(v, 0, 1)

    4. Apply the midtones transfer function:

        I_out = MTF(midtones, v_clamped)

    So, compactly:

        I_out = MTF(midtones, clip(( (I - range_low)/(range_high-range_low) - shadows )
                                   /(highlights-shadows), 0, 1))
    """

    x = (image - range_low) / (range_high - range_low)
    x = np.clip(x, 0.0, 1.0)
    x = (x - shadows) / (highlights - shadows)
    x = np.clip(x, 0.0, 1.0)
    return np.clip(mtf(midtones, x), 0.0, 1.0)


def linked_auto_stf_and_ht(rgb: np.ndarray, config: ComposeConfig) -> np.ndarray:
    """
    Apply linked STF-style auto stretch and transfer it permanently.

    PI mapping:
    - ScreenTransferFunction documentation describes auto stretch in terms of a
      shadows clipping value in sigma units and a target background.
    - With RGB linked, all three channels share the same STF, preserving the
      color ratios established by color calibration.
    - We then transfer the resulting parameters into HistogramTransformation,
      which is how PI makes the screen stretch permanent.

    Working formula used here:

    For each channel c:

        med_c  = median(channel_c)
        madn_c = 1.4826 * median(|channel_c - med_c|)

    Linked shadows clipping point:

        c0 = clip( mean( med_c + shadows_clipping * madn_c ), 0, 1 )

    Let the clipped-background estimate be:

        x_bg = mean(med_c) - c0

    Solve the MTF parameter that maps x_bg to the desired target background:

        m = x_bg * (1 - target_background) /
            (x_bg + target_background - 2*x_bg*target_background)

    Then apply one shared HT to all three channels:

        I_out = HT(I_in; shadows=c0, highlights=1, midtones=m)
    """

    med = np.median(rgb.reshape(-1, 3), axis=0)
    mad = np.array([madn(rgb[..., c]) for c in range(3)], dtype=np.float64)
    c0 = float(np.clip(np.mean(med + config.shadows_clipping * mad), 0.0, 1.0))
    m = inverse_mtf(config.target_background, float(np.mean(med) - c0))
    return histogram_transform(rgb, shadows=c0, highlights=1.0, midtones=m)


def apply_post_auto_ht_darkening(rgb: np.ndarray, config: ComposeConfig) -> np.ndarray:
    """
    Apply the second HT dimming pass after the main auto stretch.

    PI mapping:
    - This is the extra HistogramTransformation pass you described as moving the
      central HT slider to the 75% position on the right.
    - In PI terms this means `midtones = 0.75`, with shadows/highlights kept at
      their identity values.

    Formula:

        I_out = MTF(0.75, I_in)

    because:

        shadows = 0
        highlights = 1
        range_low = 0
        range_high = 1

    so the surrounding histogram remap becomes the identity and only the MTF
    remains.
    """

    return histogram_transform(rgb, shadows=0.0, highlights=1.0, midtones=config.ht_dim_midtones)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """
    Convert sRGB-encoded values to linear RGB.

    This helper is retained for compatibility with older notebooks. The active
    RGB/XYZ/Lab path below delegates color conversion to `skimage.color`.
    """

    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4))


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """
    Convert linear RGB values back to the sRGB transfer curve.

    This helper is retained for compatibility with older notebooks. The active
    RGB/XYZ/Lab path below delegates color conversion to `skimage.color`.
    """

    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def setup_rgb_to_xyz_matrix(x: np.ndarray, y: np.ndarray, y_luma: np.ndarray) -> np.ndarray:
    """
    Build the RGB -> XYZ matrix from working-space chromaticities and luma.

    PI mapping:
    - This follows the same matrix construction used in `RGBColorSystem.cpp`
      when a working space is initialized from D50 chromaticities and luminance
      coefficients.
    """

    return np.array(
        [
            [y_luma[0] * x[0] / y[0], y_luma[1] * x[1] / y[1], y_luma[2] * x[2] / y[2]],
            [y_luma[0], y_luma[1], y_luma[2]],
            [
                y_luma[0] * (1.0 - x[0] - y[0]) / y[0],
                y_luma[1] * (1.0 - x[1] - y[1]) / y[1],
                y_luma[2] * (1.0 - x[2] - y[2]) / y[2],
            ],
        ],
        dtype=np.float64,
    )


RGB_TO_XYZ = setup_rgb_to_xyz_matrix(SRGB_X, SRGB_Y, SRGB_LUMA)
XYZ_TO_RGB = np.linalg.inv(RGB_TO_XYZ)
MX = float(np.sum(RGB_TO_XYZ[0]))
MZ = float(np.sum(RGB_TO_XYZ[2]))

_X = 0.0
_Y = 1.0
_a_min = 5.0 * (
    (np.cbrt(_X) if _X > CIE_EPSILON else CIE_KAPPA_116 * _X + 16.0 / 116.0)
    - (np.cbrt(_Y) if _Y > CIE_EPSILON else CIE_KAPPA_116 * _Y + 16.0 / 116.0)
)
_X = 1.0
_Y = 0.0
_a_max = 5.0 * (
    (np.cbrt(_X) if _X > CIE_EPSILON else CIE_KAPPA_116 * _X + 16.0 / 116.0)
    - (np.cbrt(_Y) if _Y > CIE_EPSILON else CIE_KAPPA_116 * _Y + 16.0 / 116.0)
)
_Y = 0.0
_Z = 1.0
_b_min = 2.0 * (
    (np.cbrt(_Y) if _Y > CIE_EPSILON else CIE_KAPPA_116 * _Y + 16.0 / 116.0)
    - (np.cbrt(_Z) if _Z > CIE_EPSILON else CIE_KAPPA_116 * _Z + 16.0 / 116.0)
)
_Y = 1.0
_Z = 0.0
_b_max = 2.0 * (
    (np.cbrt(_Y) if _Y > CIE_EPSILON else CIE_KAPPA_116 * _Y + 16.0 / 116.0)
    - (np.cbrt(_Z) if _Z > CIE_EPSILON else CIE_KAPPA_116 * _Z + 16.0 / 116.0)
)
ZA = -_a_min
ZB = -_b_min
MA = _a_max - _a_min
MB = _b_max - _b_min


def xyz_lab_component(x: np.ndarray) -> np.ndarray:
    """
    Forward XYZ -> Lab component nonlinearity.

    PI mapping:
    - Matches `RGBColorSystem::XYZLab()` in PCL.
    """

    return np.where(x > CIE_EPSILON, np.cbrt(x), CIE_KAPPA_116 * x + 16.0 / 116.0)


def lab_xyz_component(x: np.ndarray) -> np.ndarray:
    """
    Inverse Lab -> XYZ component nonlinearity.

    PI mapping:
    - Matches `RGBColorSystem::LabXYZ()` in PCL.
    """

    x3 = x * x * x
    return np.where(x3 > CIE_EPSILON, x3, (x - 16.0 / 116.0) / CIE_KAPPA_116)


def rgb_to_xyz(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert RGB values to CIE XYZ with `skimage.color`.
    """

    xyz = skcolor.rgb2xyz(np.clip(rgb, 0.0, 1.0))
    return xyz[..., 0], xyz[..., 1], xyz[..., 2]


def xyz_to_rgb(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Convert CIE XYZ values back to RGB with `skimage.color`.
    """

    xyz = np.stack([x, y, z], axis=-1)
    return np.clip(skcolor.xyz2rgb(xyz), 0.0, 1.0)


def rgb_to_lab(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert RGB to normalized CIE L*, a*, b* using `skimage.color`.

    The public convention of this module is kept unchanged: L, a, and b are
    returned in `[0, 1]`, matching the older PCL-style normalization used by
    downstream functions.
    """

    lab = skcolor.rgb2lab(np.clip(rgb, 0.0, 1.0))
    l = np.clip(lab[..., 0] / 100.0, 0.0, 1.0)
    a = np.clip(((lab[..., 1] / 100.0) + ZA) / MA, 0.0, 1.0)
    b = np.clip(((lab[..., 2] / 100.0) + ZB) / MB, 0.0, 1.0)
    return l, a, b


def lab_to_rgb(l: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Convert normalized CIE L*, a*, b* back to RGB using `skimage.color`.
    """

    lab = np.stack(
        [
            np.clip(l, 0.0, 1.0) * 100.0,
            100.0 * ((MA * np.clip(a, 0.0, 1.0)) - ZA),
            100.0 * ((MB * np.clip(b, 0.0, 1.0)) - ZB),
        ],
        axis=-1,
    )
    return np.clip(skcolor.lab2rgb(lab), 0.0, 1.0)


def replace_lab_l_with_blue(rgb: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """
    Replace the CIELab L* channel with the stretched blue channel.

    PI mapping:
    - Equivalent to extracting CIELab, swapping L with B, and recombining.
    - This preserves the original chrominance (a*, b*) while forcing the image
      lightness structure to follow the blue channel.

    Formula:

        (L, a, b) = RGB_to_Lab(RGB)
        L' = B_stretched
        RGB_out = Lab_to_RGB(L', a, b)
    """

    _, a, b = rgb_to_lab(rgb)
    return lab_to_rgb(blue, a, b)


def rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert RGB to HSV with `matplotlib.colors`.
    """

    hsv = mpl_colors.rgb_to_hsv(np.clip(rgb, 0.0, 1.0))
    return hsv[..., 0], hsv[..., 1], hsv[..., 2]


def hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Convert HSV back to RGB with `matplotlib.colors`.
    """

    hsv = np.stack([np.mod(h, 1.0), np.clip(s, 0.0, 1.0), np.clip(v, 0.0, 1.0)], axis=-1)
    return mpl_colors.hsv_to_rgb(hsv)


def rgb_to_hsvl(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert RGB to HSV plus preserved CIE L*.

    PI mapping:
    - Matches the idea of PCL's `RGBToHSVL()`: HSV channels plus the CIE L*
      component from the current RGB working space.
    """

    h, s, v = rgb_to_hsv(rgb)
    l, _, _ = rgb_to_lab(rgb)
    return h, s, v, l


def hsvl_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray, l: np.ndarray) -> np.ndarray:
    """
    Convert HSVL back to RGB.

    PI mapping:
    - Matches the logic of `HSVLToRGB()` in `RGBColorSystem.h`: rebuild RGB from
      HSV, then restore the desired CIE L* component.
    """

    rgb = hsv_to_rgb(h, s, v)
    _, a, b = rgb_to_lab(rgb)
    return lab_to_rgb(l, a, b)


def invert(rgb: np.ndarray) -> np.ndarray:
    """
    Apply an invert transform, `1 - x`.

    PI mapping:
    - Corresponds directly to PixInsight's Invert process.
    - Used here to turn magenta excess into green excess before SCNR.
    """

    return 1.0 - rgb


def scnr_average_neutral(rgb: np.ndarray, amount: float) -> np.ndarray:
    """
    Remove green using AverageNeutral SCNR semantics, then preserve CIE L*.

    PI mapping:
    - Corresponds to SCNR with `colorToRemove = Green` and
      `protectionMethod = AverageNeutral`.
    - The core neutral estimate is `(R + B) / 2`, so excess green is pushed back
      toward that neutral value.
    - CorrectMagentaStars uses the pattern `Invert -> SCNR Green -> Invert`.
    - Our best PI match is: partial AverageNeutral reduction plus restoration of
      the original CIE L* component.

    Core AverageNeutral formula:

        G_neutral = min(G, (R + B)/2)

    With partial strength `amount`:

        G' = G + amount * (G_neutral - G)

    or equivalently:

        G' = (1 - amount) * G + amount * G_neutral

    Then we preserve the original CIE lightness:

        (L0, _, _) = RGB_to_Lab(RGB_in)
        (_, a1, b1) = RGB_to_Lab(R, G', B)
        RGB_out = Lab_to_RGB(L0, a1, b1)
    """

    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    neutral_g = np.minimum(g, 0.5 * (r + b))
    g_out = g + amount * (neutral_g - g)
    out = np.stack([r, g_out, b], axis=-1)
    l0, _, _ = rgb_to_lab(rgb)
    _, a1, b1 = rgb_to_lab(out)
    return lab_to_rgb(l0, a1, b1)


def color_saturation_hsvl(rgb: np.ndarray, amount: float) -> np.ndarray:
    """
    Apply a global saturation boost while preserving CIE L*.

    PI mapping:
    - This is our Python analogue of the final ColorSaturation step.
    - The implementation is closest to an HSVL interpretation: scale the
      saturation channel, then restore the original CIE L* on reconstruction.

    Formula:

        (H, S, V, L) = RGB_to_HSVL(RGB)
        S' = clip(S * (1 + amount), 0, 1)
        RGB_out = HSVL_to_RGB(H, S', V, L)

    So for `amount = 0.40`:

        S' = clip(1.4 * S, 0, 1)
    """

    h, s, v, l = rgb_to_hsvl(rgb)
    s_out = np.clip(s * (1.0 + amount), 0.0, 1.0)
    return np.clip(hsvl_to_rgb(h, s_out, v, l), 0.0, 1.0)


def compose_pipeline(red: np.ndarray, green: np.ndarray, blue: np.ndarray, config: ComposeConfig | None = None) -> dict[str, np.ndarray]:
    """
    Run the full RGB processing pipeline and return all named outputs.

    PI mapping:
    - ChannelCombination: stack `red`, `green`, `blue` into RGB.
    - BackgroundNeutralization
    - ColorCalibration
    - linked STF + HT transfer
    - replace Lab L* with stretched blue
    - post-HT dimming (`midtones = 0.75`)
    - invert / SCNR / invert / SCNR
    - final ColorSaturation boost
    """

    if config is None:
        config = ComposeConfig()

    if red.shape != green.shape or red.shape != blue.shape:
        raise ValueError(f"Input image shapes do not match: {red.shape}, {green.shape}, {blue.shape}")

    rgb = np.stack([red, green, blue], axis=-1)
    background = find_background_reference(rgb)
    bn, bn_scale = background_neutralization(rgb, background, target_background=config.bn_target_background)
    cc = color_calibration(bn, background, bn_scale)
    stfht = linked_auto_stf_and_ht(cc, config)
    blue_channel = stfht[..., 2]
    lab_l, lab_a, lab_b = rgb_to_lab(stfht)
    lfromb = replace_lab_l_with_blue(stfht, blue_channel)
    htdim = apply_post_auto_ht_darkening(lfromb, config)
    invert1 = invert(htdim)
    scnr1 = scnr_average_neutral(invert1, config.scnr1_amount)
    invert2 = invert(scnr1)
    scnr2 = scnr_average_neutral(invert2, config.scnr2_amount)
    final = color_saturation_hsvl(scnr2, config.saturation_amount)
    return {
        "01_rgb.tif": rgb,
        "02_bn.tif": bn,
        "03_cc.tif": cc,
        "04_stfht.tif": stfht,
        "05_blue.tif": blue_channel,
        "06_lab_L.tif": lab_l,
        "06_lab_a.tif": lab_a,
        "06_lab_b.tif": lab_b,
        "07_lfromb.tif": lfromb,
        "08_htdim.tif": htdim,
        "09_invert1.tif": invert1,
        "10_scnr1.tif": scnr1,
        "11_invert2.tif": invert2,
        "12_scnr2.tif": scnr2,
        "13_final.tif": final,
    }


def compose_asinh_pipeline(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    config: AsinhComposeConfig | None = None,
) -> dict[str, np.ndarray]:
    """
    Run the Euclid-style asinh RGB pipeline and return display-ready images.

    This path mirrors the original `EuclidSLcolor_Tian.py` behavior rather
    than the PixInsight-mapped MTF pipeline:
    raw mono channels -> per-band asinh stretch -> RGB stack ->
    optional CIE Lab luminosity replacement.
    """

    if config is None:
        config = AsinhComposeConfig()

    if red.shape != green.shape or red.shape != blue.shape:
        raise ValueError(f"Input image shapes do not match: {red.shape}, {green.shape}, {blue.shape}")

    r = _asinh_channel(red, config.q_red, clip=config.clip)
    g = _asinh_channel(green, config.q_green, clip=config.clip)
    b = _asinh_channel(blue, config.q_blue, clip=config.clip)

    rgb_u8 = np.stack([r, g, b], axis=-1)
    if config.use_luminosity:
        rgb_u8 = replace_luminosity_channel(rgb_u8, rgb_channel_for_luminosity=2)

    rgb = rgb_u8.astype(np.float32) / 255.0
    return {
        "01_r.tif": r.astype(np.float32),
        "02_g.tif": g.astype(np.float32),
        "03_b.tif": b.astype(np.float32),
        "04_rgb.tif": np.stack([r, g, b], axis=-1).astype(np.float32),
        "05_rgb_lum.tif": rgb,
        "13_final.tif": rgb,
    }


FITS_SUFFIXES = {".fits", ".fit", ".fts"}


class TianColorMaker:
    """
    Small convenience wrapper around the local MTF workflow.

    Supported inputs for `rgb_image`:
    - one RGB NumPy array with shape `(H, W, 3)`
    - one RGB raster path
    - three mono channels in `(R, G, B)` order as arrays or file paths

    Only the final JPEG is written. No intermediate TIFF products are saved.
    """

    def __init__(self, config: ComposeConfig | None = None):
        self.config = config or ComposeConfig()

    def render(
        self,
        rgb_image: np.ndarray | str | Path | list[np.ndarray | str | Path] | tuple[np.ndarray | str | Path, ...],
        output_jpg: str | Path = "mtf_color.jpg",
        *,
        input_mode: str = "auto",
        jpeg_quality: int = 100,
    ) -> Path:
        red, green, blue = _split_input_rgb_channels(rgb_image)
        mode = _resolve_input_mode(red, green, blue, input_mode)

        if mode == "raw":
            red_work, green_work, blue_work, _ = normalize_raw_channels_common(red, green, blue)
        else:
            red_work = np.clip(np.asarray(red, dtype=np.float32), 0.0, 1.0)
            green_work = np.clip(np.asarray(green, dtype=np.float32), 0.0, 1.0)
            blue_work = np.clip(np.asarray(blue, dtype=np.float32), 0.0, 1.0)

        outputs = compose_pipeline(red_work, green_work, blue_work, config=self.config)
        output_path = Path(output_jpg).expanduser().resolve()
        save_jpeg(output_path, outputs["13_final.tif"], quality=jpeg_quality)
        return output_path


def _split_input_rgb_channels(
    rgb_image: np.ndarray | str | Path | list[np.ndarray | str | Path] | tuple[np.ndarray | str | Path, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize the supported user inputs into three 2D channel arrays.
    """

    if isinstance(rgb_image, np.ndarray):
        return _split_rgb_array(rgb_image)

    if isinstance(rgb_image, (str, Path)):
        raster = np.asarray(Image.open(Path(rgb_image)))
        return _split_rgb_array(raster)

    if len(rgb_image) != 3:
        raise ValueError("rgb_image sequence must contain exactly three channels in R/G/B order.")

    red = _load_single_channel(rgb_image[0])
    green = _load_single_channel(rgb_image[1])
    blue = _load_single_channel(rgb_image[2])
    _validate_matching_shapes(red, green, blue)
    return red, green, blue


def _split_rgb_array(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split one RGB array into three float32 channels.
    """

    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(f"Expected an RGB image with shape (H, W, 3), got {array.shape}")

    array = array[..., :3]
    if np.issubdtype(array.dtype, np.integer):
        max_value = float(np.iinfo(array.dtype).max)
        array = array.astype(np.float32) / max_value
    else:
        array = array.astype(np.float32)

    red = array[..., 0]
    green = array[..., 1]
    blue = array[..., 2]
    _validate_matching_shapes(red, green, blue)
    return red, green, blue


def _load_single_channel(source: np.ndarray | str | Path) -> np.ndarray:
    """
    Load one mono channel from an array, FITS file, or single-channel raster.
    """

    if isinstance(source, np.ndarray):
        array = np.asarray(source, dtype=np.float32)
    else:
        path = Path(source)
        if path.suffix.lower() in FITS_SUFFIXES:
            array = load_fits_image_raw(path)
        else:
            raster = np.asarray(Image.open(path))
            if raster.ndim != 2:
                raise ValueError(
                    f"Single-channel raster expected for {path}, got shape {raster.shape}. "
                    "Pass RGB rasters as the top-level rgb_image input."
                )
            if np.issubdtype(raster.dtype, np.integer):
                max_value = float(np.iinfo(raster.dtype).max)
                array = raster.astype(np.float32) / max_value
            else:
                array = raster.astype(np.float32)

    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D mono channel, got shape {array.shape}")
    return np.ascontiguousarray(array, dtype=np.float32)


def _validate_matching_shapes(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> None:
    """
    Ensure all three channels have the same image shape.
    """

    if red.shape != green.shape or red.shape != blue.shape:
        raise ValueError(
            f"Input channel shapes do not match: red={red.shape}, green={green.shape}, blue={blue.shape}"
        )


def _resolve_input_mode(red: np.ndarray, green: np.ndarray, blue: np.ndarray, input_mode: str) -> str:
    """
    Decide whether the input should be treated as raw science data or as
    already-normalized display RGB.
    """

    if input_mode not in {"auto", "raw", "normalized"}:
        raise ValueError("input_mode must be one of: 'auto', 'raw', 'normalized'")

    if input_mode != "auto":
        return input_mode

    peak = max(float(np.nanmax(red)), float(np.nanmax(green)), float(np.nanmax(blue)))
    floor = min(float(np.nanmin(red)), float(np.nanmin(green)), float(np.nanmin(blue)))
    if floor < 0.0 or peak > 1.0:
        return "raw"
    return "normalized"


def mk_colorimg(
    rgb_image: np.ndarray | str | Path | list[np.ndarray | str | Path] | tuple[np.ndarray | str | Path, ...],
    output_jpg: str | Path = "mtf_color.jpg",
    *,
    input_mode: str = "auto",
    config: ComposeConfig | None = None,
    jpeg_quality: int = 100,
) -> Path:
    """
    Public convenience function for building one MTF color JPEG.

    Parameters
    ----------
    rgb_image
        One RGB image or three mono channels in `(R, G, B)` order.
    output_jpg
        Path of the final JPEG to write.
    input_mode
        - `"raw"`: apply shared linear normalization before the MTF workflow
        - `"normalized"`: assume inputs are already in display-space `[0, 1]`
        - `"auto"`: treat data outside `[0, 1]` as raw
    config
        Optional `ComposeConfig` for the MTF workflow.
    jpeg_quality
        JPEG export quality. Defaults to `100`.
    """

    maker = TianColorMaker(config=config)
    return maker.render(
        rgb_image,
        output_jpg=output_jpg,
        input_mode=input_mode,
        jpeg_quality=jpeg_quality,
    )
