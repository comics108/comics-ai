"""Small shared image-prep helpers used by multiple pipeline stages."""

from __future__ import annotations

from PIL import Image


def flatten_to_white(img: Image.Image) -> Image.Image:
    """Composite an RGBA image onto a white background, dropping alpha.

    Balloon assets in this dataset are ~88% fully-opaque (white fill + black ink) with a
    transparent margin only outside the wobbly drawn outline shape (masking the rectangular image
    bounds down to the hand-drawn silhouette) -- naively calling `.convert("RGB")` keeps whatever
    garbage RGB values sit under the transparent pixels (often black) instead of white, which
    visibly corrupts the image corners. Always flatten through this function, never convert("RGB")
    directly on one of these assets.
    """
    if img.mode != "RGBA":
        return img.convert("RGB")
    flattened = Image.new("RGB", img.size, (255, 255, 255))
    flattened.paste(img, mask=img.split()[3])
    return flattened
