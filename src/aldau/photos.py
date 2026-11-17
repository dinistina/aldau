"""Duplicate-photo search with a perceptual hash (pHash).

Scammers often reuse the same photos (stolen from other ads) in many ads.
A normal file hash (md5) changes completely if the picture is resized or
re-compressed, so it cannot find such copies. A *perceptual* hash describes what
the picture looks like, so near-identical pictures get near-identical hashes.

pHash in five steps:
1. shrink the picture to 32x32 and make it grey — small details and colour go away;
2. run a 2D DCT (discrete cosine transform) — it turns the picture into
   "frequencies": the top-left numbers describe the big shapes, the rest describe
   fine details and noise;
3. keep only the top-left 8x8 block = 64 numbers about the overall structure;
4. compare each number with the median of the block: bigger -> bit 1, else bit 0;
5. the 64 bits are the hash.

Two pictures are compared by the Hamming distance: the number of differing bits
(0 = same picture, ~32 = unrelated pictures). A small distance (<= 8 of 64) means
"probably the same photo".

Written by hand with Pillow only (no numpy), so it runs everywhere; it is a
textbook pHash, not bit-compatible with the `imagehash` library.
"""

import math
from functools import cache
from pathlib import Path

from PIL import Image

HASH_SIZE = 8  # 8x8 = 64-bit hash
IMAGE_SIZE = 32  # the picture is shrunk to 32x32 before the DCT
MAX_DISTANCE = 8  # at most 8 different bits of 64 = "same photo"


@cache
def _cosines(n: int, keep: int) -> list[list[float]]:
    """cos((2x + 1) * u * pi / 2n) for u < keep, x < n — the DCT-II basis, computed once."""
    return [[math.cos((2 * x + 1) * u * math.pi / (2 * n)) for x in range(n)] for u in range(keep)]


def _dct_top_left(pixels: list[list[float]], keep: int) -> list[list[float]]:
    """The top-left keep x keep block of the 2D DCT of a square picture.

    A 2D DCT is a 1D DCT over every row, then over every column of the result.
    We only need `keep` frequencies in each direction, so only those are computed.
    (Scaling constants are skipped: they do not change which values are above the median.)
    """
    n = len(pixels)
    cos = _cosines(n, keep)
    # 1D DCT of every row (only the first `keep` frequencies)
    rows = [[sum(row[x] * cos[u][x] for x in range(n)) for u in range(keep)] for row in pixels]
    # 1D DCT of every column of that result
    return [
        [sum(rows[y][u] * cos[v][y] for y in range(n)) for u in range(keep)] for v in range(keep)
    ]


def phash_image(image: Image.Image) -> int:
    """64-bit perceptual hash of a Pillow image."""
    small = image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    flat = list(small.tobytes())  # mode "L": one byte (0-255) per pixel, row by row
    pixels = [flat[i * IMAGE_SIZE : (i + 1) * IMAGE_SIZE] for i in range(IMAGE_SIZE)]
    block = [value for row in _dct_top_left(pixels, HASH_SIZE) for value in row]
    # The [0][0] value is the average brightness: huge and not about shape, so it
    # is left out of the median (but kept as a bit, as in the classic algorithm).
    middle = sorted(block[1:])[len(block[1:]) // 2]
    bits = 0
    for value in block:
        bits = (bits << 1) | (1 if value > middle else 0)
    return bits


def phash_file(path: str | Path) -> int:
    with Image.open(path) as image:
        return phash_image(image)


def hamming(a: int, b: int) -> int:
    """Number of bits that differ between two hashes."""
    return bin(a ^ b).count("1")


def find_duplicate_photos(
    ads: list, max_distance: int = MAX_DISTANCE
) -> list[tuple[str, str, int]]:
    """Pairs of *different* ads that share a (near-)identical photo.

    Returns (ad id, other ad id, distance), closest pairs first. Photos that cannot
    be opened are skipped. Every photo is compared with every other one — O(n^2),
    which is fine for a few thousand photos.
    """
    hashes = []  # (ad id, hash)
    for ad in ads:
        for photo in ad.photos:
            try:
                hashes.append((ad.id, phash_file(photo)))
            except OSError:  # missing or broken file
                continue

    pairs: dict[tuple[str, str], int] = {}
    for i, (id_a, hash_a) in enumerate(hashes):
        for id_b, hash_b in hashes[i + 1 :]:
            if id_a == id_b:
                continue  # two photos of one ad
            distance = hamming(hash_a, hash_b)
            if distance <= max_distance:
                key = tuple(sorted((id_a, id_b)))
                pairs[key] = min(distance, pairs.get(key, distance))
    return sorted(((a, b, d) for (a, b), d in pairs.items()), key=lambda pair: pair[2])
