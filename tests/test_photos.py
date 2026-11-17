import random

import pytest
from PIL import Image, ImageDraw, ImageFilter

from aldau.collect import build_ad
from aldau.photos import find_duplicate_photos, hamming, phash_file, phash_image


def picture(seed: int, size=(400, 300)) -> Image.Image:
    """A random 'photo': coloured rectangles and circles on a background."""
    rng = random.Random(seed)
    image = Image.new("RGB", size, tuple(rng.randrange(256) for _ in range(3)))
    draw = ImageDraw.Draw(image)
    for _ in range(12):
        x0, y0 = rng.randrange(size[0]), rng.randrange(size[1])
        box = [x0, y0, x0 + rng.randrange(40, 200), y0 + rng.randrange(40, 200)]
        colour = tuple(rng.randrange(256) for _ in range(3))
        (draw.ellipse if rng.random() < 0.5 else draw.rectangle)(box, fill=colour)
    return image


def test_hamming():
    assert hamming(0b1011, 0b1011) == 0
    assert hamming(0b1011, 0b0010) == 2


def test_same_picture_same_hash():
    assert phash_image(picture(1)) == phash_image(picture(1))


@pytest.mark.parametrize(
    "change",
    [
        lambda im: im.resize((200, 150)),  # smaller copy
        lambda im: im.filter(ImageFilter.GaussianBlur(1)),  # slightly blurred
        lambda im: im.convert("L").convert("RGB"),  # black and white
    ],
)
def test_changed_copy_stays_close(change):
    original = picture(1)
    assert hamming(phash_image(original), phash_image(change(original))) <= 8


def test_jpeg_recompressed_copy_stays_close(tmp_path):
    original = picture(2)
    original.save(tmp_path / "a.png")
    original.save(tmp_path / "b.jpg", quality=40)  # heavy compression, as on marketplaces
    assert hamming(phash_file(tmp_path / "a.png"), phash_file(tmp_path / "b.jpg")) <= 8


def test_different_pictures_are_far_apart():
    distances = [hamming(phash_image(picture(s)), phash_image(picture(s + 100))) for s in range(5)]
    assert min(distances) > 12


def test_find_duplicate_photos_across_ads(tmp_path):
    picture(1).save(tmp_path / "stolen.jpg")
    picture(1).resize((300, 225)).save(tmp_path / "stolen_small.jpg", quality=60)
    picture(7).save(tmp_path / "own.jpg")
    picture(8).save(tmp_path / "own2.jpg")

    def ad(n, photos):
        ad = build_ad("olx", "телефоны", f"Объявление {n}", f"Текст {n}")
        return ad.model_copy(update={"photos": [str(tmp_path / p) for p in photos]})

    first = ad(1, ["stolen.jpg", "own.jpg"])
    second = ad(2, ["stolen_small.jpg", "missing.jpg"])  # a missing file is skipped
    third = ad(3, ["own2.jpg"])

    pairs = find_duplicate_photos([first, second, third])
    assert [tuple(sorted((a, b))) for a, b, _ in pairs] == [tuple(sorted((first.id, second.id)))]


def test_two_photos_of_one_ad_are_not_a_duplicate(tmp_path):
    picture(1).save(tmp_path / "a.jpg")
    picture(1).save(tmp_path / "b.jpg")
    ad = build_ad("olx", "телефоны", "Объявление", "Текст")
    ad = ad.model_copy(update={"photos": [str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg")]})
    assert find_duplicate_photos([ad]) == []
