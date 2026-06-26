from pathlib import Path

from PIL import Image

from data.make_sanity_dataset import save_class_images
from data.validate_dataset import class_names, iter_images, validate_image


def test_make_sanity_dataset_writes_images(tmp_path):
    out = tmp_path / "raw_sanity"
    save_class_images(out, "distress", 3, 64)
    save_class_images(out, "no_distress", 3, 64)

    distress = list((out / "distress").glob("*.jpg"))
    no_distress = list((out / "no_distress").glob("*.jpg"))

    assert len(distress) == 3
    assert len(no_distress) == 3

    with Image.open(distress[0]) as image:
        assert image.size == (64, 64)


def test_validate_dataset_helpers(tmp_path):
    split_dir = tmp_path / "train"
    class_dir = split_dir / "distress"
    class_dir.mkdir(parents=True)
    image_path = class_dir / "sample.jpg"
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(image_path)

    assert class_names(split_dir) == ["distress"]
    assert list(iter_images(split_dir)) == [image_path]
    ok, message = validate_image(image_path)
    assert ok is True
    assert message == ""
