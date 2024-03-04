from glob import glob
from os import path
from torch.utils.data import Dataset
import numpy as np
from grayscale import to_grayscale
from prepare_image import prepare_image
from PIL import Image


class RandomImagePixelationDataset(Dataset):

    def __init__(
            self,
            image_dir,
            width_range: tuple[int, int],
            height_range: tuple[int, int],
            size_range: tuple[int, int]
    ):
        RandomImagePixelationDataset._check_range(width_range, "width")
        RandomImagePixelationDataset._check_range(height_range, "height")
        RandomImagePixelationDataset._check_range(size_range, "size")
        self.image_files = sorted(path.abspath(f) for f in glob(path.join(image_dir, "**", "*.jpg"), recursive=True))
        self.width_range = width_range
        self.height_range = height_range
        self.size_range = size_range

    @staticmethod
    def _check_range(r: tuple[int, int], name: str):
        if r[0] < 2:
            raise ValueError(f"minimum {name} must be >= 2")
        if r[0] > r[1]:
            raise ValueError(f"minimum {name} must be <= maximum {name}")

    def __getitem__(self, index):
        with Image.open(self.image_files[index]) as im:
            # im_shape = 64
            # resize = transforms.Resize(size=im_shape)
            # cCrop = transforms.CenterCrop(size=(im_shape, im_shape))
            # transform = transforms.Compose([resize, cCrop])
            # image_ = transform(im)
            image = np.array(im)
        image = to_grayscale(image)  # Image shape is now (1, H, W)
        image_width = image.shape[-1]
        image_height = image.shape[-2]

        # Create RNG in each __getitem__ call to ensure reproducibility even in
        # environments with multiple threads and/or processes
        rng = np.random.default_rng(seed=index)

        # Both width and height can be arbitrary, but they must not exceed the
        # actual image width and height
        width = min(rng.integers(low=self.width_range[0], high=self.width_range[1], endpoint=True), image_width)
        height = min(rng.integers(low=self.height_range[0], high=self.height_range[1], endpoint=True), image_height)

        # Ensure that x and y always fit with the randomly chosen width and
        # height (and not throw an error in "prepare_image")
        x = rng.integers(image_width - width, endpoint=True)
        y = rng.integers(image_height - height, endpoint=True)

        # Block size can be arbitrary again
        size = rng.integers(low=self.size_range[0], high=self.size_range[1], endpoint=True)

        pixelated_image, known_array, target_array = prepare_image(image, x, y, width, height, size)
        return image, pixelated_image, known_array, target_array

    def __len__(self):
        return len(self.image_files)
