import torch
import numpy as np
from Dataset import RandomImagePixelationDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



def stack_with_padding(batch_as_list: list):
    # Expected list elements are 4-tuples:
    # (pixelated_image, known_array, target_array, image_file)
    n = len(batch_as_list)
    shapes = []
    pixelated_images = []
    known_arrays = []
    target_arrays = []
    original_images = []

    for image, pixelated_image, known_array, target_array in batch_as_list:
        pixelated_images.append(pixelated_image)
        known_arrays.append(known_array)
        target_arrays.append(torch.from_numpy(target_array))
        original_images.append(torch.from_numpy(image))
        shapes.append(pixelated_image.shape)

    max_shape = np.max(np.stack(shapes, axis=0), axis=0)
    stacked_pixelated_images = np.zeros(shape=(n, *max_shape))
    stacked_known_arrays = np.zeros(shape=(n, *max_shape), dtype=bool)
    stacked_target_arrays = np.zeros(shape=(n, *max_shape))

    for i in range(n):
        channels, height, width = pixelated_images[i].shape
        stacked_pixelated_images[i, :channels, :height, :width] = pixelated_images[i]
        stacked_known_arrays[i, :channels, :height, :width] = known_arrays[i]
        coords = np.argwhere(known_arrays[i])
        _, x_min, y_min = np.min(coords, axis=0)
        _, x_max, y_max = np.max(coords, axis=0)
        stacked_target_arrays[i, :channels, x_min:x_max + 1, y_min:y_max + 1] = target_arrays[i]

    return torch.stack(original_images), torch.from_numpy(stacked_pixelated_images), torch.from_numpy(
        stacked_known_arrays), torch.from_numpy(stacked_target_arrays)


ds = RandomImagePixelationDataset(
    r"C:\Users\abhir\Desktop\Assignments\Python\Python II\Project 3\Test",
    width_range=(4, 32),
    height_range=(4, 32),
    size_range=(4, 16)
)
dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=stack_with_padding)
for (images, pixelated_images, known_arrays, target_arrays) in dl:
    fig, axes = plt.subplots(nrows=dl.batch_size, ncols=4)
    for i in range(dl.batch_size):
        axes[i, 0].imshow(images[i].squeeze(0), cmap="gray", vmin=0, vmax=255)
        axes[i, 1].imshow(pixelated_images[i].squeeze(0), cmap="gray", vmin=0, vmax=255)
        axes[i, 2].imshow(known_arrays[i].squeeze(0), cmap="gray", vmin=0, vmax=1)
        axes[i, 3].imshow(target_arrays[i].squeeze(0), cmap="gray", vmin=0, vmax=255)
    fig.tight_layout()
    plt.show()

