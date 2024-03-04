import pickle
import torch
from tqdm import tqdm
import sys
import numpy as np
from CNN import SimpleCNN
from submission_serialization import serialize
import random
import matplotlib.pyplot as plt


np.set_printoptions(threshold=sys.maxsize)
objects = []
path_ = r"C:\Users\abhir\Desktop\Assignments\Python\Python Project\test_set.pkl"
with (open(path_, "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

pixelated_images = objects[0]["pixelated_images"]
known_arrays = objects[0]["known_arrays"]
model = SimpleCNN(1, 16, 7, True, 1, 3)
path_ = r"D:\Project\model_second.pt"
target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(path_))
model.eval()
model.to(target_device)
predictions = []

for pixelated_image, known_array in tqdm(zip(pixelated_images, known_arrays)):
    pixelated_image = torch.from_numpy(pixelated_image).to(target_device)
    known_array = torch.from_numpy(known_array).to(target_device)
    image = pixelated_image
    image = torch.div(image, 255.0)
#     image = torch.cat((image, ~known_array), dim=0)
    image = image.to(target_device)
    known_array_reverse = ~known_array
    coords = torch.argwhere(known_array_reverse)
    _, x_min, y_min = torch.min(coords, dim=0)[0]
    _, x_max, y_max = torch.max(coords, dim=0)[0]
    image = image.unsqueeze(0)
    output = torch.mul(model(image.float().to(target_device)), 255.0).reshape(64, 64)
#     output = torch.where(known_array == 0, output, pixelated_image)
    output = torch.masked_select(output, known_array_reverse)
    output = output.cpu().detach().numpy()
    predictions.append(output.astype("uint8"))

path_save = r"C:\Users\abhir\Desktop\Assignments\Python\Python Project" + "\\output.data"
serialize(predictions, path_save)



np.set_printoptions(threshold=sys.maxsize)
objects = []
path_ = r"C:\Users\abhir\Desktop\Assignments\Python\Python Project\test_set.pkl"
with (open(path_, "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

pixelated_images = objects[0]["pixelated_images"]
known_arrays = objects[0]["known_arrays"]
model = SimpleCNN(1, 16, 7, True, 1, 3)
path_ = r"D:\Project\model_third.pt"
target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(path_))
model.eval()
model.to(target_device)
predictions = []

for pixelated_image, known_array in tqdm(zip(pixelated_images, known_arrays)):
    pixelated_image = torch.from_numpy(pixelated_image).to(target_device)
    known_array = torch.from_numpy(known_array).to(target_device)
    image = pixelated_image
    image = torch.div(image, 255.0)
#     image = torch.cat((image, ~known_array), dim=0)
    image = image.to(target_device)
    known_array_reverse = ~known_array
    coords = torch.argwhere(known_array_reverse)
    _, x_min, y_min = torch.min(coords, dim=0)[0]
    _, x_max, y_max = torch.max(coords, dim=0)[0]
    image = image.unsqueeze(0)
    output = torch.mul(model(image.float().to(target_device)), 255.0).reshape(64, 64)
    output = torch.where(known_array == 0, output, pixelated_image)
#     output = torch.masked_select(output, known_array_reverse)
    output = output.cpu().detach().numpy()
    predictions.append(output.astype("uint8"))

rng_image = random.randint(0, 6000) # 6000 to make sure there is no index error
print(rng_image) # for reproducibility in case one wants to view the same image again
prediction = predictions[rng_image].reshape((64, 64))
image_ = pixelated_images[rng_image]
image_ = image_.squeeze(0)
fig, axes = plt.subplots(nrows=1, ncols=2)
images = []
axes[0].imshow(prediction, cmap="gray", vmin=0, vmax=255)
axes[1].imshow(image_, cmap="gray", vmin=0, vmax=255)
fig.tight_layout()
plt.show()