from Dataset import RandomImagePixelationDataset
from torch.utils.data import DataLoader, random_split
import torch
from stacking import stack_with_padding
from CNN import SimpleCNN
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

'''Unfortunately, plotting the images one after another while training or after training is not possible here
 since it was done using IPython in the Jupyter notebook and I have not tested it in Pycharm. As a result, I am 
 excluding the section that visualizes several images at a time. Matplotlib seems to be severely limited in this 
 capacity and either doesn't show all images or takes an immense amount of time while trying to visualize the images.'''

'''The python notebook is where all the code was written and tested. The .py files contain essentially the same code
but has not been tested. Please note that the file paths are absolute here as well.'''


path_ = r"C:\Users\abhir\Desktop\Assignments\Python\Python Project\testing"
ds = RandomImagePixelationDataset(
    path_,
    width_range=(4, 32),
    height_range=(4, 32),
    size_range=(4, 16)
)

train_size = int(0.8*(len(ds)))
eval_size = len(ds) - train_size
train_set, validation_set = random_split(ds, [train_size, eval_size], generator=torch.Generator().manual_seed(0))
batch_size = 32
train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=stack_with_padding)
valid_dl = DataLoader(validation_set, batch_size=batch_size, shuffle=False, collate_fn=stack_with_padding)

import torch

target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def training_loop(model, dataloader):  # take the model and dataloader as parameters
    model.train()  # set the model to training state
    current_loss = 0
    for image, pixelated_image, known_array, target_array in dataloader:  # iterate over dataloader
        known_array = known_array.to(target_device)  # send the known array to the target device
        pixelated_image = pixelated_image.to(target_device)  # send the pixelated image to the target device
        known_array_reverse = ~known_array.to(
            target_device)  # reverse the known and array and send it to the target device

        # The next line was used for the concatenating the channel dimensions of the pixelated image and the known array. However, this
        # resulted in some unexpected results and so, I commented it out.
        #         input_ = (torch.cat((pixelated_image, known_array), dim=1)).float().to(target_device)
        input_ = pixelated_image.float().to(target_device)  # the pixelated image again, just for naming conventions

        # I was curious to see if the model learnt better if only presented with the pixelated region instead of the entire entire image.
        # This expectedly resulted in bad predictions since the surrounding pixels are important for the predictions.
        #         input_ = input_*known_array

        input_ = torch.div(input_, 255.0)  # divide the input image by 255 to normalize it between [0,1]
        output_ = model(input_)  # send the normalized input to the model
        optimizer.zero_grad()  # sets the optimizer gradient to zero
        target = image.float().to(target_device)  # the original image is used as the target for the loss function;
        # renamed for convention
        output_ = torch.mul(output_, 255.0)  # denormalize the output to [0, 255]

        loss = torch.sqrt(loss_function(output_ * known_array, target * known_array))  # we take the square root of the
        # loss function (MSE). The output is
        # multiplied with the known array, and so is
        # the target (original image) to ensure only
        # the pixelated regions are compared.
        loss.backward()  # calulate the differential of the loss for the requires_grad=True model parameters
        optimizer.step()  # update the model parameters
        current_loss += loss.item()  # calculate the loss per batch
    train_loss = current_loss / len(dataloader.dataset)  # calculate average loss for one epoch
    print(f"Train Loss: {train_loss:.5f}")  # print the average loss up to 5 decimal precision

    return train_loss  # return the training loss


# Since the validating loop does not have any new code, the descriptions are not included with it.

def validating_loop(model, dataloader):
    model.eval()
    current_loss = 0
    with torch.no_grad():
        for image, pixelated_image, known_array, target_array in dataloader:
            known_array = known_array.to(target_device)
            pixelated_image = pixelated_image.to(target_device)
            known_array_reverse = ~known_array.to(target_device)
            #             input_ = (torch.cat((pixelated_image, known_array), dim=1)).float().to(target_device)
            input_ = pixelated_image.float().to(target_device)
            input_ = torch.div(input_, 255.0)
            #             input_ = input_*known_array
            output_ = model(input_)
            target = image.float().to(target_device)
            output_ = torch.mul(output_, 255.0)
            loss = torch.sqrt(loss_function(output_ * known_array, target * known_array))
            current_loss += loss.item()
        test_loss = current_loss / len(dataloader.dataset)
        print(f"Test Loss: {test_loss:.5f}")

        return test_loss


epochs = 100
train_loss = []
val_loss = []
model = SimpleCNN(1, 16, 7, True, 1, 3).to(target_device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_function = torch.nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    patience=3,
    factor=0.5,
    verbose=True
)
for epoch in tqdm(range(epochs)):
    print(f"Epoch {epoch + 1} of {epochs}")
    train_epoch_loss = training_loop(model, train_dl)
    val_epoch_loss = validating_loop(model, valid_dl)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    scheduler.step(val_epoch_loss)


def plot_losses(train_losses: list, eval_losses: list):
    fig, ax = plt.subplots()
    ax.plot(train_losses, label="Train loss")
    ax.plot(eval_losses, label="Eval loss")
    ax.legend()
    ax.set_xlim(0, len(train_losses) - 1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    plt.show()
    plt.close(fig)


plot_losses(train_loss, val_loss)

model_path = r"D:\Project\model_third.pt"
torch.save(model.state_dict(), model_path)
