  Readme 

CNN to depixelate parts of image
================================

### The Jupyter Notebook file is the main code file. It includes a walkthrough of how the code was written and visualizations to help understand the process.

#### Please note: The file paths are absolute in all code files. Kindly change them to the appropriate file paths for reproduction. Please also note that cell 75 in the Jupyter notebook has a KeyboardInterrupt. This is because the cell is for visualization only and would otherwise visualize every image in the dataset.

First and foremost, the original images were converted to 64x64 grayscale images using the to\_grayscale.py file. This resulted in less computation times later.

The [grayscale.py](http://grayscale.py) and prepare\_image.py files contain the respective functions as defined in the original assignments. The [Dataset.py](http://Dataset.py) file contains the custom dataset with the only difference from the original being that this version returns the original image as well.

The stacking function returns the stacked original images, pixelated images and known arrays. These are ultimately not transformed per se, since all of them have the same shape 64x64. They are, however, stacked. The file also returns the stacked target arrays that are different from the original implementation, This version returns a 64x64 black image with the target array broadcasted onto it.

The [CNN.py](http://CNN.py) file includes the implementation of the CNN we use for this project and is just a very basic model. The file also prints the number of total and trainable parameters.

We now come to the [main.py](http://main.py) file. This is responsible for the actual implementation. The code files as well as the jupyter file explains the minutae of the code. However, what this file essentially does is create a dataset according to our custom dataset, then creates dataloaders, using the stacking function as the collate\_fn. An instance of CNN is created with the specified hyperparameters. Then the model is trained on our images. The loss is subsequently plotted to give an idea about the performace. Finally, the parameters are stored in a file.

In the [serailize.py](http://serailize.py) file, we take the saved model, initialize a model with the _same_ hyperparameters as the saved model. We then collect the predicted values for the pixelated regions for the test set and store them in a list. We encode the output using the serialize function provided to us. Additionally, we iterate over the test set again and then collect the entire image image to be able to compare what the final output looks like with respect to the input. We take a random image for visualization every time.

This wraps up the implementation of the project. The jupyter notebook contains comments at every step of the way that may be able to shed more light on how the code works.

Additionally, the model is also included since it would take a long time to train otherwise.