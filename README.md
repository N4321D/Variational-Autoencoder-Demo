```
███╗   ██╗███████╗██╗   ██╗██████╗ ███████╗███████╗██╗     ███████╗ ██████╗████████╗  
████╗  ██║██╔════╝██║   ██║██╔══██╗██╔════╝██╔════╝██║     ██╔════╝██╔════╝╚══██╔══╝    
██╔██╗ ██║█████╗  ██║   ██║██████╔╝█████╗  █████╗  ██║     █████╗  ██║        ██║       
██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██╔══╝  ██╔══╝  ██║     ██╔══╝  ██║        ██║       
██║ ╚████║███████╗╚██████╔╝██║  ██║███████╗██║     ███████╗███████╗╚██████╗   ██║       
╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝     ╚══════╝╚══════╝ ╚═════╝   ╚═╝ INC.      
```
# VARIATIONAL AUTOENCODER DEMO

This repository contains a demo showcasing how autoencoders can be used to classify patterns in timeseries data, with a focus on variational autoencoders. It is also an excellent way to learn more about different model architecture, optimizers, activation functions and other aspects of machine learning because it show you live what happens in the latent layer each epoch. Thus, basically showing you how the model "learns".

The model uses an auto encoder to learn to classify random generated repeating patterns in timeseries data. There are plots that show the performance of the model on validation data live after each epoch. It is shown how the decoder regenerates output data and how the latent dimension looks like.

The structure of this notebook makes it very easy to change the encoder or decoder and see what the effect is of different layers types and architectures. Note that this model has been mainly build to demonstrate how VAEs perform and is not optimized for any real world usage. To do that you  would need to adapt and test it further. 

# Installation
## Colab
The notebook can be executed in Colab [**here**](https://drive.google.com/file/d/1C5FNazC6efC32GGIwB8_RS1ZeaL4gu1O/view?usp=drive_link)
 
## Local
But if you want to run it locally you can download the notebook here. 

### Docker container
To setup the right environment I recommend to use this container that we built if you have an NVIDIA GPU:  
[**Docker Hub**](https://hub.docker.com/repository/docker/n4321d/rapids-keras-torch-tf/general)

### Create you own enviroment
However if you feel uncomfortable running random containers (which you should), or dont have an NVIDIA GPU, you can easily create your own enviroment. These are the key packages you need:

```
keras == 3.4.1 
tensorflow == 2.16.2
holoviews == 1.19.1
panel == 1.4.4
numpy == 1.26.4
jupyterlab == 4.2.4 
```

Other versions might work as well but have not been tested.


# Usage

**TLDR:** press Run All -> Wait for userwarning to raise and stop execution -> Run last cell

**Full instructions:**  
Just run the notebook. At the end before you start the training it will raise a User warning:
```
UserWarning: Force stop here to give live image time to load, please run next cell manually after image is loaded

---------------------------------------------------------------------------
UserWarning                               Traceback (most recent call last)
Cell In[1], line 4
      1 # The warning raised here is to stop execution of the code until the image is loaded. If you run the next cell immediately the image will not update. 
      2 # Please wait until you see the image and then manually run the next cell with the train function
----> 4 raise UserWarning("Force stop here to give live image time to load, please run next cell manually after image is loaded")
```
This is because if we dont wait for the live plots to show, they wont update. 
I havent found a better solution than this:   
Wait for the plots to show and then run the last cell where you call train():
```
%%time
train(batchsize=256, epochs=600)
```
If anyone knows how to fix that, please create a pull request or post your suggestion.
