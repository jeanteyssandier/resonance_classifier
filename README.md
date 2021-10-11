# resonance classifier


This repository contains a python script that uses a convolutional neural network to classify a planetary system as resonant or not. The network uses images of resonant angles to build a model.

## Requirements

You may want to use [REBOUND](https://github.com/hannorein/rebound) and [REBOUNDx](https://github.com/dtamayo/reboundx) to run N-body simulations:
```shell
pip install rebound
pip install reboundx
```

## The data

The model directory contains a python script that runs N-body simulations using REBOUND and REBOUNDx. In these simulations, planets with various masses and initial positions migrate at various speed in a disc, for various amounts of times, with various output time samplings. This creates a whole zoology of resonant or near resonant systems. The resonance closest to the planet is identify, and the associated resonant angles are plotted as a function of time, as low-resolution black and white squares with no axes. Angles associated with the second nearest resonance are also plotted to generate more data.

I carried 1200 such simulations. I hand-picked the output pictures and put them in three folders: "0", "180" and "nothing" according to whether the angle librates around 0, 180, or does not librate.

Run setup_data.py to prepare this data for the neural network ([taken from here](https://towardsdatascience.com/all-the-steps-to-build-your-first-image-classifier-with-code-cf244b015799)).

## The trainer

A convolutional neural network (CNN) is then applied to the data. It learns from the images themselves, not from the time series. It uses the early EarlyStopping feature to stop once a good enough model has been found. See model/trainer.ipynb.

## Application

In the res_finder.ipynb notebook, a system of 2 planets is initiated near the 3:2 resonance. The script finds the nearest potential resonances (here the 3:2, 11:7 and 14:9), plot their resonant angles, run the image through the CNN, and deduce if the angle is librating or not, and if so, around which value (currently only 0 or 180).
