# MNIST Image Classification 
Week-5 Assignment for ERAv2

## Overview

This repository uses MNIST dataset for classification of handwritten digits from 0 to 9. Convolutional neural network (CNN) has been used with PyTorch library. The project separates files for model definition, utility functions and the main training/testing scripts for clarity.

## Files Description

model.py: Defines the CNN architecture (Net class) used for digit classification.
  
utils.py: Utility functions for data transformation, training, and testing procedures.
  
s5_execution.ipynb: The main script that does data loading, model training, testing, and visualization of results.

## Setup and Requirements

Before running the project, ensure you have Python 3.x installed along with the following packages:

- PyTorch
- torchvision
- matplotlib
- tqdm

You can install the dependencies using the following command:

``pip install torch torchvision matplotlib tqdm``

## Usage

To use this project, follow these steps:

- Clone the repository to your local machine or Google Colab environment.
- Ensure all dependencies are installed.
- Execute different code blocks within `s5_execution.ipynb` in jypeter notebook or Google Colab.

